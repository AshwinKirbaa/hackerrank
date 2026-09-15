from datetime import timedelta, date
from collections import defaultdict
import calendar

import pandas as pd


def _add_months(d, months):
    """Add `months` calendar months to date `d`, clamping day to month end."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _detect_monthly_dom(dates):
    """
    Return the consistent day-of-month if all dates share the same dom (±1),
    otherwise None. Used to enable calendar-month anchoring for monthly patterns.
    """
    if not dates:
        return None
    doms = [d.day for d in dates]
    dom_median = sorted(doms)[len(doms) // 2]
    if all(abs(d - dom_median) <= 1 for d in doms):
        return dom_median
    return None


IGNORED_STATUSES = {
    "cancelled",
    "failed",
    "unrealized",
}


def _date(value):
    if value is None or pd.isna(value):
        return None

    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _amount(value):
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_event(row):
    status = str(
        row.get("status", "")
    ).strip().lower()

    return status not in IGNORED_STATUSES


def _signature(row):
    return (
        str(row.get("event_type", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
        str(row.get("direction", "")),
        str(row.get("currency", "")),
    )


def _detect_interval(dates):
    if len(dates) < 3:
        return None

    dates = sorted(dates)

    intervals = []

    for i in range(1, len(dates)):
        diff = (
                dates[i] - dates[i - 1]
        ).days

        if diff > 0:
            intervals.append(diff)

    if len(intervals) < 2:
        return None

    median = float(
        pd.Series(intervals).median()
    )

    if median <= 0:
        return None

    good = [
        x
        for x in intervals
        if abs(x - median)
           <= max(
            3,
            median * 0.15
        )
    ]

    if len(good) < 2:
        return None

    return int(round(median))


def _detect_amount(amounts):
    if not amounts:
        return None

    median = float(
        pd.Series(amounts).median()
    )

    if median <= 0:
        return None

    deviations = [
        abs(x - median) / median
        for x in amounts
    ]

    if max(deviations) > 0.15:
        return None

    return median


def detect_recurring_events(events):
    if events is None or events.empty:
        return []

    groups = defaultdict(list)

    # Track the most recent credit event per broad type (event_type, direction, currency)
    # so we can detect when a recurring income pattern has been superseded.
    latest_credit_by_broad_type = {}

    for _, row in events.iterrows():

        if not _valid_event(row):
            continue

        direction = str(
            row.get(
                "direction",
                ""
            )
        ).lower()

        if direction not in {
            "credit",
            "debit",
        }:
            continue

        amount = _amount(
            row.get("amount")
        )

        date_value = _date(
            row.get("event_date")
        )

        if amount is None or date_value is None:
            continue

        groups[
            _signature(row)
        ].append(row)

        # Build the most-recent-credit index (broad type = event_type + direction + currency).
        if direction == "credit":
            broad_key = (
                str(row.get("event_type", "")).lower(),
                "credit",
                str(row.get("currency", "")).upper(),
            )
            existing_date = latest_credit_by_broad_type.get(broad_key, (None, None))[0]
            if existing_date is None or date_value > existing_date:
                latest_credit_by_broad_type[broad_key] = (date_value, _signature(row))

    patterns = []

    for signature, rows in groups.items():

        dates = []
        amounts = []

        for row in rows:

            d = _date(
                row.get("event_date")
            )

            a = _amount(
                row.get("amount")
            )

            if d is not None and a is not None:
                dates.append(d)
                amounts.append(a)

        if len(dates) < 3:
            continue

        interval = _detect_interval(
            dates
        )

        if interval is None:
            continue

        latest_row = max(
            rows,
            key=lambda r: _date(
                r.get("event_date")
            )
        )

        latest_date = _date(
            latest_row.get("event_date")
        )

        # Prefer consistent median; fall back to latest amount for
        # variable-amount recurring patterns (e.g. salary adjustments).
        median_amount = _detect_amount(amounts)
        if median_amount is None:
            latest_amount = _amount(latest_row.get("amount"))
            if latest_amount is None or latest_amount <= 0:
                continue
            median_amount = latest_amount

        direction_val = str(latest_row.get("direction", "")).lower()
        currency_val = str(latest_row.get("currency", "")).upper()
        event_type_val = str(latest_row.get("event_type", "")).lower()

        # Detect calendar-month anchoring: if interval is roughly monthly (28-32 days)
        # and all events share the same day-of-month, use month-addition instead of
        # fixed-day intervals to prevent drift (e.g. salary on the 15th every month).
        monthly_dom = None
        if 28 <= interval <= 32:
            monthly_dom = _detect_monthly_dom(dates)

        # For credit patterns: suppress if a newer RECURRING event of the same broad type
        # (event_type + direction + currency) exists with a DIFFERENT description.
        # This detects cases like "Final employer payroll" superseding "Payroll credit".
        # One-time events (< 3 occurrences) do NOT supersede a recurring pattern.
        if direction_val == "credit":
            broad_key = (event_type_val, "credit", currency_val)
            most_recent = latest_credit_by_broad_type.get(broad_key)
            if (
                most_recent is not None
                and most_recent[0] is not None
                and most_recent[0] > latest_date
                and most_recent[1] != signature
            ):
                superseding_sig = most_recent[1]
                if len(groups.get(superseding_sig, [])) >= 3:
                    continue

        patterns.append(
            {
                "signature": signature,
                "last_date": latest_date,
                "interval_days": interval,
                "monthly_dom": monthly_dom,
                "amount": median_amount,
                "event_type": event_type_val,
                "description": str(
                    latest_row.get(
                        "description",
                        ""
                    )
                ),
                "category": str(
                    latest_row.get(
                        "category",
                        ""
                    )
                ),
                "direction": direction_val,
                "currency": currency_val,
                "flexibility": str(
                    latest_row.get(
                        "flexibility",
                        "fixed"
                    )
                ).lower(),
                "minimum_allowed_amount":
                    _amount(
                        latest_row.get(
                            "minimum_allowed_amount"
                        )
                    ),
                "source_event_id": str(
                    latest_row.get(
                        "event_id",
                        ""
                    )
                ),
            }
        )

    return patterns


def forecast_future_events(
        events,
        start_date,
        days=90,
        spending_changes=None,
        recency_threshold=None,
        extra_events=None,
):
    if events is None or events.empty:
        return pd.DataFrame()

    start_date = pd.Timestamp(
        start_date
    ).date()

    end_date = (
            start_date
            + timedelta(days=days)
    )

    patterns = detect_recurring_events(
        events
    )

    # Drop stale patterns: skip if gap from last_date to start_date exceeds
    # recency_threshold * interval_days.  Prevents long-dormant spending from
    # distorting the forecast when the pattern has clearly stopped.
    if recency_threshold is not None:
        patterns = [
            p for p in patterns
            if (start_date - p["last_date"]).days
               <= recency_threshold * p["interval_days"]
        ]

    # Merge caller-supplied extra events (e.g. message-derived income) into the
    # recognised patterns so they participate in dedup and date generation.
    if extra_events:
        patterns = patterns + list(extra_events)

    if not patterns:
        return pd.DataFrame(
            columns=events.columns
        )

    stopped = set()
    reduced = {}

    if spending_changes:

        for change in spending_changes:

            parts = str(change).split(":")

            if (
                    len(parts) == 2
                    and parts[0] == "stop"
            ):
                stopped.add(parts[1])

            elif (
                    len(parts) == 3
                    and parts[0] == "reduce_to"
            ):
                try:
                    reduced[
                        parts[1]
                    ] = float(parts[2])
                except ValueError:
                    pass

    # Existing explicit future events.
    existing = set()

    for _, row in events.iterrows():

        if not _valid_event(row):
            continue

        event_date = _date(
            row.get("event_date")
        )

        if event_date is None:
            continue

        existing.add(
            (
                _signature(row),
                event_date,
            )
        )

    generated = []

    counter = 0

    for pattern in patterns:

        source_id = pattern[
            "source_event_id"
        ]

        # Determine the amount after a permitted change.
        forecast_amount = pattern[
            "amount"
        ]

        stopped_pattern = (
                source_id in stopped
        )

        if source_id in reduced:
            forecast_amount = reduced[
                source_id
            ]

        if stopped_pattern:
            continue

        if forecast_amount <= 0:
            continue

        monthly_dom = pattern.get("monthly_dom")

        if monthly_dom is not None:
            # Calendar-month anchoring: advance by one calendar month each step,
            # anchored to the consistent day-of-month. Prevents drift on months
            # with different lengths (e.g. salary on the 15th every month).
            last = pattern["last_date"]
            month_offset = 1
            next_date = _add_months(last, month_offset)
            # Force the day-of-month to the anchor dom (clamped to month end)
            import calendar as _cal
            _max_day = _cal.monthrange(next_date.year, next_date.month)[1]
            next_date = next_date.replace(day=min(monthly_dom, _max_day))
        else:
            next_date = (
                    pattern["last_date"]
                    + timedelta(
                days=pattern[
                    "interval_days"
                ]
            )
            )
            month_offset = None

        while next_date <= end_date:

            if next_date >= start_date:

                key = (
                    pattern["signature"],
                    next_date,
                )

                if key not in existing:

                    counter += 1

                    generated.append(
                        {
                            "event_id":
                                f"forecast_{counter}",

                            "user_id":
                                None,

                            "event_type":
                                pattern[
                                    "event_type"
                                ],

                            "description":
                                pattern[
                                    "description"
                                ],

                            "category":
                                pattern[
                                    "category"
                                ],

                            "direction":
                                pattern[
                                    "direction"
                                ],

                            "amount":
                                forecast_amount,

                            "currency":
                                pattern[
                                    "currency"
                                ],

                            "event_date":
                                next_date,

                            "settlement_date":
                                next_date,

                            "status":
                                "forecast",

                            "linked_event_id":
                                source_id,

                            "flexibility":
                                pattern[
                                    "flexibility"
                                ],

                            "minimum_allowed_amount":
                                pattern[
                                    "minimum_allowed_amount"
                                ],
                        }
                    )

                    existing.add(key)

            if monthly_dom is not None:
                import calendar as _cal
                month_offset += 1
                next_date = _add_months(pattern["last_date"], month_offset)
                _max_day = _cal.monthrange(next_date.year, next_date.month)[1]
                next_date = next_date.replace(day=min(monthly_dom, _max_day))
            else:
                next_date += timedelta(
                    days=pattern[
                        "interval_days"
                    ]
                )

    if not generated:
        return pd.DataFrame(
            columns=events.columns
        )

    return pd.DataFrame(
        generated
    )[events.columns]