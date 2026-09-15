from __future__ import annotations

from datetime import timedelta

import pandas as pd


INVALID_STATUSES = {
    "cancelled",
    "failed",
    "unrealized",
}


def _to_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def _text(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip().lower()


def _event_status_valid(event):
    status = _text(
        event.get(
            "status",
            "",
        )
    )

    if status in INVALID_STATUSES:
        return False

    # Ignore pending credits.
    if (
            status == "pending"
            and _text(
        event.get(
            "direction",
            "",
        )
    )
            == "credit"
    ):
        return False

    return True


def _event_date(event):
    settlement = event.get(
        "settlement_date"
    )

    if (
            settlement is not None
            and not pd.isna(settlement)
    ):
        try:
            return pd.Timestamp(
                settlement
            ).normalize()
        except Exception:
            pass

    event_date = event.get(
        "event_date"
    )

    if (
            event_date is not None
            and not pd.isna(event_date)
    ):
        try:
            return pd.Timestamp(
                event_date
            ).normalize()
        except Exception:
            pass

    return None


def _normalise_events(events):
    """
    Convert event DataFrame into compact dictionaries.

    Uses itertuples() instead of iterrows().
    """

    if events is None or events.empty:
        return []

    records = []

    columns = list(
        events.columns
    )

    for row in events.itertuples(
            index=False,
            name=None,
    ):
        event = dict(
            zip(
                columns,
                row,
            )
        )

        if not _event_status_valid(
                event
        ):
            continue

        amount = event.get(
            "amount"
        )

        # Missing amounts are unresolved evidence.
        # Never treat them as zero.
        if (
                amount is None
                or pd.isna(amount)
        ):
            continue

        try:
            amount = float(
                amount
            )
        except (
                TypeError,
                ValueError,
        ):
            continue

        if amount <= 0:
            continue

        direction = _text(
            event.get(
                "direction",
                "",
            )
        )

        if direction not in {
            "credit",
            "debit",
        }:
            continue

        date = _event_date(
            event
        )

        if date is None:
            continue

        linked_event_id = event.get(
            "linked_event_id",
            "",
        )

        if (
                linked_event_id is None
                or pd.isna(linked_event_id)
        ):
            linked_event_id = ""

        records.append(
            {
                "date": date,
                "direction": direction,
                "amount": amount,
                "event_id": str(
                    event.get(
                        "event_id",
                        "",
                    )
                ),
                "category": _text(
                    event.get(
                        "category",
                        "",
                    )
                ),
                "event_type": _text(
                    event.get(
                        "event_type",
                        "",
                    )
                ),
                "flexibility": _text(
                    event.get(
                        "flexibility",
                        "",
                    )
                ),
                "linked_event_id": str(
                    linked_event_id
                ),
            }
        )

    return records


def _change_map(
        spending_changes
):
    stopped = set()
    reduced = {}

    for change in (
            spending_changes or []
    ):
        change = str(
            change
        ).strip()

        if change.startswith(
                "stop:"
        ):
            event_id = (
                change[5:].strip()
            )

            if event_id:
                stopped.add(
                    event_id
                )

        elif change.startswith(
                "reduce_to:"
        ):
            parts = change.split(
                ":",
                2,
            )

            if len(parts) != 3:
                continue

            event_id = (
                parts[1].strip()
            )

            try:
                amount = float(
                    parts[2]
                )
            except (
                    TypeError,
                    ValueError,
            ):
                continue

            if event_id:
                reduced[
                    event_id
                ] = max(
                    0.0,
                    amount,
                )

    return (
        stopped,
        reduced,
    )


def _apply_change_to_event(
        event,
        stopped,
        reduced,
):
    event_id = event[
        "event_id"
    ]

    linked_id = event[
        "linked_event_id"
    ]

    if (
            event_id in stopped
            or linked_id in stopped
    ):
        return 0.0

    if event_id in reduced:
        return reduced[
            event_id
        ]

    if linked_id in reduced:
        return reduced[
            linked_id
        ]

    return event[
        "amount"
    ]


def _build_daily_cashflows(
        events,
        start_date,
        end_date,
        spending_changes=None,
        recency_threshold=None,
        extra_events=None,
):
    records = _normalise_events(events)

    # Include forecasted future recurring events so the simulator
    # accounts for expenses not yet in the historical CSV.
    try:
        from finance.forecast import forecast_future_events
        days = (end_date - start_date).days + 1
        future_df = forecast_future_events(
            events,
            start_date,
            days=days,
            spending_changes=None,
            recency_threshold=recency_threshold,
            extra_events=extra_events,
        )
        if not future_df.empty:
            records.extend(_normalise_events(future_df))
    except Exception:
        pass

    stopped, reduced = _change_map(
        spending_changes
    )

    inflows = {}
    outflows = {}

    for event in records:

        date = event[
            "date"
        ]

        if (
                date < start_date
                or date > end_date
        ):
            continue

        amount = (
            _apply_change_to_event(
                event,
                stopped,
                reduced,
            )
        )

        if amount <= 0:
            continue

        if event[
            "direction"
        ] == "credit":

            inflows[date] = (
                    inflows.get(
                        date,
                        0.0,
                    )
                    + amount
            )

        else:

            outflows[date] = (
                    outflows.get(
                        date,
                        0.0,
                    )
                    + amount
            )

    return (
        inflows,
        outflows,
    )


def _get_request_date(
        state,
        payment_date=None,
):
    """
    Support both state formats:

    1. state["request"]["request_date"]
    2. state["request_date"]
    """

    if payment_date is not None:
        return pd.Timestamp(
            payment_date
        ).normalize()

    request = state.get(
        "request"
    )

    if (
            isinstance(
                request,
                dict,
            )
            and request.get(
        "request_date"
    ) is not None
    ):
        return pd.Timestamp(
            request[
                "request_date"
            ]
        ).normalize()

    if state.get(
            "request_date"
    ) is not None:
        return pd.Timestamp(
            state[
                "request_date"
            ]
        ).normalize()

    raise KeyError(
        "request_date"
    )


def _get_events(state):
    events = state.get(
        "events"
    )

    if events is None:
        return pd.DataFrame()

    return events


def simulate_cash_flow(
        state,
        payment_date=None,
        payment_amount=0.0,
        spending_changes=None,
        days=90,
        recency_threshold=None,
        extra_events=None,
):
    """
    Simulate cash flow for the next 90 days.

    The simulator does not use iterrows().
    """

    events = _get_events(
        state
    )

    start_date = (
        _get_request_date(
            state,
            payment_date,
        )
    )

    end_date = (
            start_date
            + timedelta(
        days=days - 1
    )
    )

    (
        inflows,
        outflows,
    ) = _build_daily_cashflows(
        events,
        start_date,
        end_date,
        spending_changes,
        recency_threshold=recency_threshold,
        extra_events=extra_events,
    )

    minimum_balance = _to_float(
        state.get(
            "minimum_balance_to_keep",
            0.0,
        )
    )

    balance = _to_float(
        state.get(
            "current_available_balance",
            0.0,
        )
    )

    payment_amount = max(
        0.0,
        _to_float(
            payment_amount
        ),
    )

    if payment_amount > 0:

        outflows[
            start_date
        ] = (
                outflows.get(
                    start_date,
                    0.0,
                )
                + payment_amount
        )

    rows = []

    current_date = (
        start_date
    )

    while (
            current_date
            <= end_date
    ):

        inflow = inflows.get(
            current_date,
            0.0,
        )

        outflow = outflows.get(
            current_date,
            0.0,
        )

        starting_balance = (
            balance
        )

        balance += inflow
        balance -= outflow

        rows.append(
            {
                "date": current_date,
                "starting_balance": (
                    starting_balance
                ),
                "inflow": inflow,
                "outflow": outflow,
                "ending_balance": balance,
                "safe": (
                        balance
                        >= minimum_balance
                        - 1e-9
                ),
            }
        )

        current_date += timedelta(
            days=1
        )

    return pd.DataFrame(
        rows
    )


def safe_amount_today(
        state,
        simulation=None,
):
    balance = _to_float(
        state.get(
            "current_available_balance",
            0.0,
        )
    )

    minimum = _to_float(
        state.get(
            "minimum_balance_to_keep",
            0.0,
        )
    )

    if simulation is None:
        simulation = simulate_cash_flow(
            state
        )

    if (
            simulation is None
            or simulation.empty
    ):
        return max(
            0.0,
            balance - minimum,
            )

    future_balances = (
        simulation[
            "ending_balance"
        ]
        .astype(float)
    )

    minimum_forecast_balance = (
        future_balances.min()
    )

    direct_buffer = (
            balance - minimum
    )

    forecast_buffer = (
            minimum_forecast_balance
            - minimum
    )

    return max(
        0.0,
        min(
            direct_buffer,
            forecast_buffer,
        ),
    )


def find_earliest_safe_date(
        state,
        requested_amount,
        simulation=None,
):
    requested_amount = max(
        0.0,
        _to_float(
            requested_amount
        ),
    )

    if simulation is None:
        simulation = simulate_cash_flow(
            state
        )

    if (
            simulation is None
            or simulation.empty
    ):
        return None

    minimum = _to_float(
        state.get(
            "minimum_balance_to_keep",
            0.0,
        )
    )

    balances = (
        simulation[
            "ending_balance"
        ]
        .astype(float)
        .to_numpy()
    )

    dates = (
        simulation[
            "date"
        ]
        .to_numpy()
    )

    suffix_minimum = [
                         0.0
                     ] * len(
        balances
    )

    running_minimum = float(
        "inf"
    )

    for i in range(
            len(balances) - 1,
            -1,
            -1,
    ):

        running_minimum = min(
            running_minimum,
            balances[i],
        )

        suffix_minimum[i] = (
            running_minimum
        )

    for i in range(
            len(balances)
    ):

        balance_after_payment = (
                balances[i]
                - requested_amount
        )

        if (
                balance_after_payment
                < minimum - 1e-9
        ):
            continue

        if (
                suffix_minimum[i]
                - requested_amount
                < minimum - 1e-9
        ):
            continue

        return pd.Timestamp(
            dates[i]
        ).normalize()

    return None


def payment_is_safe(
        state,
        payment_date,
        payment_amount,
        spending_changes=None,
        recency_threshold=None,
):
    simulation = simulate_cash_flow(
        state,
        payment_date=payment_date,
        payment_amount=payment_amount,
        spending_changes=spending_changes,
        recency_threshold=recency_threshold,
    )

    if (
            simulation is None
            or simulation.empty
    ):
        return False

    return bool(
        simulation[
            "safe"
        ].all()
    )