import itertools

from datetime import timedelta

import pandas as pd

from finance.simulator import (
    simulate_cash_flow,
    safe_amount_today,
    find_earliest_safe_date,
    payment_is_safe,
)
from finance.validator import (
    validate_candidate,
    _safe_after_payment,
)

# Patterns whose last occurrence was more than this many intervals ago are
# considered stale and excluded from the forecast. 1.5 means a missed cycle
# (e.g. October salary absent when requesting in November) marks the pattern stale.
_RECENCY_THRESHOLD = 1.5

def _allowed_payment_methods(state):
    """
    Return the payment methods the user is willing to consider.
    """

    value = state.get(
        "payment_methods_user_will_consider",
        "",
    )

    if value is None or pd.isna(value):
        return set()

    return {
        item.strip().lower()
        for item in str(value).split("|")
        if item.strip()
    }

def _date(value):
    if value is None or pd.isna(value):
        return None

    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _fmt_amount(value):
    value = float(value)

    text = (
        f"{value:.10f}"
        .rstrip("0")
        .rstrip(".")
    )

    if text == "-0":
        text = "0"

    return text


def _split_values(value):
    if value is None or pd.isna(value):
        return []

    return [
        x.strip().lower()
        for x in str(value).split("|")
        if x.strip()
    ]


def _category_allowed(
        state,
        category,
        action,
):
    category = str(
        category
    ).strip().lower()

    protected = set(
        _split_values(
            state.get(
                "expense_categories_to_protect",
                "",
            )
        )
    )

    if category in protected:
        return False

    if action == "stop":

        allowed = set(
            _split_values(
                state.get(
                    "expense_categories_user_is_willing_to_stop",
                    "",
                )
            )
        )

        return category in allowed

    if action == "reduce":

        allowed = set(
            _split_values(
                state.get(
                    "expense_categories_user_is_willing_to_reduce",
                    "",
                )
            )
        )

        return category in allowed

    return False


def _change_candidates(state):
    events = state.get("events")

    if events is None or events.empty:
        return []

    results = []
    seen = set()

    # Use itertuples instead of iterrows.
    for event in events.itertuples(
            index=False
    ):

        event_id = str(
            getattr(
                event,
                "event_id",
                "",
            )
        ).strip()

        if not event_id:
            continue

        if event_id in seen:
            continue

        seen.add(event_id)

        flexibility = str(
            getattr(
                event,
                "flexibility",
                "",
            )
        ).strip().lower()

        if flexibility not in {
            "stoppable",
            "reducible",
            "reducible_or_stoppable",
        }:
            continue

        category = str(
            getattr(
                event,
                "category",
                "",
            )
        ).strip().lower()

        amount = getattr(
            event,
            "amount",
            None,
        )

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

        # -----------------------------------------------------
        # STOP
        # -----------------------------------------------------

        if (
                flexibility
                in {
            "stoppable",
            "reducible_or_stoppable",
        }
                and _category_allowed(
            state,
            category,
            "stop",
        )
        ):
            results.append(
                {
                    "event_id": event_id,
                    "type": "stop",
                    "amount": amount,
                    "text": (
                        f"stop:{event_id}"
                    ),
                }
            )

        # -----------------------------------------------------
        # REDUCE
        # -----------------------------------------------------

        if (
                flexibility
                in {
            "reducible",
            "reducible_or_stoppable",
        }
                and _category_allowed(
            state,
            category,
            "reduce",
        )
        ):
            minimum = getattr(
                event,
                "minimum_allowed_amount",
                None,
            )

            if (
                    minimum is None
                    or pd.isna(minimum)
            ):
                minimum = 0.0

            else:
                try:
                    minimum = float(
                        minimum
                    )
                except (
                        TypeError,
                        ValueError,
                ):
                    minimum = 0.0

            if minimum < amount:
                results.append(
                    {
                        "event_id": event_id,
                        "type": "reduce",
                        "amount": minimum,
                        "text": (
                            f"reduce_to:"
                            f"{event_id}:"
                            f"{_fmt_amount(minimum)}"
                        ),
                    }
                )

    return results


def _spending_scenarios(
        state,
        request=None,
        limit=20,
):
    changes = _change_candidates(
        state
    )

    scenarios = [
        []
    ]

    if not changes:
        return scenarios

    # Challenge allows at most 3 spending changes.
    maximum = min(
        3,
        len(changes),
    )

    for count in range(
            1,
            maximum + 1,
    ):

        for combo in itertools.combinations(
                changes,
                count,
        ):

            ids = [
                item[
                    "event_id"
                ]
                for item in combo
            ]

            # Same event cannot be both stopped
            # and reduced.
            if len(ids) != len(
                    set(ids)
            ):
                continue

            scenario = [
                item[
                    "text"
                ]
                for item in combo
            ]

            scenarios.append(
                scenario
            )

            # Keep candidate explosion under control.
            if len(scenarios) >= limit:
                return scenarios

    # Remove duplicates.
    unique = []
    seen = set()

    for scenario in scenarios:

        key = tuple(
            sorted(
                scenario
            )
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            list(key)
        )

        if len(unique) >= limit:
            break

    return unique

def _candidate_completion_preview(
        candidate
):
    method = candidate.get(
        "payment_method"
    )

    if method == "full_payment":
        return candidate.get(
            "request_date"
        )

    if method == "wait":
        return candidate.get(
            "earliest_full_payment_date"
        )

    if method == "partial_payment":
        return candidate.get(
            "earliest_full_payment_date"
        )

    if method == "installments":

        first = candidate.get(
            "first_payment_date"
        )

        count = candidate.get(
            "number_of_payments"
        )

        frequency = candidate.get(
            "payment_frequency_days"
        )

        if (
                first is None
                or count is None
                or frequency is None
        ):
            return None

        try:
            return (
                    first
                    + pd.Timedelta(
                days=(
                             int(count) - 1
                     )
                     * int(frequency)
            ).to_pytimedelta()
            )
        except Exception:
            return None

    return None


def _prepare_candidate(
        candidate,
        request,
        state,
        baseline_safe_today,
        baseline_earliest_date,
):
    request_date = _date(
        request[
            "request_date"
        ]
    )

    requested_amount = float(
        request[
            "requested_amount"
        ]
    )

    desired_date = _date(
        request[
            "desired_completion_date"
        ]
    )

    candidate[
        "request_date"
    ] = request_date

    candidate[
        "requested_amount"
    ] = requested_amount

    candidate[
        "desired_completion_date"
    ] = desired_date

    # ---------------------------------------------------------
    # IMPORTANT OUTPUT SEMANTIC
    #
    # This is the maximum safe amount TODAY under the
    # baseline scenario, not after spending changes.
    # ---------------------------------------------------------

    candidate[
        "amount_safe_to_pay"
    ] = min(
        baseline_safe_today,
        requested_amount,
    )

    # ---------------------------------------------------------
    # IMPORTANT OUTPUT SEMANTIC
    #
    # Baseline earliest date, independent of selected plan.
    # ---------------------------------------------------------

    candidate[
        "earliest_full_payment_date"
    ] = baseline_earliest_date

    candidate[
        "completion_date"
    ] = _candidate_completion_preview(
        candidate
    )

    return candidate


def _candidate_rank(candidate):
    desired = candidate.get(
        "desired_completion_date"
    )

    completion = candidate.get(
        "completion_date"
    )

    on_time = (
            completion is not None
            and desired is not None
            and completion <= desired
    )

    changes = candidate.get(
        "spending_changes_needed",
        [],
    )

    total = float(
        candidate.get(
            "total_payable_amount",
            0.0,
        )
    )

    start = candidate.get(
        "first_payment_date"
    )

    if start is None:
        start = pd.Timestamp.max.date()

    count = candidate.get(
        "number_of_payments",
        1,
    )

    try:
        count = int(count)
    except Exception:
        count = 1

    option_id = candidate.get(
        "payment_option_id"
    )

    if option_id is None:
        option_rank = 10**9
    else:
        try:
            option_rank = int(
                str(option_id)
                .rsplit("_", 1)[-1]
            )
        except Exception:
            option_rank = 10**9

    return (
        0 if on_time else 1,
        0 if not changes else 1,
        total,
        start,
        count,
        option_rank,
    )


def _candidate_is_valid_fast(state, request, candidate, recency_threshold=None):
    """
    Validate whether a candidate is financially safe.
    Returns True if the candidate passes all checks.
    """
    method = candidate.get("payment_method")
    changes = candidate.get("spending_changes", [])
    minimum = float(state.get("minimum_balance_to_keep", 0.0))

    request_date_ts = pd.Timestamp(request["request_date"]).normalize()
    desired_date_ts = pd.Timestamp(request["desired_completion_date"]).normalize()
    requested_amount = float(request["requested_amount"])

    if method == "not_recommended":
        return True

    # Check completion date constraint
    completion = candidate.get("completion_date")
    if completion is not None:
        if not hasattr(completion, "normalize"):
            completion = pd.Timestamp(completion)
        if completion > desired_date_ts:
            return False

    if method == "full_payment":
        if request_date_ts > desired_date_ts:
            return False
        return payment_is_safe(
            state,
            payment_date=request_date_ts,
            payment_amount=requested_amount,
            spending_changes=changes,
            recency_threshold=recency_threshold,
        )

    if method == "wait":
        if completion is None:
            return False
        # For wait with no spending changes, find_earliest_safe_date already
        # validated this date using the correct baseline simulation.
        if not changes:
            return True
        # With spending changes, re-run from request_date to capture all
        # intervening cash flows correctly.
        days = max(90, (pd.Timestamp(completion) - request_date_ts).days + 5)
        simulation = simulate_cash_flow(
            state,
            payment_date=request_date_ts,
            payment_amount=0.0,
            spending_changes=changes,
            days=days,
            recency_threshold=recency_threshold,
        )
        return _safe_after_payment(
            simulation,
            [(completion, requested_amount)],
            minimum,
        )

    if method == "partial_payment":
        safe = float(candidate.get("amount_safe_to_pay", 0.0))
        if not (0.0 < safe < requested_amount - 1e-9):
            return False
        if completion is None:
            return False
        remainder = requested_amount - safe
        days = max(90, (completion - request_date_ts).days + 5)
        simulation = simulate_cash_flow(
            state,
            payment_date=request_date_ts,
            payment_amount=0.0,
            spending_changes=changes,
            days=days,
            recency_threshold=recency_threshold,
        )
        return _safe_after_payment(
            simulation,
            [(request_date_ts.date(), safe), (completion.date(), remainder)],
            minimum,
        )

    if method == "installments":
        option_dates = candidate.get("option_dates", [])
        option_amounts = candidate.get("option_amounts", [])
        if not option_dates:
            return False
        last_date = option_dates[-1]
        if not hasattr(last_date, "normalize"):
            last_date = pd.Timestamp(last_date)
        if last_date > desired_date_ts:
            return False
        max_months = int(state.get("max_installment_months", 0))
        if max_months > 0:
            max_date = request_date_ts + pd.Timedelta(days=31 * max_months)
            if last_date > max_date:
                return False
        days = max(90, (last_date - request_date_ts).days + 5)
        simulation = simulate_cash_flow(
            state,
            payment_date=request_date_ts,
            payment_amount=0.0,
            spending_changes=changes,
            days=days,
            recency_threshold=recency_threshold,
        )
        payments = list(zip(option_dates, option_amounts))
        return _safe_after_payment(simulation, payments, minimum)

    return False


def _candidate_rank_key(candidate, desired_date):
    """Rank key for sorting candidates — lower is better."""
    completion = candidate.get("completion_date")

    on_time = (
        completion is not None
        and desired_date is not None
        and (
            completion.date() if hasattr(completion, "date")
            else pd.Timestamp(completion).date()
        )
        <= (
            desired_date.date() if hasattr(desired_date, "date")
            else pd.Timestamp(desired_date).date()
        )
    )

    changes = candidate.get("spending_changes", [])
    n_changes = len(changes) if changes else 0

    total = float(candidate.get("total_paid", 0.0))

    # First payment date from payment_plan string
    start = desired_date  # fallback
    plan = candidate.get("payment_plan", "")
    if plan and plan != "none":
        try:
            first_entry = plan.split("|")[0]
            start = pd.Timestamp(first_entry.split(":")[0])
        except Exception:
            pass
    if start is None:
        start = pd.Timestamp.max

    count = candidate.get("number_of_payments", 1)
    try:
        count = int(count)
    except Exception:
        count = 1

    option_id = candidate.get("payment_option_id")
    status = candidate.get("status", "")
    # "affordable_now" (direct full_payment) ranks best; payment options come after.
    if option_id is None and status == "affordable_now":
        option_rank = -1
    elif option_id is None or option_id == "":
        option_rank = 10 ** 9
    else:
        try:
            option_rank = int(str(option_id).rsplit("_", 1)[-1])
        except Exception:
            option_rank = 10 ** 9

    return (
        0 if on_time else 1,
        n_changes,
        total,
        start,
        count,
        option_rank,
    )


def _fmt_expl(v):
    """Format currency amount for decision_explanation: comma-sep integer or 2dp."""
    v = float(v)
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _fmt_date_expl(d):
    """Format date for decision_explanation: '8 August 2025'."""
    if d is None:
        return "a future date"
    if hasattr(d, "date"):
        d = d.date()
    return f"{d.day} {d.strftime('%B %Y')}"


def _build_explanation(state, request, best, safe_today, earliest_full_date):
    """Generate a concise decision explanation."""
    method = best.get("payment_method")
    requested_amount = float(request["requested_amount"])
    desired_date = _date(request.get("desired_completion_date"))
    minimum = float(state.get("minimum_balance_to_keep", 0.0))
    currency = str(state.get("home_currency", "")).strip()
    changes = best.get("spending_changes", [])

    # Build spending-changes prefix for full_payment with plan
    def _changes_prefix():
        """Describe spending changes in natural language."""
        if not changes:
            return ""
        events = state.get("events")
        parts = []
        for chg in changes:
            if chg.startswith("stop:"):
                eid = chg[5:]
                desc = _event_description(events, eid)
                parts.append(("stop", desc, None))
            elif chg.startswith("reduce_to:"):
                bits = chg.split(":", 2)
                if len(bits) == 3:
                    eid, new_amt = bits[1], bits[2]
                    desc = _event_description(events, eid)
                    try:
                        new_amt_f = float(new_amt)
                    except ValueError:
                        new_amt_f = 0.0
                    parts.append(("reduce", desc, new_amt_f))
        if not parts:
            return ""
        sentences = []
        for kind, desc, amt in parts:
            if kind == "stop":
                sentences.append(f"Stop the {desc}")
            else:
                sentences.append(
                    f"reduce the {desc} to {currency} {_fmt_expl(amt)}"
                )
        # Capitalise the first word of the first sentence
        if sentences:
            joined = " and ".join(sentences)
            return joined[0].upper() + joined[1:] + ", then "
        return ""

    if method == "full_payment":
        prefix = _changes_prefix()
        if prefix:
            return (
                f"{prefix}pay {currency} {_fmt_expl(requested_amount)} today. "
                f"This leaves at least {currency} {_fmt_expl(minimum)} available."
            )
        if requested_amount < minimum:
            return (
                f"Pay {currency} {_fmt_expl(requested_amount)} today. "
                f"This keeps the {currency} {_fmt_expl(minimum)} minimum available over the next 90 days."
            )
        return (
            f"Pay {currency} {_fmt_expl(requested_amount)} today. "
            f"This leaves at least {currency} {_fmt_expl(minimum)} available over the next 90 days."
        )

    if method == "installments":
        # Parse from payment plan
        plan = best.get("payment_plan", "")
        option_amounts = best.get("option_amounts", [])
        option_dates = best.get("option_dates", [])
        n = best.get("number_of_payments", 3)
        if option_amounts:
            each = option_amounts[0]
        elif plan and plan != "none":
            try:
                each = float(plan.split("|")[0].split(":")[1])
            except Exception:
                each = requested_amount / n
        else:
            each = requested_amount / n
        if option_dates:
            start_date = option_dates[0]
        elif plan and plan != "none":
            try:
                start_date = pd.Timestamp(plan.split("|")[0].split(":")[0]).date()
            except Exception:
                start_date = None
        else:
            start_date = None
        date_str = _fmt_date_expl(start_date)
        return (
            f"Use {n} installments of {currency} {_fmt_expl(each)}, starting {date_str}. "
            f"This leaves at least {currency} {_fmt_expl(minimum)} available."
        )

    if method == "partial_payment":
        remainder = requested_amount - safe_today
        date_str = _fmt_date_expl(
            earliest_full_date.date()
            if hasattr(earliest_full_date, "date") else earliest_full_date
        )
        return (
            f"Pay {currency} {_fmt_expl(safe_today)} today and the remaining "
            f"{currency} {_fmt_expl(remainder)} on {date_str}. "
            f"This completes the full request and keeps the {currency} {_fmt_expl(minimum)} minimum protected."
        )

    if method == "wait":
        fd = earliest_full_date
        if fd is None:
            date_str = "a future date"
        else:
            fd_date = fd.date() if hasattr(fd, "date") else fd
            date_str = _fmt_date_expl(fd_date)
        return (
            f"Pay {currency} {_fmt_expl(requested_amount)} in full on {date_str}. "
            f"Paying earlier would take the balance below the {currency} {_fmt_expl(minimum)} minimum."
        )

    # not_recommended / not_affordable
    if safe_today > 0.01:
        return (
            f"Do not proceed with the {currency} {_fmt_expl(requested_amount)} request. "
            f"Although {currency} {_fmt_expl(safe_today)} is available today, "
            f"the full amount cannot be completed safely within 90 days."
        )
    desired_str = _fmt_date_expl(desired_date)
    return (
        f"Do not make this payment by {desired_str}. "
        f"None of the available options keeps the {currency} {_fmt_expl(minimum)} minimum protected."
    )


def _event_description(events, event_id):
    """Look up the description of an event by event_id."""
    if events is None or events.empty:
        return event_id
    matches = events[events["event_id"].astype(str) == str(event_id)]
    if matches.empty:
        return event_id
    desc = str(matches.iloc[0].get("description", event_id)).strip()
    return desc.lower() if desc else event_id


def decide(
    state,
    request,
    expected=None,
):
    """
    Main decision engine.

    Strategy:
        1. Run baseline simulation once.
        2. Calculate today's safe amount once.
        3. Calculate earliest baseline full-payment date once.
        4. Try normal payment candidates.
        5. Only search spending changes if needed.
        6. Rank valid candidates using challenge rules.
    """

    requested_amount = float(request["requested_amount"])

    request_date = pd.Timestamp(
        request["request_date"]
    ).normalize()

    desired_date = pd.Timestamp(
        request["desired_completion_date"]
    ).normalize()

    # ---------------------------------------------------------
    # MESSAGE-BASED ADJUSTMENTS
    # ---------------------------------------------------------

    try:
        from finance.message_extractor import extract_message_adjustments
        extra_income_patterns, suppressed_ids = extract_message_adjustments(state)
    except Exception:
        extra_income_patterns, suppressed_ids = [], set()

    # Spending changes that suppress message-identified pending income.
    # These are NOT reported as spending_changes_needed (they reflect facts,
    # not recommendations), but they do affect the baseline forecast.
    income_suppressions = [f"stop:{eid}" for eid in suppressed_ids]

    # Combine caller spending_changes + income suppressions for the baseline.
    baseline_changes = income_suppressions or None

    # ---------------------------------------------------------
    # BASELINE SIMULATION
    # Stale-pattern recency threshold: skip patterns whose last event is more
    # than 2 full intervals before the request date.  This prevents dormant
    # spending from artificially driving the forecast negative.
    # ---------------------------------------------------------

    baseline_simulation = simulate_cash_flow(
        state,
        payment_date=request_date,
        payment_amount=0.0,
        spending_changes=baseline_changes,
        recency_threshold=_RECENCY_THRESHOLD,
        extra_events=extra_income_patterns if extra_income_patterns else None,
    )

    safe_today = min(
        requested_amount,
        safe_amount_today(
            state,
            simulation=baseline_simulation,
        ),
    )

    earliest_full_date = find_earliest_safe_date(
        state,
        requested_amount,
        simulation=baseline_simulation,
    )

    # ---------------------------------------------------------
    # BASELINE CANDIDATES
    # ---------------------------------------------------------

    candidates = []
    allowed_methods = _allowed_payment_methods(state)

    def _fmt(amount):
        """Format plan amount: strip .00 but keep meaningful decimals."""
        text = f"{float(amount):.2f}"
        if text.endswith(".00"):
            return text[:-3]
        return text

    # ---------------------------------------------------------
    # FULL PAYMENT TODAY (direct, cheapest, no financing fee)
    # ---------------------------------------------------------

    if (
        "full_payment" in allowed_methods
        and safe_today >= requested_amount - 1e-9
    ):
        candidates.append(
            {
                "payment_method": "full_payment",
                "payment_plan": (
                    f"{request_date.strftime('%Y-%m-%d')}:"
                    f"{_fmt(requested_amount)}"
                ),
                "amount_safe_to_pay": requested_amount,
                "completion_date": request_date,
                "spending_changes": [],
                "total_paid": requested_amount,
                "number_of_payments": 1,
                "payment_option_id": None,
                "status": "affordable_now",
            }
        )

    # ---------------------------------------------------------
    # EXACT PAYMENT OPTIONS (installments and other methods)
    # Skip full_payment — handled directly above.
    # ---------------------------------------------------------

    options = state.get("payment_options")

    if options is not None and not options.empty:

        for row in options.itertuples(index=False):
            method = str(
                getattr(row, "payment_method", "")
            ).strip().lower()

            # full_payment is handled directly; skip here to avoid duplicates.
            if method == "full_payment":
                continue

            if method not in allowed_methods:
                continue

            option_id = str(
                getattr(row, "payment_option_id", "")
            )

            try:
                payment_amount = float(
                    getattr(row, "payment_amount")
                )
            except Exception:
                continue

            try:
                number_of_payments = int(
                    getattr(row, "number_of_payments")
                )
            except Exception:
                continue

            if payment_amount <= 0 or number_of_payments <= 0:
                continue

            try:
                first_payment = pd.Timestamp(
                    getattr(row, "first_payment_date")
                ).normalize()
            except Exception:
                continue

            freq_raw = getattr(row, "payment_frequency_days", 0)
            if freq_raw is None or (isinstance(freq_raw, float) and pd.isna(freq_raw)):
                frequency = 0
            else:
                try:
                    frequency = int(freq_raw)
                except Exception:
                    frequency = 0

            dates = [
                first_payment + timedelta(days=frequency * i)
                for i in range(number_of_payments)
            ]
            amounts = [payment_amount] * number_of_payments

            # Basic date-range check (financial safety checked later)
            if any(d < request_date or d > desired_date for d in dates):
                continue

            plan = "|".join(
                f"{d.strftime('%Y-%m-%d')}:{_fmt(a)}"
                for d, a in zip(dates, amounts)
            )

            candidates.append(
                {
                    "payment_method": method,
                    "payment_plan": plan,
                    "amount_safe_to_pay": safe_today,
                    "completion_date": dates[-1],
                    "spending_changes": [],
                    "total_paid": sum(amounts),
                    "number_of_payments": number_of_payments,
                    "payment_option_id": option_id,
                    "status": "affordable_with_plan",
                    "option_dates": dates,
                    "option_amounts": amounts,
                }
            )

    # ---------------------------------------------------------
    # WAIT
    # Eligible only when the user accepts full_payment (problem spec).
    # ---------------------------------------------------------

    if (
        "full_payment" in allowed_methods
        and earliest_full_date is not None
        and earliest_full_date <= desired_date
    ):
        candidates.append(
            {
                "payment_method": "wait",
                "payment_plan": (
                    f"{earliest_full_date.strftime('%Y-%m-%d')}:"
                    f"{_fmt(requested_amount)}"
                ),
                "amount_safe_to_pay": safe_today,
                "completion_date": earliest_full_date,
                "spending_changes": [],
                "total_paid": requested_amount,
                "number_of_payments": 1,
                "payment_option_id": None,
                "status": "affordable_later",
            }
        )

    # ---------------------------------------------------------
    # PARTIAL PAYMENT
    # ---------------------------------------------------------

    allows_partial = str(
        request.get("allows_partial_payment", "")
    ).strip().lower() in {"true", "1", "yes", "y"}

    if (
        allows_partial
        and "partial_payment" in allowed_methods
        and safe_today > 1e-9
        and safe_today < requested_amount - 1e-9
        and earliest_full_date is not None
        and earliest_full_date <= desired_date
    ):
        remainder = requested_amount - safe_today
        plan = (
            f"{request_date.strftime('%Y-%m-%d')}:{_fmt(safe_today)}|"
            f"{earliest_full_date.strftime('%Y-%m-%d')}:{_fmt(remainder)}"
        )
        candidates.append(
            {
                "payment_method": "partial_payment",
                "payment_plan": plan,
                "amount_safe_to_pay": safe_today,
                "completion_date": earliest_full_date,
                "spending_changes": [],
                "total_paid": requested_amount,
                "number_of_payments": 2,
                "payment_option_id": None,
                "status": "affordable_with_plan",
            }
        )

    # ---------------------------------------------------------
    # VALIDATE NORMAL CANDIDATES
    # ---------------------------------------------------------

    valid_candidates = []

    for candidate in candidates:
        if _candidate_is_valid_fast(
                state,
                request,
                candidate,
                recency_threshold=_RECENCY_THRESHOLD,
        ):
            valid_candidates.append(candidate)

    # ---------------------------------------------------------
    # SPENDING-CHANGE SCENARIOS
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Do this ONLY when baseline candidates cannot complete
    # the request.
    #
    # This is the biggest performance improvement compared
    # with the previous implementation.
    # ---------------------------------------------------------

    if not valid_candidates:
        scenarios = _spending_scenarios(
            state,
            request,
            limit=20,
        )

        for changes in scenarios:
            # Avoid evaluating an enormous number of combinations.
            if len(changes) > 3:
                continue

            # Full payment today under spending changes
            candidate = {
                "payment_method": "full_payment",
                "payment_plan": (
                    f"{request_date.strftime('%Y-%m-%d')}:"
                    f"{_fmt(requested_amount)}"
                ),
                "amount_safe_to_pay": safe_today,
                "completion_date": request_date,
                "spending_changes": changes,
                "total_paid": requested_amount,
                "number_of_payments": 1,
                "payment_option_id": None,
                "status": "affordable_with_plan",
            }

            if _candidate_is_valid_fast(
                    state,
                    request,
                    candidate,
                    recency_threshold=_RECENCY_THRESHOLD,
            ):
                valid_candidates.append(candidate)
                continue

            # Wait with spending changes (only if full_payment is accepted)
            if "full_payment" in allowed_methods:
                candidate = {
                    "payment_method": "wait",
                    "payment_plan": (
                        f"{earliest_full_date.strftime('%Y-%m-%d')}:"
                        f"{_fmt(requested_amount)}"
                        if earliest_full_date is not None
                        else "none"
                    ),
                    "amount_safe_to_pay": safe_today,
                    "completion_date": earliest_full_date,
                    "spending_changes": changes,
                    "total_paid": requested_amount,
                    "number_of_payments": 1,
                    "payment_option_id": None,
                    "status": "affordable_with_plan",
                }

                if (
                        earliest_full_date is not None
                        and earliest_full_date <= desired_date
                        and _candidate_is_valid_fast(
                    state,
                    request,
                    candidate,
                    recency_threshold=_RECENCY_THRESHOLD,
                )
                ):
                    valid_candidates.append(candidate)

    # ---------------------------------------------------------
    # FINAL FALLBACK
    # ---------------------------------------------------------

    if not valid_candidates:
        return {
            "request_id": request["request_id"],
            "amount_safe_to_pay": round(
                safe_today,
                2,
            ),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": (
                earliest_full_date.strftime("%Y-%m-%d")
                if earliest_full_date is not None
                else None
            ),
            "spending_changes_needed": "none",
            "decision_explanation": (
                "The requested amount cannot be paid safely "
                "within the allowed forecast and completion period "
                "while preserving the required minimum balance."
            ),
        }

    # ---------------------------------------------------------
    # RANK
    # ---------------------------------------------------------

    ranked = sorted(
        valid_candidates,
        key=lambda c: _candidate_rank_key(
            c,
            desired_date,
        ),
    )

    best = ranked[0]

    changes = best.get(
        "spending_changes",
        [],
    )

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": round(
            safe_today,
            2,
        ),
        "affordability_status": best["status"],
        "recommended_payment_method": best[
            "payment_method"
        ],
        "payment_plan": best[
            "payment_plan"
        ],
        "earliest_date_for_full_payment": (
            earliest_full_date.strftime("%Y-%m-%d")
            if earliest_full_date is not None
            else None
        ),
        "spending_changes_needed": (
            "|".join(changes)
            if changes
            else "none"
        ),
        "decision_explanation": (
            _build_explanation(
                state,
                request,
                best,
                safe_today,
                earliest_full_date,
            )
        ),
    }