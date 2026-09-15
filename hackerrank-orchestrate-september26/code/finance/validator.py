import pandas as pd

from finance.simulator import (
    simulate_cash_flow,
)


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


def _safe_after_payment(
        simulation,
        payments,
        minimum_balance,
):
    payment_by_date = {}

    for payment_date, amount in payments:

        payment_date = _date(
            payment_date
        )

        if payment_date is None:
            return False

        payment_by_date.setdefault(
            payment_date,
            0.0,
        )

        payment_by_date[
            payment_date
        ] += float(amount)

    balance = None

    dates = simulation["date"].to_numpy()
    inflows = simulation["inflow"].to_numpy()
    outflows = simulation["outflow"].to_numpy()
    starting_balances = simulation["starting_balance"].to_numpy()

    minimum = float(minimum_balance)

    for i in range(len(simulation)):

        if balance is None:
            balance = float(starting_balances[i])

        balance += float(inflows[i])
        balance -= float(outflows[i])

        balance -= payment_by_date.get(
            dates[i],
            0.0,
        )

        if balance < minimum - 1e-9:
            return False

    return True

def _installment_payments(candidate):
    first_date = _date(
        candidate.get(
            "first_payment_date"
        )
    )

    number_of_payments = candidate.get(
        "number_of_payments"
    )

    frequency = candidate.get(
        "payment_frequency_days"
    )

    amount = candidate.get(
        "payment_amount"
    )

    if first_date is None:
        return []

    if (
            number_of_payments is None
            or pd.isna(number_of_payments)
    ):
        return []

    if (
            frequency is None
            or pd.isna(frequency)
    ):
        return []

    if (
            amount is None
            or pd.isna(amount)
    ):
        return []

    try:
        number_of_payments = int(
            number_of_payments
        )

        frequency = int(
            frequency
        )

        amount = float(
            amount
        )

    except (
            TypeError,
            ValueError,
    ):
        return []

    if number_of_payments <= 0:
        return []

    if frequency < 0:
        return []

    payments = []

    for i in range(
            number_of_payments
    ):

        payment_date = (
                first_date
                + pd.Timedelta(
            days=frequency * i
        ).to_pytimedelta()
        )

        payments.append(
            (
                payment_date,
                amount,
            )
        )

    return payments


def validate_candidate(
        state,
        request,
        candidate,
):
    request_date = _date(
        request[
            "request_date"
        ]
    )

    desired_date = _date(
        request[
            "desired_completion_date"
        ]
    )

    requested_amount = float(
        request[
            "requested_amount"
        ]
    )

    minimum_balance = float(
        state[
            "minimum_balance_to_keep"
        ]
    )

    method = candidate.get(
        "payment_method"
    )

    changes = candidate.get(
        "spending_changes_needed",
        [],
    )

    # =========================================================
    # NOT RECOMMENDED
    # =========================================================

    if method == "not_recommended":

        return (
            True,
            "fallback",
            candidate,
        )

    # =========================================================
    # FULL PAYMENT
    # =========================================================

    if method == "full_payment":

        payments = [
            (
                request_date,
                requested_amount,
            )
        ]

        simulation = simulate_cash_flow(
            state,
            request_date,
            days=90,
            spending_changes=changes,
        )

        if not _safe_after_payment(
                simulation,
                payments,
                minimum_balance,
        ):
            return (
                False,
                "full payment violates minimum balance",
                candidate,
            )

        if request_date > desired_date:
            return (
                False,
                "full payment is after desired date",
                candidate,
            )

        candidate[
            "payment_plan"
        ] = (
            f"{request_date.isoformat()}:"
            f"{_fmt_amount(requested_amount)}"
        )

        candidate[
            "completion_date"
        ] = request_date

        candidate[
            "first_payment_date"
        ] = request_date

        candidate[
            "number_of_payments"
        ] = 1

        candidate[
            "total_payable_amount"
        ] = requested_amount

        return (
            True,
            "safe",
            candidate,
        )

    # =========================================================
    # PARTIAL PAYMENT
    # =========================================================

    if method == "partial_payment":

        allows_partial = (
                str(
                    request[
                        "allows_partial_payment"
                    ]
                )
                .strip()
                .lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                }
        )

        if not allows_partial:
            return (
                False,
                "partial payment not allowed",
                candidate,
            )

        safe_amount = float(
            candidate.get(
                "amount_safe_to_pay",
                0.0,
            )
        )

        if not (
                0.0
                < safe_amount
                < requested_amount
        ):
            return (
                False,
                "invalid partial amount",
                candidate,
            )

        second_date = _date(
            candidate.get(
                "earliest_full_payment_date"
            )
        )

        if second_date is None:
            return (
                False,
                "no safe completion date",
                candidate,
            )

        if second_date > desired_date:
            return (
                False,
                "partial payment cannot complete by desired date",
                candidate,
            )

        remainder = (
                requested_amount
                - safe_amount
        )

        payments = [
            (
                request_date,
                safe_amount,
            ),
            (
                second_date,
                remainder,
            ),
        ]

        simulation = simulate_cash_flow(
            state,
            request_date,
            days=90,
            spending_changes=changes,
        )

        if not _safe_after_payment(
                simulation,
                payments,
                minimum_balance,
        ):
            return (
                False,
                "partial payment violates minimum balance",
                candidate,
            )

        candidate[
            "payment_plan"
        ] = (
            f"{request_date.isoformat()}:"
            f"{_fmt_amount(safe_amount)}|"
            f"{second_date.isoformat()}:"
            f"{_fmt_amount(remainder)}"
        )

        candidate[
            "completion_date"
        ] = second_date

        candidate[
            "first_payment_date"
        ] = request_date

        candidate[
            "number_of_payments"
        ] = 2

        candidate[
            "total_payable_amount"
        ] = requested_amount

        return (
            True,
            "safe",
            candidate,
        )

    # =========================================================
    # INSTALLMENTS
    # =========================================================

    if method == "installments":

        payments = _installment_payments(
            candidate
        )

        if not payments:
            return (
                False,
                "invalid installment option",
                candidate,
            )

        first_payment_date = (
            payments[0][0]
        )

        last_payment_date = (
            payments[-1][0]
        )

        if first_payment_date < request_date:
            return (
                False,
                "installment starts before request date",
                candidate,
            )

        if last_payment_date > desired_date:
            return (
                False,
                "installment plan completes too late",
                candidate,
            )

        max_months = int(
            state.get(
                "max_installment_months",
                0,
            )
        )

        if max_months > 0:

            maximum_date = (
                    request_date
                    + pd.Timedelta(
                days=31 * max_months
            ).to_pytimedelta()
            )

            if last_payment_date > maximum_date:
                return (
                    False,
                    "installment exceeds maximum months",
                    candidate,
                )

        simulation = simulate_cash_flow(
            state,
            request_date,
            days=90,
            spending_changes=changes,
        )

        if not _safe_after_payment(
                simulation,
                payments,
                minimum_balance,
        ):
            return (
                False,
                "installment violates minimum balance",
                candidate,
            )

        candidate[
            "payment_plan"
        ] = "|".join(
            f"{date.isoformat()}:"
            f"{_fmt_amount(amount)}"
            for date, amount
            in payments
        )

        # This is selected-plan completion date.
        # Do NOT overwrite earliest_full_payment_date.
        candidate[
            "completion_date"
        ] = last_payment_date

        candidate[
            "first_payment_date"
        ] = first_payment_date

        candidate[
            "number_of_payments"
        ] = len(payments)

        return (
            True,
            "safe",
            candidate,
        )

    # =========================================================
    # WAIT
    # =========================================================

    if method == "wait":

        full_date = _date(
            candidate.get(
                "earliest_full_payment_date"
            )
        )

        if full_date is None:
            return (
                False,
                "no safe future date",
                candidate,
            )

        if full_date > desired_date:
            return (
                False,
                "cannot complete by desired date",
                candidate,
            )

        # IMPORTANT:
        # amount_safe_to_pay remains the baseline safe amount.

        candidate[
            "payment_plan"
        ] = (
            f"{full_date.isoformat()}:"
            f"{_fmt_amount(requested_amount)}"
        )

        candidate[
            "completion_date"
        ] = full_date

        candidate[
            "first_payment_date"
        ] = full_date

        candidate[
            "number_of_payments"
        ] = 1

        candidate[
            "total_payable_amount"
        ] = requested_amount

        return (
            True,
            "safe",
            candidate,
        )

    return (
        False,
        "unknown payment method",
        candidate,
    )