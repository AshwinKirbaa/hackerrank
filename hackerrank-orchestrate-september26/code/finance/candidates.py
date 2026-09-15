import pandas as pd


def _date(value):
    if value is None or pd.isna(value):
        return None

    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _parse_bool(value):
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def _split_values(value):
    if value is None or pd.isna(value):
        return []

    return [
        item.strip().lower()
        for item in str(value).split("|")
        if item.strip()
    ]


def generate_payment_candidates(
        state,
        request,
):
    requested_amount = float(
        request[
            "requested_amount"
        ]
    )

    request_date = _date(
        request[
            "request_date"
        ]
    )

    allowed_methods = _split_values(
        state.get(
            "payment_methods_user_will_consider",
            "",
        )
    )

    allows_partial = _parse_bool(
        request[
            "allows_partial_payment"
        ]
    )

    candidates = []

    # =========================================================
    # FULL PAYMENT
    # =========================================================

    if "full_payment" in allowed_methods:

        candidates.append(
            {
                "payment_method":
                    "full_payment",

                "payment_option_id":
                    None,

                "payment_amount":
                    requested_amount,

                "number_of_payments":
                    1,

                "first_payment_date":
                    request_date,

                "payment_frequency_days":
                    None,

                "financing_fee":
                    0.0,

                "total_payable_amount":
                    requested_amount,

                "payment_plan":
                    "none",

                "spending_changes_needed":
                    [],
            }
        )

    # =========================================================
    # PARTIAL PAYMENT
    # =========================================================

    if (
            allows_partial
            and "partial_payment"
            in allowed_methods
    ):

        candidates.append(
            {
                "payment_method":
                    "partial_payment",

                "payment_option_id":
                    None,

                "payment_amount":
                    requested_amount,

                "number_of_payments":
                    2,

                "first_payment_date":
                    request_date,

                "payment_frequency_days":
                    None,

                "financing_fee":
                    0.0,

                "total_payable_amount":
                    requested_amount,

                "payment_plan":
                    "none",

                "spending_changes_needed":
                    [],
            }
        )

    # =========================================================
    # INSTALLMENTS
    # =========================================================

    payment_options = state.get(
        "payment_options"
    )

    if (
            "installments"
            in allowed_methods
            and payment_options is not None
    ):

        if not isinstance(
                payment_options,
                pd.DataFrame,
        ):
            payment_options = pd.DataFrame(
                payment_options
            )

        if not payment_options.empty:

            options = payment_options[
                payment_options[
                    "payment_method"
                ]
                .astype(str)
                .str.lower()
                == "installments"
                ]

            for _, option in (
                    options.iterrows()
            ):

                first_date = _date(
                    option[
                        "first_payment_date"
                    ]
                )

                if first_date is None:
                    continue

                try:
                    number_of_payments = int(
                        option[
                            "number_of_payments"
                        ]
                    )
                except (
                        TypeError,
                        ValueError,
                ):
                    continue

                try:
                    payment_amount = float(
                        option[
                            "payment_amount"
                        ]
                    )
                except (
                        TypeError,
                        ValueError,
                ):
                    continue

                frequency = option[
                    "payment_frequency_days"
                ]

                if pd.isna(frequency):
                    frequency = None
                else:
                    try:
                        frequency = int(
                            frequency
                        )
                    except (
                            TypeError,
                            ValueError,
                    ):
                        continue

                financing_fee = option[
                    "financing_fee"
                ]

                if pd.isna(financing_fee):
                    financing_fee = 0.0
                else:
                    financing_fee = float(
                        financing_fee
                    )

                total_payable = option[
                    "total_payable_amount"
                ]

                if pd.isna(total_payable):
                    total_payable = (
                            payment_amount
                            * number_of_payments
                            + financing_fee
                    )
                else:
                    total_payable = float(
                        total_payable
                    )

                candidates.append(
                    {
                        "payment_method":
                            "installments",

                        "payment_option_id":
                            str(
                                option[
                                    "payment_option_id"
                                ]
                            ),

                        "payment_amount":
                            payment_amount,

                        "number_of_payments":
                            number_of_payments,

                        "first_payment_date":
                            first_date,

                        "payment_frequency_days":
                            frequency,

                        "financing_fee":
                            financing_fee,

                        "total_payable_amount":
                            total_payable,

                        "payment_plan":
                            "none",

                        "spending_changes_needed":
                            [],
                    }
                )

    # =========================================================
    # WAIT
    # =========================================================

    candidates.append(
        {
            "payment_method":
                "wait",

            "payment_option_id":
                None,

            "payment_amount":
                requested_amount,

            "number_of_payments":
                1,

            "first_payment_date":
                None,

            "payment_frequency_days":
                None,

            "financing_fee":
                0.0,

            "total_payable_amount":
                requested_amount,

            "payment_plan":
                "none",

            "spending_changes_needed":
                [],
        }
    )

    # =========================================================
    # NOT RECOMMENDED
    # =========================================================

    candidates.append(
        {
            "payment_method":
                "not_recommended",

            "payment_option_id":
                None,

            "payment_amount":
                0.0,

            "number_of_payments":
                0,

            "first_payment_date":
                None,

            "payment_frequency_days":
                None,

            "financing_fee":
                0.0,

            "total_payable_amount":
                0.0,

            "payment_plan":
                "none",

            "spending_changes_needed":
                [],
        }
    )

    return candidates