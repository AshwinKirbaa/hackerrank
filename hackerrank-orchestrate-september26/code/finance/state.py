import pandas as pd


def build_financial_state(user_id, request_id, data):
    profiles = data["profiles"]
    events = data["events"]
    messages = data["messages"]
    payment_options = data["payment_options"]

    profile_rows = profiles[
        profiles["user_id"] == user_id
        ]

    if profile_rows.empty:
        raise ValueError(
            f"User profile not found: {user_id}"
        )

    profile = profile_rows.iloc[0]

    user_events = events[
        events["user_id"] == user_id
        ].copy()

    user_messages = messages[
        (messages["user_id"] == user_id)
        & (
                (messages["request_id"] == request_id)
                | messages["request_id"].isna()
        )
        ].copy()

    request_payment_options = payment_options[
        payment_options["request_id"] == request_id
        ].copy()

    max_months = profile[
        "max_installment_months"
    ]

    if pd.isna(max_months):
        max_months = 0
    else:
        try:
            max_months = int(max_months)
        except (TypeError, ValueError):
            max_months = 0

    def text_value(column):
        value = profile[column]

        if pd.isna(value):
            return ""

        return str(value)

    return {
        "user_id": user_id,
        "request_id": request_id,

        "home_currency": str(
            profile["home_currency"]
        ),

        "current_available_balance": float(
            profile["current_available_balance"]
        ),

        "minimum_balance_to_keep": float(
            profile["minimum_balance_to_keep"]
        ),

        "financial_priorities": text_value(
            "financial_priorities"
        ),

        "expense_categories_to_protect": text_value(
            "expense_categories_to_protect"
        ),

        "expense_categories_user_is_willing_to_reduce":
            text_value(
                "expense_categories_user_is_willing_to_reduce"
            ),

        "expense_categories_user_is_willing_to_stop":
            text_value(
                "expense_categories_user_is_willing_to_stop"
            ),

        "payment_methods_user_will_consider":
            text_value(
                "payment_methods_user_will_consider"
            ),

        "max_installment_months": max_months,

        "events": user_events,

        "messages": user_messages,

        "payment_options":
            request_payment_options,
    }