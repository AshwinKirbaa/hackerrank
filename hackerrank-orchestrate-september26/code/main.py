import sys

import pandas as pd

from finance.decision import decide
from finance.state import build_financial_state


def load_data():
    return {
        "profiles":
            pd.read_csv(
                "dataset/financial_profiles.csv"
            ),
        "events":
            pd.read_csv(
                "dataset/financial_events.csv"
            ),
        "messages":
            pd.read_csv(
                "dataset/messages.csv"
            ),
        "payment_options":
            pd.read_csv(
                "dataset/request_payment_options.csv"
            ),
    }


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def main():
    data = load_data()

    requests = pd.read_csv("dataset/requests.csv")

    results = []

    for idx, request in requests.iterrows():
        request_id = request["request_id"]
        user_id = request["user_id"]

        try:
            state = build_financial_state(user_id, request_id, data)
            result = decide(state, request)
        except Exception as exc:
            print(
                f"ERROR {request_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            result = {
                "request_id": request_id,
                "amount_safe_to_pay": 0.0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": None,
                "spending_changes_needed": "none",
                "decision_explanation": "Unable to evaluate this request due to a processing error.",
            }

        results.append(result)

        n = idx + 1
        total = len(requests)
        if n % 10 == 0 or n == total:
            print(f"Processed {n}/{total} requests...", file=sys.stderr)

    output = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output.to_csv("dataset/output.csv", index=False)
    print(f"\nWrote {len(output)} rows to dataset/output.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
