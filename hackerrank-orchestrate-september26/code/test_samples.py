import math

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


def normalize(value):

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value):
            return None

        return round(
            value,
            2,
        )

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
    }:
        return None

    return text


def compare_number(
        expected,
        actual,
):
    if expected is None:
        return actual is None

    try:

        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=1e-7,
            abs_tol=0.01,
        )

    except Exception:
        return False


def main():

    data = load_data()

    samples = pd.read_csv(
        "dataset/sample_requests.csv"
    )

    output_columns = [
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    passed = 0

    print()
    print("=" * 40)
    print("     SAMPLE REQUEST EVALUATION")
    print("=" * 40)
    print()

    for _, expected in samples.iterrows():

        request_id = expected[
            "request_id"
        ]

        try:

            state = build_financial_state(
                expected["user_id"],
                request_id,
                data,
            )

            result = decide(
                state,
                expected,
            )

            failures = []

            for column in output_columns:

                expected_value = normalize(
                    expected[column]
                )

                actual_value = normalize(
                    result.get(column)
                )

                if column == "amount_safe_to_pay":

                    if not compare_number(
                            expected_value,
                            actual_value,
                    ):
                        failures.append(
                            (
                                column,
                                expected_value,
                                actual_value,
                            )
                        )

                else:

                    if (
                            expected_value
                            != actual_value
                    ):
                        failures.append(
                            (
                                column,
                                expected_value,
                                actual_value,
                            )
                        )

            if not failures:

                print(
                    f"{request_id:<15} PASS"
                )

                passed += 1

            else:

                print(
                    f"{request_id:<15} FAIL"
                )

                for (
                        column,
                        expected_value,
                        actual_value,
                ) in failures:

                    print(
                        f"    {column}"
                    )

                    print(
                        f"      expected: {expected_value}"
                    )

                    print(
                        f"      actual:   {actual_value}"
                    )

        except Exception as exc:

            print(
                f"{request_id:<15} ERROR"
            )

            print(
                f"    {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 40)

    print(
        f"RESULT: {passed}/{len(samples)} passed"
    )

    accuracy = (
            passed / len(samples) * 100
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()