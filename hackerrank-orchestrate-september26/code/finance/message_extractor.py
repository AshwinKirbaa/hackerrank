"""
Extract financial adjustments from messages so the simulator can incorporate
information the recurring-pattern detector cannot see (salary changes,
pending payouts, new income starting on a specific date, etc.).

Returns:
  - extra_income_patterns: list of pattern dicts that get appended to the
    forecast (same schema as detect_recurring_events output)
  - suppressed_source_ids: set of event_ids whose future recurrence should be
    blocked (pending payouts, unconfirmed bonuses, etc.)

Deterministic regex + keyword matching.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_CURRENCIES = ("EUR", "IDR", "INR", "USD", "ZAR")


def _parse_date_from_text(text: str) -> date | None:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    m = re.search(
        r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{4})\b",
        text, re.IGNORECASE,
    )
    if m:
        day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = _MONTH_NAMES.get(month_name)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass
    return None


def _extract_currency_amount(text: str) -> tuple[str | None, float | None]:
    """Return (currency, amount) from text like 'EUR 1,422.85' or 'IDR 42750000'."""
    for cur in _CURRENCIES:
        m = re.search(
            r"\b" + cur + r"\s+([\d,]+(?:\.\d+)?)",
            text, re.IGNORECASE,
        )
        if m:
            try:
                return cur, float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None, None


def _infer_currency(state: dict) -> str:
    return str(state.get("home_currency", "")).upper() or "EUR"


def _build_income_pattern(
        description: str, amount: float, currency: str,
        last_date: date, interval_days: int,
) -> dict:
    return {
        "signature": ("payroll", description, "salary", "credit", currency),
        "last_date": last_date,
        "interval_days": interval_days,
        "amount": amount,
        "event_type": "payroll",
        "description": description,
        "category": "salary",
        "direction": "credit",
        "currency": currency,
        "flexibility": "fixed",
        "minimum_allowed_amount": None,
        "source_event_id": f"msg_salary_{hash(description) & 0xFFFF:04x}",
    }


def _all_credit_event_ids(events) -> list[str]:
    if events is None or events.empty:
        return []
    credits = events[events["direction"] == "credit"]
    return [str(r["event_id"]) for _, r in credits.iterrows()]


def _latest_credit_event(events):
    """Return the most-recent credit event row (Series) or None."""
    if events is None or events.empty:
        return None
    credits = events[events["direction"] == "credit"].sort_values("event_date")
    if credits.empty:
        return None
    return credits.iloc[-1]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _is_payout_pending(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in [
        "still pending",
        "not withdrawable",
        "hasn't been credited",
        "has not been credited",
        "still in payment processing",
        "payment processing",
        "reversal has not been posted",
        "reversal has not posted",
        "not yet cleared",
        "not been posted",
    ])


def _is_no_income(text: str) -> bool:
    tl = text.lower()
    return (
        any(kw in tl for kw in [
            "seasonal contract has ended",
            "no off-season income",
            "no renewal has been confirmed",
        ])
        or bool(re.search(r"no off.season income", tl))
    )


def _is_commission_pending(text: str) -> bool:
    tl = text.lower()
    return bool(re.search(
        r"(commission|bonus).{0,100}(still|pending|not yet|not.*approv)",
        tl,
    ))


def _extract_base_salary(text: str) -> tuple[str | None, float | None]:
    """Parse 'confirmed base salary is EUR 3,072' → (currency, amount)."""
    m = re.search(
        r"confirmed base salary is\s+(" + "|".join(_CURRENCIES) + r")\s+([\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if m:
        cur = m.group(1).upper()
        try:
            amt = float(m.group(2).replace(",", ""))
            return cur, amt
        except ValueError:
            pass
    return None, None


def extract_message_adjustments(
        state: dict,
) -> tuple[list[dict], set[str]]:
    """
    Read messages for this request and return:
      (extra_income_patterns, suppressed_source_ids)
    """
    messages = state.get("messages")
    events = state.get("events")

    extra_income_patterns: list[dict] = []
    suppressed_source_ids: set[str] = set()

    if messages is None or messages.empty:
        return extra_income_patterns, suppressed_source_ids

    msgs = messages.sort_values("sent_at") if "sent_at" in messages.columns else messages

    for _, msg in msgs.iterrows():
        msg_text = str(msg.get("message_text", ""))
        source = str(msg.get("source_type", "")).lower()
        tl = msg_text.lower()

        # --------------------------------------------------------------
        # Service-provider / financial-service / bank: payout/prize/reversal
        # pending → suppress only the directly related event (if any).
        # If there is no related_event_id, the pending payout hasn't appeared
        # in the events data at all, so there's nothing to suppress.
        # Never suppress unrelated salary/payroll credits.
        # --------------------------------------------------------------
        if source in {"service_provider", "financial_service", "bank"}:
            if _is_payout_pending(msg_text):
                related_eid = msg.get("related_event_id")
                if related_eid and not (isinstance(related_eid, float) and pd.isna(related_eid)):
                    suppressed_source_ids.add(str(related_eid))
            continue

        # --------------------------------------------------------------
        # Employer messages
        # --------------------------------------------------------------
        if source != "employer":
            continue

        # Seasonal contract ended → suppress all income
        if _is_no_income(msg_text):
            for eid in _all_credit_event_ids(events):
                suppressed_source_ids.add(eid)
            continue

        # Salary RESUMES on a specific date
        if "resumes" in tl or "resume" in tl:
            effective_date = _parse_date_from_text(msg_text)
            currency, amount = _extract_currency_amount(msg_text)
            currency = currency or _infer_currency(state)
            if amount and effective_date:
                pattern = _build_income_pattern(
                    description="Message-derived salary",
                    amount=amount,
                    currency=currency,
                    last_date=effective_date - timedelta(days=30),
                    interval_days=30,
                )
                extra_income_patterns.append(pattern)
            continue

        # First salary (new employee)
        if "first salary" in tl:
            effective_date = _parse_date_from_text(msg_text)
            currency, amount = _extract_currency_amount(msg_text)
            currency = currency or _infer_currency(state)
            if amount and effective_date:
                pattern = _build_income_pattern(
                    description="First-job payroll",
                    amount=amount,
                    currency=currency,
                    last_date=effective_date - timedelta(days=30),
                    interval_days=30,
                )
                extra_income_patterns.append(pattern)
            continue

        # Salary date shifted
        if re.search(r"salary.*expected\s+on|expected.*salary\s+on", tl):
            effective_date = _parse_date_from_text(msg_text)
            if effective_date:
                latest = _latest_credit_event(events)
                if latest is not None:
                    amount = float(latest["amount"]) if latest["amount"] else None
                    currency = str(latest.get("currency", "")).upper() or _infer_currency(state)
                    if amount:
                        suppressed_source_ids.add(str(latest["event_id"]))
                        pattern = _build_income_pattern(
                            description=str(latest.get("description", "Salary")),
                            amount=amount,
                            currency=currency,
                            last_date=effective_date - timedelta(days=30),
                            interval_days=30,
                        )
                        extra_income_patterns.append(pattern)
            continue

        # Confirmed base salary + pending commission
        # "Your confirmed base salary is EUR 3,072. The commission shown for open deals
        #  is still pending..." → use the confirmed base salary as the recurring pattern,
        #  suppress the current credit event (which might include commission).
        if _is_commission_pending(msg_text):
            base_currency, base_salary = _extract_base_salary(msg_text)
            if base_salary:
                latest = _latest_credit_event(events)
                if latest is not None:
                    suppressed_source_ids.add(str(latest["event_id"]))
                    latest_date = pd.Timestamp(latest["event_id"] if False else latest["event_date"]).date()
                    pattern = _build_income_pattern(
                        description=str(latest.get("description", "Salary")),
                        amount=base_salary,
                        currency=base_currency or _infer_currency(state),
                        last_date=latest_date,
                        interval_days=30,
                    )
                    extra_income_patterns.append(pattern)
            continue

        # Salary amount changed: "Your next salary is reduced to EUR 1422.85"
        # or Indonesian: "Gaji bulanan Anda naik menjadi IDR 42750000"
        currency, amount = _extract_currency_amount(msg_text)
        if amount:
            effective_date = _parse_date_from_text(msg_text)
            currency = currency or _infer_currency(state)
            latest = _latest_credit_event(events)
            if latest is not None:
                latest_date = pd.Timestamp(latest["event_date"]).date()
                suppressed_source_ids.add(str(latest["event_id"]))
                last_date = (
                    effective_date - timedelta(days=30)
                    if effective_date
                    else latest_date
                )
                pattern = _build_income_pattern(
                    description=str(latest.get("description", "Salary")),
                    amount=amount,
                    currency=currency,
                    last_date=last_date,
                    interval_days=30,
                )
                extra_income_patterns.append(pattern)

    return extra_income_patterns, suppressed_source_ids
