# Usage Report — Buy or Wait? Final Full-Dataset Run

## Run Summary

- **Dataset**: `dataset/requests.csv` — 250 requests
- **Output**: `dataset/output.csv` — 250 rows
- **Entry point**: `python code/main.py`
- **Run mode**: Fully deterministic, no external API calls

## Model Providers and Names

This solution is **fully local and deterministic** — it uses no external language-model API calls, no cloud-hosted model inference, and no third-party AI services.

| Component | Technology |
|---|---|
| Financial state builder | Pure Python (`code/finance/state.py`) |
| Recurring-pattern detector | Statistical (median interval/amount, `code/finance/forecast.py`) |
| Cash-flow simulator | Deterministic day-by-day balance walk (`code/finance/simulator.py`) |
| Decision engine | Rule-based (`code/finance/decision.py`) |
| Message extractor | Regex + keyword matching (`code/finance/message_extractor.py`) |

## Model Calls

**Total model calls: 0**

No language-model inference was performed. All decisions are computed deterministically from the provided dataset files using rule-based logic and statistical pattern detection.

## Token Usage

| Metric | Value |
|---|---|
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Average tokens per request | 0 |

## Estimated Cost

**Total estimated cost: $0.00**

The solution requires only standard Python libraries (`pandas`, `numpy`) and produces fully deterministic output without any model inference costs.

## Runtime

- Typical full-dataset run time: ~60–90 seconds on a standard laptop
- Per-request average: ~0.3 seconds

## Setup and Run Instructions

```bash
# Install dependencies
pip install pandas numpy

# Run the solution
cd <repo-root>
python code/main.py

# Output written to: dataset/output.csv
```
