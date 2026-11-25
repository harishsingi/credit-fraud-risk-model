# Credit Fraud Risk Model

An interpretable fraud-risk scoring model in pure Python (standard library
only, no ML framework), built on a fully synthetic dataset. It scores one
credit application at a time and ships every score with its top drivers,
so a human reviewer can see *why* a case scored the way it did and
override it. The model supports a decision; it does not make a final
eligibility decision on its own.

**Original context:** Fraud prediction models used to support whether a
customer can be leased or financed, in a financial services setting.
Qualitative production claim only. No real customer data, features,
thresholds, or model details from production are reproduced here, and no
proprietary code, data, schema names, or internal system details from any
employer are included.

**Synthetic-data disclaimer:** Every row in the demo dataset is generated
by a seeded random generator in `credit_fraud/data.py`. The weights are
hand-picked illustrative values, not fitted to real data, and the review
threshold is illustrative only. Metrics in `reports/metrics.json` are
computed at runtime from whatever dataset was loaded; they describe the
synthetic demo, not real-world performance, and will change if the seed,
row count, or config changes.

## Setup

Requires Python 3.10+ and nothing else at runtime (standard library only).

```bash
git clone <this repo>
cd credit-fraud-risk-model
python -m venv .venv
source .venv/bin/activate
pip install -e .            # makes the credit_fraud package + CLIs available
pip install -e .[dev]       # also installs pytest for the test suite
```

Without installing, every command below also works with
`PYTHONPATH=src` from the repo root.

## Run

```bash
# 1. Generate the synthetic dataset (2,000 rows, seed 11 by default)
python -m credit_fraud.generate_data

# 2. Load it (validating every row), evaluate the model, write metrics
python -m credit_fraud.train

# 3. Score one application; the JSON result is the only stdout output
python -m credit_fraud.score --income 72000 --delinquencies 0 --utilisation 0.31
```

Example score output:

```json
{
  "fraud_probability": 0.1539,
  "top_drivers": ["utilisation"],
  "review_required": false,
  "contributions": {"delinquencies": 0.0, "utilisation": 0.496, "low_income": 0.0},
  "model_version": "0.2.0"
}
```

After `train`, `reports/metrics.json` holds the confusion matrix,
precision, recall, accuracy, F1, and calibration buckets (predicted vs
observed fraud rate per probability bin), plus row counts and the config
used, so a run is reproducible from its own output file. Re-running any
command overwrites the same files, so the pipeline is idempotent.

## Test

```bash
python -m pytest tests/ -q        # needs the [dev] install
python -m unittest discover -s tests -v   # standard library only, no install needed
```

60 tests cover every module: model behaviour (including hand-computed
logit and confusion-matrix values), input and CSV-row validation,
feature boundaries, data generation determinism, quarantine handling,
evaluation math, config loading, and the three CLIs end to end.

## Configuration

All tunable values live in `src/credit_fraud/config.py` as a frozen
`ModelConfig` dataclass: model weights and intercept, the low-income
cutoff, the review threshold, the synthetic-data row count and seed, and
the default paths. Override them with a JSON file:

```bash
python -m credit_fraud.train --config my-config.json
```

Unknown keys and wrong-typed values are rejected with an error instead of
being silently ignored. Paths resolve against the project root, so the
CLIs work from any working directory; `CF_DATA_DIR` overrides the data
directory.

## Architecture

```
Application inputs
      |
      v
validation.py        strict range/type checks; bad values rejected, never coerced
      |
      v
features.py          raw inputs -> model features (delinquencies, utilisation,
                     low_income flag). No protected attributes are used.
      |
      v
model.py             logistic-style scorer; every score carries per-feature
                     contributions + top drivers + review flag
      |
      v
evaluate.py          confusion matrix, precision/recall/accuracy/F1,
                     calibration buckets (predicted vs observed)
```

| Module | Responsibility |
|---|---|
| `config.py` | All settings in one immutable dataclass; optional JSON overrides |
| `exceptions.py` | Typed errors: `ValidationError`, `RowValidationError`, `DataQualityError`, `ConfigurationError` |
| `logging_config.py` | Standard-library logging setup (stderr, levelled, named loggers) |
| `validation.py` | Single-application validation and per-row CSV validation |
| `features.py` | Feature building, the one place raw inputs become features |
| `model.py` | `FraudRiskModel` scorer + `ScoreResult` |
| `data.py` | Seeded synthetic generation; CSV load with quarantine |
| `evaluate.py` | Confusion matrix, metrics, calibration |
| `generate_data.py` / `train.py` / `score.py` | Thin CLIs (arg parsing + exit codes only) |

There is no network or external API call anywhere in this repo, so there
is nothing to retry. If a data source is ever added, retries with
backoff belong in that new module, not retrofitted here.

## Data quality

`train` loads the CSV through `load_applications`, which:

- rejects a file whose header is missing required columns, or that
  yields zero valid rows (`DataQualityError`, exit code 1);
- quarantines individual bad rows instead of crashing: missing values,
  unparseable numbers, out-of-range values, labels other than 0/1, and
  exact duplicates (first occurrence kept). Each quarantined row keeps
  its source line number and reason, and is written to
  `data/quarantine.csv` for review;
- logs rows read, valid, and quarantined on every run, and records the
  same counts in `reports/metrics.json`.

## Guardrails

- No protected attributes are used as features
- Every score ships with its top drivers so a human reviewer can override
- Thresholds in this demo are illustrative only
- No credentials, no hardcoded absolute paths, no company names,
  locations, or proprietary details anywhere in the code or docs

## Architecture — Before / After

Hover any component to see what it is and why it is needed.

![Before / After architecture](docs/before-after-elaborated.svg)
<details>
<summary><b>Scoring + calibration — click to expand the 4 steps</b></summary>

Collapsed, this is just: Application inputs -> Feature review -> Scoring model -> Calibration checks

1. Application inputs are taken for scoring, with no protected attributes used as features.
2. Features are reviewed for fairness before use, so the model stays explainable.
3. An interpretable scoring model produces each score, so it can be understood and audited.
4. Calibration checks run on scores, so the model stays honest over time.
</details>

### Earlier issues, risk, and impact after

| Aspect | Detail |
|---|---|
| Earlier issues | Every case reviewed by hand, so decisions were slow and inconsistent; application data scattered across sources, so no consistent inputs existed; no consistent score, so similar cases could get different outcomes; no explanation for a decision; no calibration checks; no audit trail. |
| Risk | Similar cases could get different outcomes; a decision that cannot be checked cannot be defended if challenged; scores could drift without anyone noticing. |
| Impact after | Every score is consistent and interpretable, shipped with its top drivers so a human reviewer sees why it scored that way; calibration checks keep scores honest over time; fairness and audit notes are kept with the model; a human reviewer still makes the final eligibility call. |
