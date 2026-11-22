# Lease and Finance Fraud Risk Model

Portfolio implementation based on production work experience. All data is synthetic, all code is original and clean-room, and no proprietary code, data, schema names, or internal system details from any employer are included.

**Original context:** Fraud prediction models used to support whether a customer can be leased or financed, in a financial services setting. Qualitative production claim only, no real customer data, features, thresholds or model details from production are reproduced here.

**This portfolio version:** A fully synthetic dataset and an interpretable logistic-style scoring model in pure Python, with calibration checks, fairness-aware feature review notes, and a clear human-in-the-loop decision policy. The model supports a decision, it does not make a final eligibility decision on its own.

## Run
```bash
python -m src.generate_data
python -m src.train
python -m src.score --income 72000 --delinquencies 0 --utilisation 0.31
python -m pytest tests/ -q
```
## Guardrails
- No protected attributes are used as features
- Every score ships with its top drivers so a human reviewer can override
- Thresholds in this demo are illustrative only
