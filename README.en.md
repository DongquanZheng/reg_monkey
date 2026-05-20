# Reg Monkey

Reg Monkey is a bilingual empirical analysis assistant for business, economics, and management research workflows.

It helps users inspect data, configure deterministic statistical models, read diagnostics, and export reports. It supports English and Chinese UI/report wording and includes synthetic sample datasets for demo workflows.

[中文说明](README.zh.md) | [User guide](USER_GUIDE.en.md)

![Reg Monkey setup screenshot](screenshots/readme/en_setup.png)

## What It Does

- Upload CSV or Excel data, or start with built-in synthetic sample datasets.
- Inspect data quality, missingness, variable roles, and resource warnings before modeling.
- Configure and run OLS, Logit, Probit, Panel Fixed Effects, DID, IV/2SLS, and PSM workflows.
- Keep DID, IV/2SLS, and PSM as guarded manual research-design workflows.
- Export brief reports, full technical reports, and reproducibility packs.

Reg Monkey does not prove causality, rank models automatically, or replace econometric judgment.

## Try It

For first-time users, start with the OLS synthetic business sample.

1. Load a sample dataset.
2. Confirm preprocessing and variable roles.
3. Inspect data quality.
4. Apply the recommended setup or configure a model manually.
5. Run the model.
6. Read the result and diagnostics.
7. Export a report or reproducibility pack.

![Reg Monkey result screenshot](screenshots/readme/en_interpret_ols.png)

See the full [English user guide](USER_GUIDE.en.md) for step-by-step instructions.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Demo Setup

Use `app.py` as the Streamlit entrypoint.

For public-demo mode, set:

```text
REG_MONKEY_PUBLIC_DEMO_MODE=true
```

No API keys, login system, database, telemetry service, or external hosted model service is required for the demo.

## Supported Model Families

| Family | Status | Notes |
|---|---:|---|
| OLS | Supported | Continuous-outcome regression with diagnostics and report export. |
| Logit | Supported | Binary-outcome model; raw coefficients are not probability-point changes by default. |
| Probit | Supported | Binary-outcome model with family-specific fit metrics. |
| Panel Fixed Effects | Supported | Uses within-entity variation; time-invariant differences are absorbed. |
| DID | Guarded manual workflow | Requires research-design confirmation, especially parallel-trends reasoning. |
| IV/2SLS | Guarded manual workflow | Requires instrument relevance and exclusion-restriction judgment. |
| PSM | Guarded manual workflow | Balances observed covariates only; unobserved confounding may remain. |

## Demo Boundaries

- Use synthetic sample data first.
- Do not upload sensitive, confidential, or personal data to a public demo.
- Hosted resource limits apply; use small cleaned files.
- Review exported content before sharing.

![Reg Monkey export screenshot](screenshots/readme/en_export.png)
