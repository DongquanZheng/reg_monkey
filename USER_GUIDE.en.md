# Reg Monkey User Guide

This guide explains how to use Reg Monkey in the public demo or a local Streamlit run.

Reg Monkey helps you move through a structured empirical workflow: prepare data, inspect data quality, choose a model path, run analysis, read results, and export reports. It does not prove causality or replace research-design judgment.

## 1. Start the App

Open the hosted demo URL, or run locally with:

```bash
streamlit run app.py
```

If public demo mode is enabled, the app will show a public-demo notice. Use sample data first, and do not upload sensitive, confidential, or personal data.

## 2. Choose Data on Setup

You have two starting options:

- **Use sample data**: best for a first walkthrough. The sample datasets are synthetic and are meant to demonstrate the workflow.
- **Upload your own data**: upload a cleaned CSV or Excel file. Keep public-demo files small and non-sensitive.

After loading data, confirm preprocessing and variable roles. Variable roles tell the app which column is the outcome, which columns are predictors, and which columns are controls or identifiers.

## 3. Understand Data

The Understand Data page helps you check whether the dataset is ready for modeling.

Use it to review:

- row and column counts
- missingness
- variable summaries
- numeric and categorical variables
- correlation patterns
- resource or data-size warnings

If the data looks wrong, go back to Setup, adjust preprocessing, or upload a cleaner file.

## 4. Plan Analysis

The Plan page has two main paths:

- **Apply recommended settings and go to Run Analysis**: use the generated setup for the main model and continue.
- **Select manual configuration**: choose the model family and variables yourself.

DID, IV/2SLS, and PSM are guarded manual workflows. They are available for structured research designs, but Reg Monkey does not automatically confirm the assumptions behind them.

## 5. Configure Manual Models

Manual configuration lets you choose:

- model family
- outcome variable
- main predictor or treatment variable
- controls
- panel, time, group, instrument, or matching variables when required

The Run page will stay disabled or show risks if required fields are missing.

## 6. Run Analysis

Before running, check:

- selected model family
- included variables
- sample size
- pre-run risk warnings

Click **Run model** to execute the selected model. After a successful run, use **Continue to Interpret Results** to read the output.

## 7. Interpret Results

The Interpret Results page starts with a result snapshot, then shows beginner guidance, key findings, interpretation, diagnostics, and technical details.

Read results carefully:

- OLS focuses on coefficient direction, uncertainty, R², and diagnostics.
- Logit/Probit coefficients are not probability-point changes by default.
- Panel fixed effects use within-entity variation.
- DID requires research-design support, especially parallel-trends reasoning.
- IV/2SLS requires instrument relevance and exclusion-restriction judgment.
- PSM balances observed covariates only; unobserved confounding may remain.

Use diagnostics and cautions before sharing conclusions.

## 8. Export Reports

The Export page provides:

- brief report download
- full technical report download
- reproducibility pack download
- optional CSV downloads for results and cleaned data

Review reports before sharing. Public-demo exports should not be treated as proof of causality or as a substitute for review.

## 9. Switch Language

Use the sidebar language selector to switch between English and Chinese. The workflow state should remain available while text changes language.

## 10. Reset Session

Use **Reset session** when you want to start over. The confirmation step explains what will be cleared. Resetting clears the in-app analysis workflow state, but it does not delete local files on your computer.

## 11. Common File Issues

If a file cannot be loaded, Reg Monkey shows a friendly error near the upload area. Typical causes include:

- unsupported file type
- empty file
- malformed CSV or Excel file
- duplicate or blank column names
- no usable rows or columns
- no numeric variables for models that require numeric input

Fix the file and upload again, or use sample data.

## 12. Demo Safety Boundaries

- Use sample data first.
- Do not upload sensitive, confidential, or personal data to a public demo.
- Hosted demos have resource limits.
- Statistical estimates are not automatic causal conclusions.
- Review diagnostics, assumptions, and exports before sharing.
