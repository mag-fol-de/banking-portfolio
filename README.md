# Credit Risk Portfolio

A collection of retail credit risk models built end-to-end on
synthetic data. Each project follows the methodology used by European
banks for IRB scorecards and IFRS 9 provisioning, kept as a prototype
rather than a production system.

## Projects

**[01-pd-scorecard](./01-pd-scorecard)**
Probability of Default scorecard for unsecured retail loans. WoE
encoding, Information-Value selection, L2-regularised logistic
regression, intercept calibration to long-run-average PD, mapped to a
9-grade master scale. Validated across ten origination vintages
spanning five years with an out-of-time backtest.

**[02-pd-model](./02-pd-model)**
Full retail PD pipeline built for a recruitment case exercise. IV
screening, iterative VIF removal, forward stepwise selection with a
combined Gini + Brier criterion, WoE encoding, and comparison across
logistic regression, XGBoost, and LightGBM. Out-of-time Gini around
0.89 on the case dataset. Case data not included.

**[03-lgd-model](./03-lgd-model)**
Loss Given Default scorecard on ~15,000 defaulted facilities across
mortgage, auto, personal secured, and unsecured collateral. Fractional
logit, OLS, and XGBoost compared. OOT MAE 0.100, RMSE 0.126, R² 0.72.
Downturn LGD uplift via P90 quantile per collateral bucket.

**[04-ead-model](./04-ead-model)**
Exposure at Default via CCF regression for ~20,000 revolving retail
facilities (credit cards, overdrafts, lines of credit). Fractional
logit chosen: CCF R² 0.31, derived EAD R² 0.98 (mean EUR
9,325 predicted vs 9,277 realised).

**[05-ecl-pipeline](./05-ecl-pipeline)**
End-to-end IFRS 9 Expected Credit Loss pipeline combining PD × LGD ×
EAD across three probability-weighted macro scenarios (baseline,
adverse, severe adverse) on a 50,000-loan performing portfolio with
3.78 billion EUR EAD. Stage classification, lifetime PD term
structure, discounting. Weighted provision 108m EUR (2.86%
coverage), stage-migration cliff +1,546 EUR per loan.

## Stack

- Python for pipelines and post-processing
- DuckDB for the local warehouse
- scikit-learn for the logistic regression
- Matplotlib for diagnostics
- uv for dependency management

## Running a project

Each subfolder has its own README with a run recipe. Typical flow:

```bash
cd 01-pd-scorecard
python generate_data.py
python train.py
```

## Data note

All datasets are synthetic and generated with fixed seeds for
reproducibility. There is no real customer, loan, or bureau data in
any project.
