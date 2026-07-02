# SEB PD Case Study

Full retail Probability of Default modelling pipeline built for a
recruitment case exercise. The input dataset (`case_data.csv`) was
provided as part of the exercise and is not included in this
repository; the code and report show the methodology and results.

## Methodology

- Temporal split: train 2008--2011, out-of-time test 2012--2013
- Information Value screening on 45 features
- Iterative Variance Inflation Factor removal for multicollinearity
- Forward stepwise selection with combined Gini + Brier score criterion
  (5-fold CV) to jointly optimise discrimination and calibration
- WoE encoding for the final logistic regression
- Model families compared: L2-regularised logistic, XGBoost, LightGBM
- Feature stability: PSI per feature between train and test periods
- Calibration curves per model on the OOT cohort

## Headline result

Out-of-time Gini around 0.89 on the case dataset. The dataset is
recruitment-case synthetic data, which typically contains stronger
predictors than production retail unsecured portfolios: the same
methodology on the sibling 01-pd-scorecard project (with realistic
signal-to-noise) lands at Gini 0.54.

## Files

- `pd_model.py` --- full pipeline from load to model comparison
- `report.pdf` --- final report
- `figures/` --- generated plots

## Note on running

The case data is not included, so `pd_model.py` cannot be executed as-is
from this repository. It is here as a record of methodology and code.
The sibling project `01-pd-scorecard` in this repository has a
self-contained synthetic-data generator and runs end-to-end.
