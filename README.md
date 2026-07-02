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
