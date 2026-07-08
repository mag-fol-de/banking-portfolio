# IFRS 9 Expected Credit Loss (ECL) Pipeline

End-to-end ECL calculation combining PD, LGD, EAD with IFRS 9 staging
and probability-weighted macro scenarios. Synthetic performing retail
portfolio of 50,000 loans and 3.78 billion EUR total EAD across five
products.

## Methodology

- **Portfolio**: mortgage, auto, personal loan, credit card, overdraft
- **IFRS 9 staging**:
  - Stage 1: performing, no significant deterioration, DPD < 30 -> 12m ECL
  - Stage 2: DPD 30-89 OR score drop >= 60 OR watchlist -> lifetime ECL
  - Stage 3: DPD >= 90 -> lifetime with PD = 1
- **Macro scenarios**: baseline (60%), adverse (30%), severe adverse (10%)
- **Lifetime PD**: cumulative under constant-hazard,
  `1 - (1 - PD_12m)^T` with T = 5y (term) or 3y (revolving)
- **Discounting**: 3% effective annual rate

## Headline results

- Baseline ECL: 77.5m EUR (2.05% coverage)
- Adverse ECL: 132.5m EUR (+71%)
- Severe adverse ECL: 217.9m EUR (+181%)
- **Weighted (final IFRS 9 provision): 108.0m EUR (2.86% coverage)**
- Stage split: Stage 1 (76.3%, 1.10% coverage), Stage 2 (23.0%, 7.81%),
  Stage 3 (0.8%, 28.05%)
- Stage-migration cliff: mean +1,546 EUR per loan if all stage-1 loans
  migrated to stage 2, total potential +59m EUR

## Files

- `generate_portfolio.py` --- synthetic performing portfolio + macro
  scenarios
- `compute_ecl.py` --- full pipeline (staging, lifetime PD, discounting,
  three scenarios, weighted total)
- `report.tex` / `report.pdf` --- full report
- `data/`, `outputs/`, `figures/`

## Reproducibility

```bash
python generate_portfolio.py
python compute_ecl.py
```
