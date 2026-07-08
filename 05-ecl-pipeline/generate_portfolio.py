"""Synthetic performing retail loan portfolio for ECL calculation.

Generates ~50,000 performing loans across secured (mortgage/auto) and
unsecured (personal/card/overdraft) products. Each loan has features
used by the PD/LGD/EAD models plus IFRS 9 staging signals (days past
due, watchlist status, credit-score deterioration since origination).

Outputs
-------
data/portfolio.csv    : one row per performing loan
data/macro_scenarios.csv : baseline / adverse / severe adverse macro paths
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260706)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

N_LOANS = 50_000


# -----------------------------------------------------------------------------
# Product mix
# -----------------------------------------------------------------------------
PRODUCT_MIX = {
    "mortgage":       0.35,
    "auto":           0.15,
    "personal_loan":  0.20,   # term unsecured
    "credit_card":    0.20,   # revolving
    "overdraft":      0.10,   # revolving
}
products = RNG.choice(list(PRODUCT_MIX.keys()), size=N_LOANS,
                      p=list(PRODUCT_MIX.values()))

is_revolving = np.isin(products, ["credit_card", "overdraft"])
is_secured   = np.isin(products, ["mortgage", "auto"])

# Collateral mapping used later by LGD lookup
collateral_map = {"mortgage": "mortgage", "auto": "auto",
                  "personal_loan": "unsecured",
                  "credit_card": "unsecured",
                  "overdraft": "unsecured"}
collateral = np.array([collateral_map[p] for p in products])


# -----------------------------------------------------------------------------
# Balance / limit
# -----------------------------------------------------------------------------
mu_bal = {"mortgage": 12.0, "auto": 9.9, "personal_loan": 9.2,
          "credit_card": 8.2, "overdraft": 7.9}
sd_bal = {"mortgage": 0.6, "auto": 0.45, "personal_loan": 0.5,
          "credit_card": 0.55, "overdraft": 0.55}
current_balance = np.array([
    RNG.lognormal(mean=mu_bal[p], sigma=sd_bal[p]) for p in products
])

# Limit only meaningful for revolving; set NaN otherwise
credit_limit = np.where(
    is_revolving,
    current_balance * np.clip(RNG.normal(loc=2.2, scale=0.5, size=N_LOANS), 1.05, 5.0),
    np.nan,
)

# For revolving, drawn = current_balance
drawn = np.where(is_revolving, current_balance, current_balance)

# LTV for secured products (loan / collateral value)
ltv = np.where(
    is_secured,
    np.clip(RNG.normal(loc=0.68, scale=0.14, size=N_LOANS), 0.15, 1.10),
    np.nan,
)


# -----------------------------------------------------------------------------
# Borrower features
# -----------------------------------------------------------------------------
borrower_age = np.clip(RNG.normal(42, 12, size=N_LOANS), 20, 80).astype(int)
income_annual = np.clip(RNG.lognormal(mean=10.6, sigma=0.4, size=N_LOANS), 12_000, 500_000)
employment = RNG.choice(["employed", "self_employed", "unemployed", "retired"],
                        p=[0.68, 0.14, 0.10, 0.08], size=N_LOANS)

# Credit-score at origination (300-850 FICO-like)
score_origination = np.clip(RNG.normal(loc=690, scale=70, size=N_LOANS), 300, 850)
# Current score with some drift
score_current = np.clip(score_origination + RNG.normal(loc=-5, scale=45, size=N_LOANS), 300, 850)

# Days past due (mostly 0 for performing)
dpd_probs = np.array([0.75, 0.15, 0.06, 0.03, 0.01])
dpd_buckets = np.array([0, 15, 30, 60, 89])
dpd_indices = RNG.choice(np.arange(len(dpd_buckets)), size=N_LOANS, p=dpd_probs)
dpd = dpd_buckets[dpd_indices] + RNG.integers(0, 5, size=N_LOANS)

# Watchlist flag: escalated risk marker
watchlist = (RNG.random(N_LOANS) < 0.05).astype(int)

# Behavioural: minimum payment ratio (for revolving)
min_payment_ratio = np.where(
    is_revolving,
    np.clip(RNG.beta(a=3, b=4, size=N_LOANS), 0.05, 0.98),
    np.nan,
)

# Utilisation at t_0 for revolving
util = np.where(
    is_revolving,
    np.clip(drawn / np.where(is_revolving, credit_limit, 1.0), 0.02, 0.99),
    np.nan,
)


# -----------------------------------------------------------------------------
# Simulate "true" 12-month PD as a function of features
# (In reality PD would come from a fitted PD model. We simulate the ground
#  truth here so downstream ECL numbers are internally consistent.)
# -----------------------------------------------------------------------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

score_z = (score_current - 690) / 70
income_z = (np.log(income_annual) - 10.6) / 0.4
age_z = (borrower_age - 42) / 12

emp_effect = np.where(employment == "unemployed", 1.1,
              np.where(employment == "self_employed", 0.3,
              np.where(employment == "retired", -0.2, 0.0)))

logit_pd_base = {
    "mortgage": -4.5,
    "auto": -3.8,
    "personal_loan": -3.2,
    "credit_card": -3.0,
    "overdraft": -2.7,
}
pd_intercept = np.array([logit_pd_base[p] for p in products])

logit_pd = (
    pd_intercept
    - 0.7 * score_z
    - 0.15 * income_z
    - 0.05 * age_z
    + emp_effect
    + 0.02 * dpd
    + 0.8 * watchlist
)

pd_12m = sigmoid(logit_pd)

# LGD lookup by collateral (long-run-average from LGD project)
lgd_lookup = {"mortgage": 0.22, "auto": 0.43, "unsecured": 0.71}
lgd = np.array([lgd_lookup[c] for c in collateral])

# EAD: for revolving, drawn + CCF * (limit - drawn) with product-specific CCF
ccf_lookup = {"credit_card": 0.59, "overdraft": 0.72, "line_of_credit": 0.76}
ead = np.where(
    is_revolving,
    drawn + np.array([ccf_lookup.get(p, 0.6) for p in products]) *
        (np.where(is_revolving, credit_limit - drawn, 0.0)),
    current_balance,
)


# -----------------------------------------------------------------------------
# IFRS 9 stage classification (per bank policy)
#   Stage 1: no significant deterioration, DPD < 30
#   Stage 2: significant deterioration OR DPD >= 30 (up to 89)
#             deterioration = score dropped by >= 60 pts OR watchlist=1
#   Stage 3: DPD >= 90 (defaulted / impaired). Not present in performing book
#             but we do not have any at this stage by construction.
# -----------------------------------------------------------------------------
score_drop = score_origination - score_current
significant_deterioration = (score_drop >= 60) | (watchlist == 1)

stage = np.full(N_LOANS, 1, dtype=int)
stage_2_mask = (dpd >= 30) | significant_deterioration
stage[stage_2_mask] = 2
# Stage 3: DPD >= 90 (rare for a performing snapshot; some go into 90+ dpd)
stage[dpd >= 90] = 3


# -----------------------------------------------------------------------------
# Assemble
# -----------------------------------------------------------------------------
df = pd.DataFrame({
    "loan_id": np.arange(N_LOANS),
    "product": products,
    "collateral_type": collateral,
    "is_revolving": is_revolving,
    "is_secured": is_secured,
    "current_balance_eur": np.round(current_balance, 2),
    "credit_limit_eur": np.round(credit_limit, 2),
    "drawn_eur": np.round(drawn, 2),
    "ltv": np.round(ltv, 4),
    "utilisation": np.round(util, 4),
    "min_payment_ratio": np.round(min_payment_ratio, 4),
    "borrower_age": borrower_age,
    "income_annual_eur": np.round(income_annual, 2),
    "employment_status": employment,
    "score_origination": np.round(score_origination, 0).astype(int),
    "score_current": np.round(score_current, 0).astype(int),
    "dpd": dpd.astype(int),
    "watchlist": watchlist,
    "pd_12m": np.round(pd_12m, 5),
    "lgd": np.round(lgd, 4),
    "ead_eur": np.round(ead, 2),
    "ifrs9_stage": stage,
})

df.to_csv(DATA / "portfolio.csv", index=False)


# -----------------------------------------------------------------------------
# Macro scenarios: baseline, adverse, severe adverse
# Multipliers applied to PD and LGD to derive scenario-conditional values
# -----------------------------------------------------------------------------
scenarios = pd.DataFrame({
    "scenario": ["baseline", "adverse", "severe_adverse"],
    "weight": [0.60, 0.30, 0.10],
    "pd_multiplier": [1.00, 1.60, 2.40],
    "lgd_multiplier": [1.00, 1.20, 1.50],
    "unemployment_pct": [7.5, 9.0, 11.5],
    "hpi_yoy_pct":     [2.5, -2.0, -6.0],
})
scenarios.to_csv(DATA / "macro_scenarios.csv", index=False)


# -----------------------------------------------------------------------------
# Sanity print
# -----------------------------------------------------------------------------
print(f"Generated {len(df):,} performing loans.")
print("\nProduct mix:")
print(df["product"].value_counts())
print("\nStage distribution:")
print(df["ifrs9_stage"].value_counts().sort_index())
print("\nMean 12m PD by product:")
print(df.groupby("product")["pd_12m"].mean().sort_values())
print("\nMean EAD (EUR) by product:")
print(df.groupby("product")["ead_eur"].mean().round(0))
print("\nMacro scenarios written:")
print(scenarios)
