"""Synthetic Exposure at Default (EAD) dataset generator.

Simulates ~20,000 revolving-credit facilities (credit cards, overdrafts,
lines of credit) observed for twelve months, of which a subset default
within the horizon. For each defaulted facility we record limit,
drawn balance at observation date, and drawn balance at default. The
target is the Credit Conversion Factor (CCF):

    CCF = (EAD - drawn_t0) / (limit - drawn_t0)

CCF is the fraction of the undrawn commitment that gets used before
default. Basel IRB requires CCF to be estimated for off-balance-sheet
exposures and is used to compute EAD:

    EAD = drawn + CCF * (limit - drawn)

Outputs
-------
data/ead_facilities.csv : one row per defaulted revolving facility with
    features, observed CCF, EAD, and vintage.
data/vintage_macro.csv : macro context per vintage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260705)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Vintage macro context
# -----------------------------------------------------------------------------
VINTAGES = ["2019H1", "2019H2", "2020H1", "2020H2", "2021H1",
            "2021H2", "2022H1", "2022H2", "2023H1", "2023H2"]

unemployment = np.array([6.8, 6.9, 9.2, 9.6, 8.7, 7.9, 7.4, 8.1, 7.6, 7.2])
gdp_yoy      = np.array([2.1, 1.9, -3.4, -2.1, 3.8, 4.5, 2.4, -0.8, 0.6, 1.4])

macro = pd.DataFrame({
    "vintage": VINTAGES,
    "unemployment_pct": unemployment,
    "gdp_yoy_pct": gdp_yoy,
})
macro.to_csv(DATA / "vintage_macro.csv", index=False)


# -----------------------------------------------------------------------------
# Defaulted revolving facilities
# -----------------------------------------------------------------------------
N_FACILITIES = 20_000

# Product mix
PRODUCT_MIX = {
    "credit_card": 0.55,
    "overdraft":   0.25,
    "line_of_credit": 0.20,
}
products = RNG.choice(list(PRODUCT_MIX.keys()), size=N_FACILITIES,
                      p=list(PRODUCT_MIX.values()))

# Vintage assignment
vintage_weights = np.array([0.06, 0.07, 0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.12])
vintage_ids = RNG.choice(np.arange(len(VINTAGES)), size=N_FACILITIES, p=vintage_weights)
vintages = np.array(VINTAGES)[vintage_ids]

# Credit limit by product (EUR)
mu_limit = {"credit_card": 8.5, "overdraft": 8.0, "line_of_credit": 10.4}
sd_limit = {"credit_card": 0.55, "overdraft": 0.5, "line_of_credit": 0.7}
limit = np.array([
    RNG.lognormal(mean=mu_limit[p], sigma=sd_limit[p]) for p in products
])

# Utilisation at observation date (12 months before default)
# Beta distribution with product-specific alpha/beta
def util_beta_params(product):
    if product == "credit_card":
        return (2.5, 3.5)   # mean ~0.42
    elif product == "overdraft":
        return (1.8, 3.2)   # mean ~0.36
    else:  # line_of_credit
        return (2.0, 4.0)   # mean ~0.33

util_t0 = np.array([
    RNG.beta(*util_beta_params(p)) for p in products
])
drawn_t0 = util_t0 * limit

# Time to default (months, uniform 3-12)
time_to_default = RNG.integers(low=3, high=13, size=N_FACILITIES)

# Borrower risk score at observation (0-1 higher = riskier)
risk_score_t0 = np.clip(RNG.beta(a=2.5, b=2.5, size=N_FACILITIES), 0.02, 0.98)

# Borrower age
borrower_age = np.clip(RNG.normal(42, 13, size=N_FACILITIES), 20, 80).astype(int)

# Behavioural feature: minimum payment ratio (0-1, higher = healthier)
min_payment_ratio = np.clip(RNG.beta(a=3, b=4, size=N_FACILITIES), 0.05, 0.95)


# -----------------------------------------------------------------------------
# Target: CCF (Credit Conversion Factor) and drawn at default
# -----------------------------------------------------------------------------
# Base CCF by product (empirical values roughly match retail literature)
base_ccf = {
    "credit_card":    0.55,
    "overdraft":      0.68,
    "line_of_credit": 0.72,
}
ccf_mean = np.array([base_ccf[p] for p in products])

# Higher risk score at observation -> higher CCF (borrower draws more)
ccf_mean = ccf_mean + 0.30 * (risk_score_t0 - 0.5)

# Longer time to default -> more time to draw
ccf_mean = ccf_mean + 0.010 * (time_to_default - 6)

# Higher initial utilisation -> less headroom, but borrowers already
# maxed out draw the remaining space fast (weak effect)
ccf_mean = ccf_mean + 0.05 * (util_t0 - 0.4)

# Weaker payment discipline (low min_payment_ratio) -> higher CCF
ccf_mean = ccf_mean - 0.20 * (min_payment_ratio - 0.5)

# Macro stress: high unemployment vintages push CCF up (draws to cover
# income loss)
vintage_ur = macro.set_index("vintage")["unemployment_pct"].to_dict()
ur_shock = np.array([vintage_ur[v] for v in vintages])
ccf_mean = ccf_mean + 0.015 * (ur_shock - 7.5)

# Clip to (0, 1) and draw from Beta
ccf_mean = np.clip(ccf_mean, 0.03, 0.98)
kappa = 8.0
alpha = ccf_mean * kappa
beta  = (1 - ccf_mean) * kappa
ccf_realised = RNG.beta(alpha, beta)

# Drawn at default follows from CCF definition
drawn_default = drawn_t0 + ccf_realised * (limit - drawn_t0)
ead = drawn_default
util_default = drawn_default / limit

# Assemble
df = pd.DataFrame({
    "vintage": vintages,
    "product": products,
    "credit_limit_eur": np.round(limit, 2),
    "drawn_t0_eur": np.round(drawn_t0, 2),
    "utilisation_t0": np.round(util_t0, 4),
    "time_to_default_months": time_to_default,
    "risk_score_t0": np.round(risk_score_t0, 4),
    "min_payment_ratio": np.round(min_payment_ratio, 4),
    "borrower_age": borrower_age,
    "unemployment_pct": np.round(ur_shock, 2),
    "gdp_yoy_pct": np.round([macro.set_index('vintage')['gdp_yoy_pct'][v] for v in vintages], 2),
    "ccf_realised": np.round(ccf_realised, 5),
    "utilisation_default": np.round(util_default, 4),
    "ead_eur": np.round(ead, 2),
})

df.to_csv(DATA / "ead_facilities.csv", index=False)

# -----------------------------------------------------------------------------
# Sanity print
# -----------------------------------------------------------------------------
print(f"Generated {len(df):,} defaulted revolving facilities across "
      f"{df['vintage'].nunique()} vintages.")
print(f"\nCCF summary:")
print(f"  Mean: {df['ccf_realised'].mean():.3f}   Std: {df['ccf_realised'].std():.3f}")
print(f"\nCCF by product:")
print(df.groupby("product")["ccf_realised"].agg(["mean", "std", "count"]))
print(f"\nEAD summary (EUR):")
print(df["ead_eur"].describe(percentiles=[0.5, 0.9, 0.99]).round(0))
