"""Synthetic Loss Given Default (LGD) dataset generator.

Simulates ~15,000 defaulted retail exposures with mixed collateralisation
across five origination vintages. Each row represents one defaulted
facility observed to workout resolution, with realised LGD as the
target.

Design choices follow Basel IRB downturn-LGD guidance and typical
industrialised retail credit portfolios: LGD depends primarily on
collateral coverage at default (LTV), collateral type, jurisdictional
recovery efficiency, workout duration, and vintage-level macro
conditions.

Outputs
-------
data/lgd_defaults.csv : one row per defaulted facility with features,
    realised LGD, workout metadata, and vintage assignment.
data/vintage_macro.csv : per-vintage macro context (unemployment,
    house-price index) that drives downturn behaviour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260704)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Vintage macro context
# -----------------------------------------------------------------------------
VINTAGES = ["2019H1", "2019H2", "2020H1", "2020H2", "2021H1",
            "2021H2", "2022H1", "2022H2", "2023H1", "2023H2"]

# Higher unemployment and falling house prices during the stress window
# (2020 pandemic and 2022 inflation shock) produce downturn LGDs later.
unemployment = np.array([6.8, 6.9, 9.2, 9.6, 8.7, 7.9, 7.4, 8.1, 7.6, 7.2])
hpi_growth   = np.array([4.2, 3.8, -1.5, -3.2, 1.4, 5.6, 3.1, -4.4, -2.1, 1.8])

macro = pd.DataFrame({
    "vintage": VINTAGES,
    "unemployment_pct": unemployment,
    "hpi_yoy_pct": hpi_growth,
})
macro.to_csv(DATA / "vintage_macro.csv", index=False)


# -----------------------------------------------------------------------------
# Defaulted facilities
# -----------------------------------------------------------------------------
N_DEFAULTS = 15_000

# Collateral mix: mortgage-backed (secured), auto/asset-backed, unsecured
COLLATERAL_MIX = {
    "mortgage":  0.42,
    "auto":      0.18,
    "personal_secured": 0.12,
    "unsecured": 0.28,
}
collateral_types = RNG.choice(
    list(COLLATERAL_MIX.keys()),
    size=N_DEFAULTS,
    p=list(COLLATERAL_MIX.values()),
)

# Vintage assignment weighted toward more recent years
vintage_weights = np.array([0.06, 0.07, 0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.12])
vintage_ids = RNG.choice(np.arange(len(VINTAGES)), size=N_DEFAULTS, p=vintage_weights)
vintages = np.array(VINTAGES)[vintage_ids]

# Exposure at default (EUR) log-normal by collateral type
mu_ead = {
    "mortgage": 11.6,       # ~110k EUR median
    "auto": 9.7,            # ~16k EUR
    "personal_secured": 9.3,
    "unsecured": 8.9,
}
sigma_ead = {"mortgage": 0.55, "auto": 0.45, "personal_secured": 0.45,
             "unsecured": 0.60}

ead = np.array([
    RNG.lognormal(mean=mu_ead[c], sigma=sigma_ead[c])
    for c in collateral_types
])

# LTV at default (loan / collateral value). Unsecured is undefined so we
# encode a sentinel then handle in the LGD generator.
ltv_at_default = np.where(
    collateral_types == "unsecured",
    np.nan,
    np.clip(RNG.normal(loc=0.78, scale=0.15, size=N_DEFAULTS), 0.15, 1.45),
)

# Days past due at workout start (typically 90+)
dpd_at_workout = RNG.integers(low=90, high=210, size=N_DEFAULTS)

# Workout duration (months). Depends on jurisdiction efficiency and
# collateral (secured takes longer to enforce).
jurisdiction_effiency = RNG.beta(a=6, b=3, size=N_DEFAULTS)  # 0-1, higher=faster
base_workout_months = {
    "mortgage": 26.0, "auto": 14.0, "personal_secured": 18.0, "unsecured": 9.0,
}
workout_months = np.array([
    max(3.0, RNG.normal(base_workout_months[c] / (0.5 + jurisdiction_effiency[i]), 4.0))
    for i, c in enumerate(collateral_types)
])

# Borrower age at default and employment status (soft correlates)
borrower_age = np.clip(RNG.normal(45, 12, size=N_DEFAULTS), 20, 80).astype(int)
employment_status = RNG.choice(
    ["employed", "self_employed", "unemployed", "retired"],
    size=N_DEFAULTS,
    p=[0.62, 0.15, 0.15, 0.08],
)


# -----------------------------------------------------------------------------
# Target: realised LGD
# -----------------------------------------------------------------------------
# Baseline LGD per collateral type. Mortgage recoveries are high (low LGD),
# unsecured have poor recovery (high LGD).
base_lgd = {
    "mortgage": 0.18,
    "auto": 0.42,
    "personal_secured": 0.48,
    "unsecured": 0.72,
}

lgd_mean = np.array([base_lgd[c] for c in collateral_types])

# LTV effect: over-collateralised defaults recover almost fully;
# under-collateralised (LTV > 1) recover much less.
ltv_effect = np.zeros(N_DEFAULTS)
mask_secured = collateral_types != "unsecured"
ltv_effect[mask_secured] = np.clip((ltv_at_default[mask_secured] - 0.75) * 0.42, -0.15, 0.35)
lgd_mean = lgd_mean + ltv_effect

# Macro effect: downturn vintages push LGD up (recoveries fall in
# stressed markets, especially for mortgages via HPI). Match vintage.
vintage_hpi = macro.set_index("vintage")["hpi_yoy_pct"].to_dict()
vintage_ur = macro.set_index("vintage")["unemployment_pct"].to_dict()
hpi_shock = np.array([vintage_hpi[v] for v in vintages])
ur_shock  = np.array([vintage_ur[v]  for v in vintages])
# Falling HPI raises secured LGD; high unemployment raises unsecured LGD
macro_lift = np.where(
    mask_secured,
    -0.020 * hpi_shock,        # HPI drop by 3pt -> LGD +0.06
    +0.010 * (ur_shock - 7.5), # UR above 7.5 -> LGD up
)
lgd_mean = lgd_mean + macro_lift

# Workout duration effect: longer workouts cost more (raise LGD via costs)
workout_effect = (workout_months - 12.0) * 0.0035
lgd_mean = lgd_mean + workout_effect

# Employment effect (soft)
emp_effect = np.where(employment_status == "unemployed", 0.05,
             np.where(employment_status == "retired", -0.02, 0.0))
lgd_mean = lgd_mean + emp_effect

# Clip mean into (0, 1) then draw from Beta with heteroskedastic sigma
lgd_mean = np.clip(lgd_mean, 0.02, 0.95)

# Convert (mu, sigma) to Beta parameters. Use kappa (concentration) for tightness
kappa = 12.0
alpha = lgd_mean * kappa
beta  = (1 - lgd_mean) * kappa
realised_lgd = RNG.beta(alpha, beta)

# Assemble
df = pd.DataFrame({
    "vintage": vintages,
    "collateral_type": collateral_types,
    "ead_eur": np.round(ead, 2),
    "ltv_at_default": np.round(ltv_at_default, 4),
    "dpd_at_workout": dpd_at_workout,
    "workout_months": np.round(workout_months, 2),
    "borrower_age": borrower_age,
    "employment_status": employment_status,
    "unemployment_pct": np.round(ur_shock, 2),
    "hpi_yoy_pct": np.round(hpi_shock, 2),
    "realised_lgd": np.round(realised_lgd, 5),
})

df.to_csv(DATA / "lgd_defaults.csv", index=False)

# -----------------------------------------------------------------------------
# Sanity print
# -----------------------------------------------------------------------------
print(f"Generated {len(df):,} defaulted facilities across {df['vintage'].nunique()} vintages.")
print(f"Mean realised LGD: {df['realised_lgd'].mean():.3f}")
print(f"Std realised LGD:  {df['realised_lgd'].std():.3f}")
print("\nLGD by collateral type:")
print(df.groupby("collateral_type")["realised_lgd"].agg(["mean", "std", "count"]))
print("\nLGD by vintage:")
print(df.groupby("vintage")["realised_lgd"].agg(["mean", "count"]))
