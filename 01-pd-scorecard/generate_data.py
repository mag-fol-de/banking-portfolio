"""Synthetic loan portfolio generator with bureau attributes and origination vintages.

Produces ten origination cohorts (2019H1 through 2023H2) of ~25,000 loans each,
totaling 250,000 applicants. This gives 5 years of observation data, which is
the Basel minimum for retail IRB PD estimation. Each loan has a known realized
default outcome observed over an 18-month performance window.

The data-generating process uses a logit model where bureau attributes
(score, prior bankruptcies, recent delinquency status, inquiries, utilization)
are the strong drivers, application attributes (DTI, income, employment) are
moderate drivers, and demographics (age, loan purpose) are weak drivers.
Vintage-level fixed effects introduce cohort-to-cohort variation in the
realized default rate, mimicking real macroeconomic cycle effects from
pre-COVID through the 2023 rate-hike stress.

Target Gini under correctly specified logistic regression: ~0.55.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_PER_VINTAGE = 25_000
VINTAGES = [
    "2019H1", "2019H2",   # pre-COVID, benign
    "2020H1", "2020H2",   # COVID + forbearance (masked stress)
    "2021H1", "2021H2",   # recovery, low rates
    "2022H1", "2022H2",   # rate-hike cycle begins
    "2023H1", "2023H2",   # rate hikes biting hardest
]
VINTAGE_EFFECT = {                          # macro shift in logit space
    "2019H1": -0.25,                         # pre-COVID best cohort
    "2019H2": -0.20,
    "2020H1": -0.10,                         # COVID, masked by forbearance
    "2020H2": -0.05,
    "2021H1": -0.10,                         # recovery
    "2021H2": -0.15,                         # low-rate sweet spot
    "2022H1":  0.00,                         # rate hikes begin
    "2022H2": +0.05,
    "2023H1": +0.15,                         # rate hikes biting
    "2023H2": +0.25,                         # worst cohort
}

HOME_OWNERSHIP_LEVELS = ["RENT", "OWN", "MORTGAGE"]
LOAN_PURPOSE_LEVELS = ["debt_consolidation", "credit_card",
                       "home_improvement", "medical", "other"]

# -----------------------------------------------------------------------------
# Generator per vintage
# -----------------------------------------------------------------------------
def generate_vintage(vintage: str, n: int, start_id: int) -> pd.DataFrame:
    # Demographics
    age = np.clip(rng.normal(42, 12, n).round(), 20, 75).astype(int)

    # Income, log-normal
    log_income = rng.normal(np.log(45_000), 0.55, n)
    annual_income = np.exp(log_income).round().astype(int)

    # Employment length, exponential capped
    employment_length_years = np.clip(rng.exponential(5, n), 0, 40).round(1)

    # Loan terms
    loan_amount = rng.choice([5_000, 10_000, 15_000, 25_000, 40_000, 75_000],
                             size=n, p=[0.18, 0.25, 0.22, 0.18, 0.12, 0.05])
    loan_term_months = rng.choice([36, 60], size=n, p=[0.55, 0.45])

    # Bureau score, 300-900 range, mean 700
    bureau_score = np.clip(rng.normal(700, 80, n), 300, 900).round().astype(int)

    # Credit history length, correlated with age
    credit_history_length_years = np.clip(
        (age - 20) * 0.6 + rng.normal(0, 3, n), 0, 50).round(1)

    # Number of credit lines, weakly correlated with age
    num_credit_lines = np.clip(
        rng.poisson(4 + age * 0.05, n), 0, 30).astype(int)

    # Prior bankruptcies (rare)
    prior_bankruptcies = rng.choice([0, 1, 2], size=n, p=[0.95, 0.045, 0.005])

    # Delinquencies in past 24 months (count)
    base_delinq_rate = 0.7 + (730 - bureau_score) / 100 * 0.15
    delinquencies_2y = rng.poisson(np.clip(base_delinq_rate, 0, 5), n)

    # Worst delinquency status in past 24 months (0 = current, 4 = 120+)
    worst_status_24m = np.where(
        delinquencies_2y == 0, 0,
        rng.choice([1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.15, 0.05]))

    # Inquiries in last 6 months (count)
    inquiries_6m = rng.poisson(1.2 + (730 - bureau_score) / 100 * 0.4, n)
    inquiries_6m = np.clip(inquiries_6m, 0, 15)

    # Revolving utilization (percent)
    base_util = 30 + (730 - bureau_score) / 100 * 25
    utilization_pct = np.clip(rng.normal(base_util, 20, n), 0, 130).round(1)

    # Public records (Kronofogde-style), rare
    public_records = rng.choice([0, 1, 2], size=n, p=[0.94, 0.05, 0.01])

    # Debt to income
    debt_to_income_pct = np.clip(rng.gamma(2.5, 6, n), 0, 80).round(1)

    # Home ownership, age-correlated
    p_mortgage = np.clip(0.05 + (age - 25) * 0.012, 0, 0.65)
    p_own = np.clip(0.02 + (age - 30) * 0.005, 0, 0.20)
    p_rent = 1 - p_mortgage - p_own
    home_ownership = np.empty(n, dtype=object)
    for i in range(n):
        home_ownership[i] = rng.choice(
            HOME_OWNERSHIP_LEVELS, p=[p_rent[i], p_own[i], p_mortgage[i]])

    # Loan purpose
    loan_purpose = rng.choice(LOAN_PURPOSE_LEVELS, size=n,
                              p=[0.45, 0.18, 0.15, 0.10, 0.12])

    # Quoted interest rate, function of bureau score
    interest_rate_pct = np.clip(
        16 - (bureau_score - 600) * 0.025 + rng.normal(0, 0.8, n), 4, 25).round(2)

    # -------------------------------------------------------------------------
    # True logit and realized default
    # -------------------------------------------------------------------------
    z_score = (bureau_score - 700) / 80
    log_loan = np.log(loan_amount)

    logit = (
        -3.6                                          # base level (calibrated to ~10% overall DR)
        + VINTAGE_EFFECT[vintage]
        - 0.55 * z_score                              # strong: bureau score
        + 1.20 * prior_bankruptcies                   # strong: prior BK
        + 0.40 * worst_status_24m                     # strong: recent delinq
        + 0.12 * inquiries_6m                         # moderate
        + 0.012 * utilization_pct                     # moderate
        + 0.018 * debt_to_income_pct                  # moderate
        + 0.18 * delinquencies_2y                     # moderate
        + 0.55 * public_records                       # rare but strong
        - 0.025 * employment_length_years             # weak: stability
        - 0.18 * (log_income - np.log(45_000))        # weak: income
        + 0.04 * (log_loan - np.log(15_000))          # weak: loan size
        - 0.008 * age                                 # weak: age
    )
    # Loan purpose effect
    purpose_effect = {
        "debt_consolidation": +0.10,
        "credit_card":         +0.08,
        "home_improvement":    -0.05,
        "medical":             +0.02,
        "other":                0.00,
    }
    logit += np.array([purpose_effect[p] for p in loan_purpose])
    # Home ownership effect
    ho_effect = {"RENT": +0.15, "OWN": -0.10, "MORTGAGE": 0.0}
    logit += np.array([ho_effect[h] for h in home_ownership])

    true_pd = 1.0 / (1.0 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < true_pd).astype(int)

    # -------------------------------------------------------------------------
    # Introduce realistic missingness
    # -------------------------------------------------------------------------
    annual_income_obs = annual_income.astype(float)
    miss_inc = rng.uniform(0, 1, n) < (0.02 + 0.015 * (default == 1))
    annual_income_obs[miss_inc] = np.nan

    employment_length_obs = employment_length_years.astype(float)
    miss_emp = rng.uniform(0, 1, n) < (0.025 + 0.02 * (default == 1))
    employment_length_obs[miss_emp] = np.nan

    months_since_last_delinquency = np.where(
        delinquencies_2y > 0,
        np.clip(rng.exponential(8, n), 0, 24).round(),
        np.nan)

    df = pd.DataFrame({
        "loan_id": np.arange(start_id, start_id + n),
        "origination_vintage": vintage,
        "age": age,
        "annual_income": annual_income_obs,
        "employment_length_years": employment_length_obs,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate_pct": interest_rate_pct,
        "debt_to_income_pct": debt_to_income_pct,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "bureau_score": bureau_score,
        "credit_history_length_years": credit_history_length_years,
        "num_credit_lines": num_credit_lines,
        "delinquencies_2y": delinquencies_2y,
        "worst_status_24m": worst_status_24m,
        "months_since_last_delinquency": months_since_last_delinquency,
        "inquiries_6m": inquiries_6m,
        "utilization_pct": utilization_pct,
        "public_records": public_records,
        "prior_bankruptcies": prior_bankruptcies,
        "true_pd": true_pd,
        "default": default,
    })
    return df


frames = []
next_id = 1
for v in VINTAGES:
    df_v = generate_vintage(v, N_PER_VINTAGE, next_id)
    frames.append(df_v)
    next_id += N_PER_VINTAGE

panel = pd.concat(frames, ignore_index=True)
panel = panel.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# Save
panel.to_csv(DATA / "loan_applicants.csv", index=False)
panel_no_truth = panel.drop(columns=["true_pd"])
con = duckdb.connect(str(DATA / "bank.duckdb"))
con.execute("DROP TABLE IF EXISTS raw_loan_applicants")
con.register("temp_panel", panel_no_truth)
con.execute("CREATE TABLE raw_loan_applicants AS SELECT * FROM temp_panel")
con.close()

print(f"Wrote {DATA / 'loan_applicants.csv'} ({len(panel):,} rows)")
print(f"Wrote {DATA / 'bank.duckdb'} (table raw_loan_applicants)")

print("\nDefault rate by vintage:")
print(panel.groupby("origination_vintage")
      .agg(n=("default", "size"),
           defaults=("default", "sum"),
           dr=("default", "mean"))
      .round(4)
      .to_string())

print(f"\nOverall default rate: {panel['default'].mean():.4f}")
print(f"True PD range: [{panel['true_pd'].min():.4f}, {panel['true_pd'].max():.4f}]")
print(f"True PD median: {panel['true_pd'].median():.4f}")
