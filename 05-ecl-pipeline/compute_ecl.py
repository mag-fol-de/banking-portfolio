"""End-to-end IFRS 9 Expected Credit Loss (ECL) calculation.

Combines the PD, LGD, and EAD components on the performing retail
portfolio, applies the IFRS 9 staging rules, and computes 12-month
and lifetime ECL under three macro scenarios (baseline, adverse,
severe adverse) with probability weights.

ECL identity:

    ECL = sum over scenarios of weight_s * (PD_s * LGD_s * EAD * DiscFactor)

where:
    - Stage 1: PD is 12-month; ECL is 12-month PD * LGD * EAD
    - Stage 2 and 3: PD is lifetime (approximated by cumulative PD across
      remaining maturity); ECL is lifetime
    - Stage 3: PD = 1 (already impaired), lifetime LGD * EAD

For simplicity a flat lifetime PD term structure is used:
    lifetime_PD_1y = 1 - (1 - PD_12m)^T
where T is remaining maturity in years (5 for term loans, 3 for
revolving as a policy default).

Outputs
-------
outputs/loan_level_ecl.csv    : per-loan ECL under each scenario and weighted
outputs/portfolio_ecl.csv     : totals by stage, product, scenario
figures/*.png                 : diagnostic plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
OUT  = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA / "portfolio.csv")
scenarios = pd.read_csv(DATA / "macro_scenarios.csv")
print(f"Portfolio: {len(df):,} loans, total EAD {df['ead_eur'].sum()/1e6:.1f}m EUR")
print(f"Scenarios: {len(scenarios)} macro paths")


# -----------------------------------------------------------------------------
# Lifetime PD term structure
# -----------------------------------------------------------------------------
# Remaining maturity (years). Term loans default to 5 years, revolving 3.
remaining_maturity = np.where(df["is_revolving"], 3.0, 5.0)


def lifetime_pd(pd_12m, T):
    """Cumulative PD across T years assuming constant hazard."""
    return 1.0 - (1.0 - pd_12m) ** T


# -----------------------------------------------------------------------------
# Discounting: simple annual factor (effective rate ~ 3% p.a.)
# For lifetime we discount at the average of years 1..T.
# -----------------------------------------------------------------------------
DISCOUNT_RATE = 0.03

def disc_lifetime(T):
    """Weighted-average discount factor over T years for a simple annuity of
    default probabilities.
    """
    years = np.arange(1, int(np.ceil(T.max())) + 1)
    df_year = 1.0 / (1.0 + DISCOUNT_RATE) ** years
    return np.array([df_year[:int(np.ceil(t))].mean() for t in T])

disc_12m = 1.0 / (1.0 + DISCOUNT_RATE)  # single-year discount
disc_life = disc_lifetime(remaining_maturity)


# -----------------------------------------------------------------------------
# Per-loan ECL under each scenario
# -----------------------------------------------------------------------------
records = []
for _, row in scenarios.iterrows():
    scen = row["scenario"]
    w = float(row["weight"])
    pd_mult = float(row["pd_multiplier"])
    lgd_mult = float(row["lgd_multiplier"])

    pd_12m_s = np.clip(df["pd_12m"].to_numpy() * pd_mult, 0, 1)
    lgd_s    = np.clip(df["lgd"].to_numpy() * lgd_mult, 0, 1)
    lifetime_pd_s = lifetime_pd(pd_12m_s, remaining_maturity)

    stage = df["ifrs9_stage"].to_numpy()
    ead = df["ead_eur"].to_numpy()

    # ECL per stage
    ecl_12m = pd_12m_s * lgd_s * ead * disc_12m
    ecl_life = lifetime_pd_s * lgd_s * ead * disc_life

    ecl = np.where(
        stage == 1, ecl_12m,
        np.where(stage == 2, ecl_life, lgd_s * ead)   # stage 3: PD=1
    )

    records.append({
        "scenario": scen,
        "weight": w,
        "ecl": ecl,
    })

# Weighted ECL
loan_ecl_weighted = np.zeros(len(df))
for r in records:
    loan_ecl_weighted += r["weight"] * r["ecl"]

loan_df = df[["loan_id", "product", "collateral_type", "is_revolving",
              "is_secured", "current_balance_eur", "ead_eur", "pd_12m",
              "lgd", "ifrs9_stage"]].copy()
for r in records:
    loan_df[f"ecl_{r['scenario']}"] = np.round(r["ecl"], 2)
loan_df["ecl_weighted"] = np.round(loan_ecl_weighted, 2)
loan_df.to_csv(OUT / "loan_level_ecl.csv", index=False)


# -----------------------------------------------------------------------------
# Portfolio aggregates
# -----------------------------------------------------------------------------
total_ecl_by_scenario = {r["scenario"]: float(r["ecl"].sum()) for r in records}
weighted_total = float(loan_ecl_weighted.sum())

print("\n== Portfolio-level ECL ==")
for scen, val in total_ecl_by_scenario.items():
    print(f"  {scen:15s}: {val/1e6:.2f}m EUR")
print(f"  {'weighted':15s}: {weighted_total/1e6:.2f}m EUR")

# By stage
by_stage = (
    loan_df.groupby("ifrs9_stage")
    .agg(n_loans=("loan_id", "size"),
         total_ead=("ead_eur", "sum"),
         ecl_baseline=("ecl_baseline", "sum"),
         ecl_adverse=("ecl_adverse", "sum"),
         ecl_severe=("ecl_severe_adverse", "sum"),
         ecl_weighted=("ecl_weighted", "sum"))
    .reset_index()
)
by_stage["coverage_ratio_weighted"] = (by_stage["ecl_weighted"] / by_stage["total_ead"]).round(4)
by_stage.to_csv(OUT / "portfolio_ecl_by_stage.csv", index=False)
print("\n== ECL by IFRS 9 stage ==")
print(by_stage.to_string(index=False))

# By product
by_product = (
    loan_df.groupby("product")
    .agg(n_loans=("loan_id", "size"),
         total_ead=("ead_eur", "sum"),
         ecl_weighted=("ecl_weighted", "sum"),
         mean_pd_12m=("pd_12m", "mean"),
         mean_lgd=("lgd", "mean"))
    .reset_index()
)
by_product["coverage_ratio_weighted"] = (by_product["ecl_weighted"] / by_product["total_ead"]).round(4)
by_product.to_csv(OUT / "portfolio_ecl_by_product.csv", index=False)
print("\n== ECL by product ==")
print(by_product.to_string(index=False))


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

# 1. ECL by scenario (bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
scens = [r["scenario"] for r in records] + ["weighted"]
vals  = [total_ecl_by_scenario[s] for s in scens[:-1]] + [weighted_total]
colors = ["#2b6b8b", "#e0a04a", "#c14e4e", "#5a8b6a"]
ax.bar(scens, [v/1e6 for v in vals], color=colors)
for i, v in enumerate(vals):
    ax.text(i, v/1e6 + 0.5, f"{v/1e6:.1f}m", ha="center", fontweight="bold")
ax.set_ylabel("ECL (million EUR)")
ax.set_title("Portfolio ECL by macro scenario (weighted = final IFRS 9 provision)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "01_ecl_by_scenario.png")
plt.close()

# 2. ECL by stage
fig, ax = plt.subplots(figsize=(8, 5))
stages = by_stage["ifrs9_stage"].astype(str).tolist()
vals_s = (by_stage["ecl_weighted"] / 1e6).tolist()
ns = by_stage["n_loans"].tolist()
bars = ax.bar([f"Stage {s}" for s in stages], vals_s,
              color=["#5a8b6a", "#e0a04a", "#c14e4e"])
for i, (v, n) in enumerate(zip(vals_s, ns)):
    ax.text(i, v + max(vals_s)*0.02, f"{v:.2f}m\n({n:,} loans)",
            ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Weighted ECL (million EUR)")
ax.set_title("IFRS 9 weighted ECL by stage")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "02_ecl_by_stage.png")
plt.close()

# 3. Coverage ratio by stage (ECL / EAD)
fig, ax = plt.subplots(figsize=(8, 5))
cov = by_stage["coverage_ratio_weighted"].tolist()
ax.bar([f"Stage {s}" for s in stages], cov,
       color=["#5a8b6a", "#e0a04a", "#c14e4e"])
for i, v in enumerate(cov):
    ax.text(i, v + max(cov)*0.02, f"{v*100:.1f}%",
            ha="center", fontweight="bold")
ax.set_ylabel("Coverage ratio (ECL / EAD)")
ax.set_title("Coverage ratio by IFRS 9 stage")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "03_coverage_by_stage.png")
plt.close()

# 4. ECL by product
fig, ax = plt.subplots(figsize=(10, 5))
prods = by_product["product"].tolist()
vals_p = (by_product["ecl_weighted"] / 1e6).tolist()
bars = ax.bar(prods, vals_p, color="#2b6b8b")
for i, v in enumerate(vals_p):
    ax.text(i, v + max(vals_p)*0.02, f"{v:.2f}m",
            ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Weighted ECL (million EUR)")
ax.set_title("Weighted ECL by product")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "04_ecl_by_product.png")
plt.close()

# 5. Scenario sensitivity: ECL uplift over baseline
adverse_uplift = (total_ecl_by_scenario["adverse"] - total_ecl_by_scenario["baseline"]) / total_ecl_by_scenario["baseline"] * 100
severe_uplift  = (total_ecl_by_scenario["severe_adverse"] - total_ecl_by_scenario["baseline"]) / total_ecl_by_scenario["baseline"] * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Baseline", "Adverse", "Severe adverse"],
       [total_ecl_by_scenario["baseline"]/1e6,
        total_ecl_by_scenario["adverse"]/1e6,
        total_ecl_by_scenario["severe_adverse"]/1e6],
       color=["#2b6b8b", "#e0a04a", "#c14e4e"])
ax.axhline(weighted_total/1e6, color="black", linestyle="--", lw=1.5,
           label=f"Weighted (final) = {weighted_total/1e6:.1f}m")
ax.set_ylabel("ECL (million EUR)")
ax.set_title(f"Scenario ECL sensitivity: adverse +{adverse_uplift:.0f}%, severe +{severe_uplift:.0f}%")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "05_scenario_sensitivity.png")
plt.close()

# 6. Stage migration risk: how much ECL a stage-1 -> stage-2 migration triggers
# For each stage-1 loan, compare its stage-1 ECL to its hypothetical stage-2 ECL
stage1_mask = df["ifrs9_stage"] == 1
pd_baseline = df["pd_12m"].to_numpy() * 1.0
lgd_baseline = df["lgd"].to_numpy() * 1.0
lifetime_pd_baseline = lifetime_pd(pd_baseline, remaining_maturity)
ead = df["ead_eur"].to_numpy()

ecl_as_stage1 = pd_baseline * lgd_baseline * ead * disc_12m
ecl_as_stage2 = lifetime_pd_baseline * lgd_baseline * ead * disc_life

migration_uplift = (ecl_as_stage2 - ecl_as_stage1)[stage1_mask]
avg_uplift = float(migration_uplift.mean())
total_potential = float(migration_uplift.sum())

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(migration_uplift, bins=60, color="#e0a04a", edgecolor="black", linewidth=0.3)
ax.axvline(avg_uplift, color="black", ls="--", lw=1.5,
           label=f"Mean uplift = {avg_uplift:,.0f} EUR")
ax.set_xlabel("Per-loan ECL uplift on stage 1 -> stage 2 migration (EUR)")
ax.set_ylabel("Number of loans")
ax.set_title(f"Stage-migration ECL cliff on stage-1 population\n"
             f"Total potential uplift if all migrated: {total_potential/1e6:.1f}m EUR")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "06_migration_cliff.png")
plt.close()


# -----------------------------------------------------------------------------
# Headline
# -----------------------------------------------------------------------------
headline = {
    "n_loans": int(len(df)),
    "total_ead_eur": float(df["ead_eur"].sum()),
    "ecl_baseline_eur": float(total_ecl_by_scenario["baseline"]),
    "ecl_adverse_eur":  float(total_ecl_by_scenario["adverse"]),
    "ecl_severe_eur":   float(total_ecl_by_scenario["severe_adverse"]),
    "ecl_weighted_eur": float(weighted_total),
    "coverage_ratio_weighted": float(weighted_total / df["ead_eur"].sum()),
    "n_stage_1": int((df["ifrs9_stage"] == 1).sum()),
    "n_stage_2": int((df["ifrs9_stage"] == 2).sum()),
    "n_stage_3": int((df["ifrs9_stage"] == 3).sum()),
    "adverse_uplift_pct": float(adverse_uplift),
    "severe_uplift_pct":  float(severe_uplift),
    "avg_migration_uplift_eur": float(avg_uplift),
    "total_migration_potential_eur": float(total_potential),
}
pd.Series(headline).to_csv(OUT / "headline.csv")
print("\nHeadline:")
for k, v in headline.items():
    print(f"  {k}: {v}")

print("\nDone.")
