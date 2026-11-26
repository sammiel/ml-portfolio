"""Customer health scoring: synthetic B2B (enterprise) accounts.

Trains a renewal model on synthetic account usage, turns the predicted renewal
probability into a 0-100 health score, and exports the model plus a sample
portfolio of accounts for the browser dashboard.

Run: python training/health.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(11)
N = 6_000
N_PORTFOLIO = 120
OUT = Path(__file__).resolve().parents[1] / "site" / "health" / "model.json"

FEATURES = [
    ("seat_utilisation", "Seat utilisation", 0, 100, 1, 60, "%"),
    ("active_pct_30d", "Learners active (last 30 days)", 0, 100, 1, 35, "%"),
    ("hours_per_learner", "Learning hours per active learner (quarter)", 0, 30, 0.5, 6, "hours"),
    ("utilisation_trend", "Utilisation change vs last quarter", -40, 40, 1, 0, "pts"),
    ("admin_logins_30d", "Admin logins (last 30 days)", 0, 40, 1, 6, "logins"),
    ("tickets_90d", "Support tickets (last 90 days)", 0, 15, 1, 2, "tickets"),
    ("exec_sponsor", "Named executive sponsor", 0, 1, 1, 1, "yes/no"),
]

TIERS = [
    {"name": "Healthy", "min": 70},
    {"name": "Watch", "min": 40},
    {"name": "At risk", "min": 0},
]


def make_data(n: int):
    sponsor = RNG.binomial(1, 0.5, n)
    util = (RNG.beta(3, 2, n) * 100 + 6 * sponsor).clip(0, 100)
    active = (util * RNG.uniform(0.35, 0.8, n) + RNG.normal(0, 6, n)).clip(0, 100)
    hours = RNG.gamma(2.2, 2.8, n).clip(0, 30)
    trend = RNG.normal(0, 11, n).clip(-40, 40)
    admin = (RNG.poisson(5, n) + 3 * sponsor).clip(0, 40)
    tickets = RNG.poisson(2.0, n).clip(0, 15)

    logit = (
        -3.0
        + 0.035 * util
        + 0.03 * active
        + 0.08 * hours
        + 0.06 * trend
        + 0.07 * admin
        - 0.18 * tickets
        + 0.9 * sponsor
        + RNG.normal(0, 0.7, n)
    )
    renewed = RNG.binomial(1, 1 / (1 + np.exp(-logit)))
    X = np.column_stack([util, active, hours, trend, admin, tickets, sponsor])
    return X.round(1), renewed


def main():
    X, y = make_data(N)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    p = lr.predict_proba(scaler.transform(X_te))[:, 1]
    score = 100 * p

    tier_stats = []
    lo_bounds = [70, 40, 0]
    hi_bounds = [101, 70, 40]
    for t, lo, hi in zip(TIERS, lo_bounds, hi_bounds):
        mask = (score >= lo) & (score < hi)
        tier_stats.append({
            "tier": t["name"],
            "share": round(float(mask.mean()), 3),
            "renewal_rate": round(float(y_te[mask].mean()), 3) if mask.any() else None,
        })

    # A portfolio of fresh accounts for the dashboard (not used in training).
    Xp, _ = make_data(N_PORTFOLIO)
    steps = np.array([f[4] for f in FEATURES])
    Xp = np.round(Xp / steps) * steps  # snap to slider steps
    portfolio = [
        {"id": f"ACC-{i + 1:03d}", "values": [round(float(v), 1) for v in row]}
        for i, row in enumerate(Xp)
    ]

    metrics = {
        "rows": N,
        "test_rows": int(len(y_te)),
        "renewal_rate": round(float(y.mean()), 3),
        "auc": round(float(roc_auc_score(y_te, p)), 3),
        "tiers": tier_stats,
    }
    model = {
        "features": [
            {"name": f[0], "label": f[1], "min": f[2], "max": f[3], "step": f[4], "default": f[5], "unit": f[6]}
            for f in FEATURES
        ],
        "mean": scaler.mean_.round(5).tolist(),
        "scale": scaler.scale_.round(5).tolist(),
        "coef": lr.coef_[0].round(5).tolist(),
        "intercept": round(float(lr.intercept_[0]), 5),
        "tiers": TIERS,
        "metrics": metrics,
        "portfolio": portfolio,
    }
    OUT.write_text(json.dumps(model))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
