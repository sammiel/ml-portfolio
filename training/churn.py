"""Churn risk scorer: synthetic retail-bank customers.

Generates a synthetic customer table, trains a logistic regression and a
gradient-boosting benchmark, and exports the logistic model to JSON so the
browser can score customers without a server.

Run: python training/churn.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
N = 20_000
OUT = Path(__file__).resolve().parents[1] / "site" / "churn" / "model.json"

FEATURES = [
    # name, label, min, max, step, default, unit
    ("tenure_months", "Tenure", 1, 180, 1, 36, "months"),
    ("products_held", "Products held", 1, 6, 1, 2, "products"),
    ("monthly_txns", "Monthly transactions", 0, 120, 1, 25, "txns"),
    ("balance_change_pct", "Balance change (last 3 months)", -80, 80, 1, 0, "%"),
    ("complaints_90d", "Complaints (last 90 days)", 0, 6, 1, 0, "complaints"),
    ("digital_logins_30d", "App logins (last 30 days)", 0, 60, 1, 12, "logins"),
    ("salary_account", "Salary paid into account", 0, 1, 1, 1, "yes/no"),
]


def make_data(n: int):
    tenure = RNG.gamma(2.0, 20.0, n).clip(1, 180)
    products = RNG.choice([1, 2, 3, 4, 5, 6], n, p=[0.32, 0.30, 0.18, 0.11, 0.06, 0.03])
    salary = RNG.binomial(1, 0.55, n)
    txns = (RNG.gamma(2.5, 8.0, n) + 10 * salary + 3 * products).clip(0, 120)
    balance = RNG.normal(0, 22, n).clip(-80, 80)
    complaints = RNG.poisson(0.35, n).clip(0, 6)
    logins = (RNG.gamma(2.0, 5.0, n) + 0.08 * txns).clip(0, 60)

    # Ground-truth churn propensity, with a non-linear tenure effect and an
    # interaction (complaints hurt more when the customer is barely engaged).
    logit = (
        1.6
        - 0.9 * np.log1p(tenure / 12)
        - 0.45 * (products - 1)
        - 0.035 * txns
        - 0.028 * balance
        + 0.55 * complaints
        + 0.25 * complaints * (logins < 5)
        - 0.04 * logins
        - 0.9 * salary
        + RNG.normal(0, 0.6, n)
    )
    churn = RNG.binomial(1, 1 / (1 + np.exp(-logit)))
    X = np.column_stack([tenure, products, txns, balance, complaints, logins, salary])
    return X.round(1), churn


def main():
    X, y = make_data(N)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    p_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]

    gb = HistGradientBoostingClassifier(random_state=7).fit(X_tr, y_tr)
    p_gb = gb.predict_proba(X_te)[:, 1]

    # Lift: churn rate among the riskiest 10% versus the overall rate.
    top = np.argsort(-p_lr)[: len(p_lr) // 10]
    lift = y_te[top].mean() / y_te.mean()

    # Calibration by decile, for the reliability table on the page.
    order = np.argsort(p_lr)
    deciles = np.array_split(order, 10)
    calibration = [
        {"predicted": round(float(p_lr[d].mean()), 3), "actual": round(float(y_te[d].mean()), 3)}
        for d in deciles
    ]

    metrics = {
        "rows": N,
        "test_rows": int(len(y_te)),
        "churn_rate": round(float(y.mean()), 3),
        "logreg_auc": round(float(roc_auc_score(y_te, p_lr)), 3),
        "logreg_pr_auc": round(float(average_precision_score(y_te, p_lr)), 3),
        "gbm_auc": round(float(roc_auc_score(y_te, p_gb)), 3),
        "top_decile_lift": round(float(lift), 2),
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
        "metrics": metrics,
        "calibration": calibration,
    }
    OUT.write_text(json.dumps(model, indent=1))
    print(json.dumps(metrics, indent=1))
    for f, c in zip(FEATURES, lr.coef_[0]):
        print(f"{f[0]:>20}  {c:+.3f}")


if __name__ == "__main__":
    main()
