"""Forecasting demo: weekly enrolments with trend and yearly seasonality.

Fits a regression on log(enrolments) with a linear trend and Fourier terms,
written in plain NumPy. Backtests it on the last 26 weeks against two
baselines, then refits on all data and exports a 26-week forecast with a
90% prediction interval.

Run: python training/forecast.py
"""
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(5)
WEEKS = 156          # three years of history
HORIZON = 26         # weeks to forecast
PERIOD = 52.18       # weeks per year
OFFSET = 48          # week 0 (4 Dec 2023) is week 48 of the year, counting from 0
K = 4                # Fourier pairs
Z90 = 1.645
OUT = Path(__file__).resolve().parents[1] / "site" / "forecast" / "model.json"


def make_series(n: int):
    t = np.arange(n)
    week_of_year = (t + OFFSET) % 52
    level = 1800 * np.exp(0.0035 * t)                      # ~20% growth a year
    season = 1 + 0.18 * np.sin(2 * np.pi * (t + OFFSET - 2) / PERIOD) + 0.08 * np.cos(4 * np.pi * (t + OFFSET) / PERIOD)
    holiday = np.where((week_of_year >= 51) | (week_of_year == 0), 0.7, 1.0)  # last week of Dec, first of Jan
    noise = RNG.lognormal(0, 0.06, n)
    return np.round(level * season * holiday * noise)


def design(t: np.ndarray):
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * (t + OFFSET) / PERIOD), np.cos(2 * np.pi * k * (t + OFFSET) / PERIOD)]
    week_of_year = (t + OFFSET) % 52
    cols.append(((week_of_year >= 51) | (week_of_year == 0)).astype(float))
    return np.column_stack(cols)


def fit(t, y):
    X = design(t)
    beta, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    resid = np.log(y) - X @ beta
    sigma = resid.std(ddof=X.shape[1])
    return beta, sigma


def predict(beta, sigma, t):
    mu = design(t) @ beta
    return np.exp(mu), np.exp(mu - Z90 * sigma), np.exp(mu + Z90 * sigma)


def mape(actual, pred):
    return float(np.mean(np.abs(actual - pred) / actual))


def main():
    y = make_series(WEEKS)
    t = np.arange(WEEKS)

    # Backtest: train on the first 130 weeks, forecast the last 26.
    cut = WEEKS - HORIZON
    beta, sigma = fit(t[:cut], y[:cut])
    bt_pred, bt_lo, bt_hi = predict(beta, sigma, t[cut:])
    actual = y[cut:]
    naive = np.full(HORIZON, y[cut - 1])
    seasonal_naive = y[cut - 52: cut - 52 + HORIZON]
    coverage = float(np.mean((actual >= bt_lo) & (actual <= bt_hi)))

    # Final model on all history.
    beta, sigma = fit(t, y)
    fitted, _, _ = predict(beta, sigma, t)
    tf = np.arange(WEEKS, WEEKS + HORIZON)
    f, lo, hi = predict(beta, sigma, tf)

    metrics = {
        "weeks": WEEKS,
        "horizon": HORIZON,
        "model_mape": round(mape(actual, bt_pred), 4),
        "naive_mape": round(mape(actual, naive), 4),
        "seasonal_naive_mape": round(mape(actual, seasonal_naive), 4),
        "interval_coverage": round(coverage, 3),
        "annual_growth": round(float(np.exp(beta[1] * 52) - 1), 3),
    }
    model = {
        "start": "2023-12-04",
        "history": y.astype(int).tolist(),
        "fitted": np.round(fitted).astype(int).tolist(),
        "backtest": {
            "start_index": cut,
            "pred": np.round(bt_pred).astype(int).tolist(),
            "seasonal_naive": seasonal_naive.astype(int).tolist(),
        },
        "forecast": {
            "mean": np.round(f).astype(int).tolist(),
            "lo": np.round(lo).astype(int).tolist(),
            "hi": np.round(hi).astype(int).tolist(),
        },
        "metrics": metrics,
    }
    OUT.write_text(json.dumps(model))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
