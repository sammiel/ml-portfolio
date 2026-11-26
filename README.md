# Applied ML for customer analytics

Three small, interactive machine learning projects based on problems from banking and enterprise education. Each model is trained in Python, exported to JSON and scored in the browser, so the site is fully static and hosted on Netlify.

**Live site:** _add your Netlify URL here_

All data is synthetic. No employer or customer data is used.

| Project | Question | Method | Result on held-out data |
|---|---|---|---|
| [Churn risk scorer](site/churn/) | Which retail-bank customers are likely to leave next quarter, and why? | Logistic regression, benchmarked against gradient boosting | AUC 0.795 (gradient boosting 0.785); top-decile lift 3.0× |
| [Customer health scoring](site/health/) | Which B2B accounts are at risk of not renewing? | Logistic regression on renewal outcomes, turned into a 0 to 100 score with tiers | AUC 0.796; renewal rate 86% for Healthy accounts vs 25% for At risk |
| [Enrolment forecasting](site/forecast/) | How many weekly enrolments should we plan for over the next six months? | Log-linear regression with trend, Fourier seasonality and a holiday flag, written in NumPy | 5.8% MAPE vs 19.9% for seasonal naive |

## Design choices

- **Simple, explainable models first.** On the churn data, logistic regression matched gradient boosting. It's easier to explain to a business team, and its coefficients can be scored in a few lines of JavaScript.
- **Scores fitted to outcomes.** The health score is fitted to renewal outcomes rather than to hand-picked weights, and each tier is checked against its actual renewal rate.
- **Baselines before trust.** The forecast is backtested on the last 26 weeks against naive and seasonal-naive baselines.
- **Honest uncertainty.** The forecast's 90% range covered 81% of backtest weeks. That is slightly narrow because it ignores uncertainty in the fitted trend. A bootstrap would be the next improvement.

## Repository layout

```
training/          Python scripts that generate the data, train and evaluate the models, and export JSON
  churn.py
  health.py
  forecast.py
site/              Static site published by Netlify
  churn/  health/  forecast/   one page and one model.json per project
  assets/          shared CSS, scoring helpers, Chart.js (MIT licence)
netlify.toml       Netlify publishes the site/ folder; no build step needed
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python training/churn.py
python training/health.py
python training/forecast.py
python -m http.server 8000 -d site   # then open http://localhost:8000
```

Each training script prints its evaluation metrics and overwrites the matching `site/<project>/model.json`.
