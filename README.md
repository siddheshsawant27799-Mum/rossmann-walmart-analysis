# Retail Demand Forecasting Dashboards

One Streamlit page presents the Rossmann and Walmart forecasting results in separate tabs. Both retain their existing design and Plotly charts built from the project's numeric evaluation and prediction files. Each tab has its own model selection, which updates the model-specific KPIs and charts.

## Install

From this project folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the combined dashboard

```powershell
streamlit run rossmann_dashboard.py
```

Switch between the Rossmann and Walmart tabs at the top of the page. You can also run `streamlit run walmart_dashboard.py` to open the same combined page with Walmart selected initially.

Each dashboard reads its data from the corresponding `*-sales-forecast/experiment/outputs/` directory. Keep those output folders with the dashboard files. If a required result file is missing or unreadable, the app shows the path and stops with an error.

## Metrics and interpretation

- Validation MAE and WAPE and historical-only MAE are loaded from the saved candidate metrics.
- Improvement from contextual variables is `(historical-only MAE - contextual MAE) / historical-only MAE × 100`.
- Validation rows come from the saved evaluation results.
- Rossmann final holdout predictions are available for the selected XGBoost model. Walmart final holdout predictions and the shared seasonal-naive comparison are available for Random Forest. The dashboards disclose when model-specific holdout data is unavailable.
- Forecasting performance does not directly measure inventory outcomes. Neither dataset provides complete stock positions, replenishment records, stockout events, supplier lead times, holding costs or service-level targets.
- Rossmann is daily store-level data and Walmart is weekly store-department data; their sales totals and error magnitudes are not directly comparable.

## Streamlit Community Cloud

Publish this folder to a GitHub repository, then create an app in Streamlit Community Cloud with that repository, its published branch, and `app.py` as the main file. Both dataset tabs are included in this entry point. Dependencies are declared in `requirements.txt`.

The `.gitignore` is a deployment allowlist: only dashboard code, documentation, requirements, and the specific CSV/Parquet result files used by the dashboards are included. Raw datasets, thesis files, trained models, virtual environments, and local secrets stay out of the deployment repository. Preserve the result files' directory structure when uploading.
