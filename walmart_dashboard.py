from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_components import (
    BLUE, NAVY, ORANGE, WALMART, WALMART_METHOD, bottom_section, chart_layout,
    comparison_chart, forecast_chart, hero, kpi_row, pair_chart, required, show_chart, style,
)

def render_dashboard() -> None:
    hero("Walmart Retail Demand Forecasting Dashboard", "Comparison of contextual machine-learning forecasting with a seasonal benchmark", "Walmart")

    metrics = required(WALMART / "figures" / "research" / "validation_candidate_metrics.csv")
    validation = required(WALMART / "metrics" / "phase_7_evaluation" / "validation_metrics.csv")
    holdout = required(WALMART / "metrics" / "phase_7_evaluation" / "final_holdout_metrics.csv")
    selected = st.selectbox("Selected model", ["XGBoost", "Random Forest"], index=1, key="walmart-model")
    model_key = selected.lower().replace(" ", "_")
    st.caption(f"Selected model: {selected}")
    context = metrics[(metrics.model == model_key) & (metrics.feature_set == "historical_plus_context")].iloc[0]
    historical = metrics[(metrics.model == model_key) & (metrics.feature_set == "historical_only")].iloc[0]
    improvement = (float(historical.mae) - float(context.mae)) / float(historical.mae) * 100

    kpi_row([
        ("Validation MAE", f"{context.mae:,.2f}"),
        ("Validation WAPE", f"{context.wape_pct:.2f}%"),
        ("Historical-only MAE", f"{historical.mae:,.2f}"),
        ("Improvement from contextual variables", f"{improvement:.1f}%"),
        ("Validation rows", f"{int(context.rows):,}"),
    ])
    if model_key == "random_forest":
        interpretation = ("Contextual variables produced a modest improvement over the historical-only Random Forest specification. "
                          "The result is useful for forecasting comparison, but it does not independently prove improved inventory-control performance.")
    else:
        interpretation = ("Contextual variables reduced XGBoost validation error compared with the historical-only specification. "
                          "The result supports their forecasting value, but it does not independently prove improved inventory-control performance.")
    model_interpretation = (f"For {selected}, contextual variables changed validation MAE from {historical.mae:,.2f} "
                            f"to {context.mae:,.2f} ({improvement:.1f}% improvement).")
    st.markdown(f'<div class="interpretation">{model_interpretation}</div>', unsafe_allow_html=True)

    show_chart(comparison_chart(metrics, selected, "Walmart"), "walmart-model-comparison")
    col1, col2 = st.columns(2)
    with col1:
        show_chart(pair_chart(selected, float(context.mae), float(historical.mae), "Walmart"), "walmart-selected-mae")
    with col2:
        # Shared holdout comparison exists only for the Random Forest candidate.
        if model_key == "random_forest":
            common = holdout[(holdout.split == "final_test_common_naive_coverage") & holdout.run_id.isin(["random_forest__historical_plus_context", "seasonal_naive_52w"])]
            selected_row = common[common.run_id == "random_forest__historical_plus_context"].iloc[0]
            naive_row = common[common.run_id == "seasonal_naive_52w"].iloc[0]
            fig = go.Figure(go.Bar(x=[selected_row.mae, naive_row.mae], y=["Random Forest", "Seasonal-naive"], orientation="h",
                marker_color=[BLUE, ORANGE], text=[f"{selected_row.mae:,.2f}", f"{naive_row.mae:,.2f}"], textposition="outside", cliponaxis=False,
                customdata=[["Random Forest", selected_row.wape_pct, int(selected_row.rows_scored)], ["Seasonal-naive", naive_row.wape_pct, int(naive_row.rows_scored)]],
                hovertemplate="Dataset: Walmart<br>Model: %{customdata[0]}<br>Feature specification: Shared holdout<br>Metric: Shared holdout MAE<br>MAE: %{x:,.2f}<br>Shared holdout WAPE: %{customdata[1]:.2f}%<br>Comparison rows: %{customdata[2]:,}<extra></extra>"))
            fig.update_layout(title="Shared holdout: model versus seasonal-naive", xaxis_title="Shared holdout MAE (lower is better)", yaxis_title="Model", yaxis=dict(autorange="reversed"), showlegend=False)
            show_chart(chart_layout(fig, 350), "walmart-seasonal-comparison")
            st.caption("The seasonal-naive benchmark performed slightly better on shared Walmart holdout rows.")
        else:
            st.info("Shared holdout comparison values are available for Random Forest only; no XGBoost shared-holdout score is in the result files.")

    # Monthly WAPE is computed from model-specific numeric validation predictions;
    # seasonal-naive has validation data but no XGBoost final-holdout counterpart.
    pred_path = WALMART / "predictions" / "validation" / "all_candidate_validation_predictions.parquet"
    pred = required(pred_path, "parquet")
    subset = pred[(pred.Model == model_key) & (pred.FeatureSet == "historical_plus_context")].copy()
    subset["abs_err"] = (subset.ActualSales - subset.PredictedSales).abs()
    subset["abs_actual"] = subset.ActualSales.abs()
    summary = subset.groupby("TargetMonth", as_index=False).agg(abs_err=("abs_err", "sum"), abs_actual=("abs_actual", "sum"), rows=("ActualSales", "size"))
    summary["WAPE"] = 100 * summary.abs_err / summary.abs_actual
    naive = pred[pred.Model == "seasonal_naive_52w"].copy()
    naive["abs_err"] = (naive.ActualSales - naive.PredictedSales).abs()
    naive["abs_actual"] = naive.ActualSales.abs()
    naive_summary = naive.groupby("TargetMonth", as_index=False).agg(abs_err=("abs_err", "sum"), abs_actual=("abs_actual", "sum"), rows=("ActualSales", "size"))
    naive_summary["WAPE"] = 100 * naive_summary.abs_err / naive_summary.abs_actual
    monthly_fig = go.Figure()
    monthly_fig.add_trace(go.Scatter(x=summary.TargetMonth, y=summary.WAPE, name=selected, mode="lines+markers", line=dict(color=BLUE, width=3),
        customdata=summary.rows, hovertemplate=f"Dataset: Walmart<br>Model: {selected}<br>Metric: WAPE<br>Target month: %{{x}}<br>WAPE: %{{y:.2f}}%<br>Rows: %{{customdata:,}}<extra></extra>"))
    monthly_fig.add_trace(go.Scatter(x=naive_summary.TargetMonth, y=naive_summary.WAPE, name="Seasonal-naive", mode="lines+markers", line=dict(color=ORANGE, width=2),
        customdata=naive_summary.rows, hovertemplate="Dataset: Walmart<br>Model: Seasonal-naive<br>Metric: WAPE<br>Target month: %{x}<br>WAPE: %{y:.2f}%<br>Rows: %{customdata:,}<extra></extra>"))
    monthly_fig.update_layout(title=f"Monthly validation WAPE · {selected} and seasonal-naive", xaxis_title="Target month", yaxis_title="WAPE (%)")
    show_chart(chart_layout(monthly_fig, 360), "walmart-monthly-wape")

    # Interactive actual-versus-forecast data: all candidate validation forecasts;
    # final holdout predictions are available only for Random Forest.
    forecast_frame = subset.copy()
    forecast_frame = forecast_frame.rename(columns={"ActualSales": "ActualSales", "PredictedSales": "PredictedSales"})
    if model_key == "random_forest":
        holdout_path = WALMART / "predictions" / "final_holdout" / "selected_model_predictions.parquet"
        forecast_frame = pd.concat([forecast_frame, required(holdout_path, "parquet")], ignore_index=True)
    if not forecast_frame.empty:
        show_chart(forecast_chart(forecast_frame, "Walmart", selected, "2012-06-30", "2012-07-06" if model_key == "random_forest" else None), "walmart-actual-forecast")
    else:
        st.info(f"Interactive actual-versus-forecast data is not available for {selected}.")

    bottom_section(WALMART_METHOD, interpretation, key="methodology-walmart")


if __name__ == "__main__":
    from dashboard_components import render_application

    render_application(default_dataset="Walmart")
