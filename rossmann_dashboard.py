from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_components import (
    BLUE, NAVY, ORANGE, ROSSMANN, ROSSMANN_METHOD, bottom_section, chart_layout,
    comparison_chart, forecast_chart, hero, kpi_row, pair_chart, required, show_chart, style,
)

def render_dashboard() -> None:
    hero("Rossmann Retail Demand Forecasting Dashboard", "Evaluation of contextual machine-learning forecasting using chronological validation", "Rossmann")

    metrics = required(ROSSMANN / "figures" / "research" / "validation_candidate_metrics.csv")
    selected = st.selectbox("Selected model", ["XGBoost", "Random Forest"], index=0, key="rossmann-model")
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
    interpretation = ("Adding contextual variables reduced validation error compared with the historical-only specification. "
                      "The result supports the usefulness of contextual information for demand forecasting, but the dataset does not directly measure inventory outcomes.")
    st.markdown(f'<div class="interpretation">{interpretation}</div>', unsafe_allow_html=True)

    show_chart(comparison_chart(metrics, selected, "Rossmann"), "rossmann-model-comparison")
    col1, col2 = st.columns(2)
    with col1:
        show_chart(pair_chart(selected, float(context.mae), float(historical.mae), "Rossmann"), "rossmann-selected-mae")
    with col2:
        selected_pred = required(ROSSMANN / "predictions" / "validation" / f"{model_key}__historical_plus_context.parquet", "parquet")
        selected_pred["abs_err"] = (selected_pred.ActualSales - selected_pred.PredictedSales).abs()
        selected_pred["abs_actual"] = selected_pred.ActualSales.abs()
        selected_pred["period"] = "Validation"
        if model_key == "xgboost":
            holdout_path = ROSSMANN / "predictions" / "final_test" / "selected_final_predictions.parquet"
            if holdout_path.is_file():
                final_pred = required(holdout_path, "parquet")
                final_pred["abs_err"] = (final_pred.ActualSales - final_pred.PredictedSales).abs()
                final_pred["abs_actual"] = final_pred.ActualSales.abs()
                final_pred["period"] = "Final holdout"
                selected_pred = pd.concat([selected_pred, final_pred], ignore_index=True)
        wape_month = selected_pred.groupby(["TargetMonth", "period"], as_index=False).agg(abs_err=("abs_err", "sum"), abs_actual=("abs_actual", "sum"), rows=("ActualSales", "size"))
        wape_month["wape_pct"] = 100 * wape_month.abs_err / wape_month.abs_actual
        fig = go.Figure()
        for period, color in [("Validation", NAVY), ("Final holdout", ORANGE)]:
            rows = wape_month[wape_month.period == period]
            if not rows.empty:
                fig.add_trace(go.Bar(x=rows.TargetMonth, y=rows.wape_pct, name=period, marker_color=color,
                    customdata=[[selected, period, int(r)] for r in rows.rows],
                    hovertemplate="Dataset: Rossmann<br>Model: %{customdata[0]}<br>Metric: WAPE<br>Month: %{x}<br>Period type: %{customdata[1]}<br>WAPE: %{y:.2f}%<br>Rows: %{customdata[2]:,}<extra></extra>"))
        fig.update_layout(title=f"Monthly WAPE · {selected}", xaxis_title="Target month", yaxis_title="WAPE (%)", barmode="group")
        show_chart(chart_layout(fig, 350), "rossmann-monthly-wape")

    # The archived project contains model-specific validation predictions for both models;
    # final holdout predictions exist only for the selected XGBoost candidate.
    pred_path = ROSSMANN / "predictions" / "validation" / f"{model_key}__historical_plus_context.parquet"
    if pred_path.is_file():
        pred = required(pred_path, "parquet")
        if model_key == "xgboost":
            holdout_path = ROSSMANN / "predictions" / "final_test" / "selected_final_predictions.parquet"
            if holdout_path.is_file():
                pred = pd.concat([pred, required(holdout_path, "parquet")], ignore_index=True)
        show_chart(forecast_chart(pred, "Rossmann", selected, "2015-03-31", "2015-04-01" if model_key == "xgboost" else None), "rossmann-actual-forecast")
    else:
        st.info(f"Interactive actual-versus-forecast data is not available for {selected}.")

    bottom_section(ROSSMANN_METHOD, interpretation, key="methodology-rossmann")


if __name__ == "__main__":
    from dashboard_components import render_application

    render_application(default_dataset="Rossmann")
