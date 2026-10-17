"""Shared presentation and data helpers for the standalone dashboards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
BLUE, NAVY, PALE, ORANGE, HIST = "#2874D9", "#172B46", "#F0F4FA", "#00897D", "#526B89"

ROSSMANN = ROOT / "rossmann-sales-forecasting" / "experiment" / "outputs"
WALMART = ROOT / "walmart-sales-forecast" / "experiment" / "outputs"

ROSSMANN_METHOD = (
    "The Rossmann models were evaluated using chronological training, validation and final holdout periods. "
    "MAE and WAPE compare forecasting performance. Historical-only and historical-plus-context specifications "
    "were compared. Rossmann contains sales and contextual forecasting variables but no direct inventory-control "
    "outcomes such as stockouts, reorder points, safety stock, holding costs or service levels."
)
WALMART_METHOD = (
    "The Walmart models were evaluated using chronological training, validation and final holdout periods. "
    "MAE and WAPE compare forecasting performance. Historical-only and historical-plus-context specifications "
    "were compared. Walmart contains sales and contextual forecasting variables but no direct inventory-control "
    "outcomes such as stockouts, reorder points, safety stock, holding costs or service levels."
)
INVENTORY_LIMIT = (
    "The datasets support demand forecasting analysis but do not support direct measurement of inventory-control "
    "improvement. Neither dataset contains complete product-level stock positions, replenishment records, stockout "
    "events, supplier lead times, holding costs or service-level targets. Therefore, inventory decision-making is "
    "discussed as a planning implication rather than a directly measured operational outcome."
)


def required(path: Path, kind: str = "csv") -> Any:
    if not path.is_file():
        st.error(f"Required result file is missing: `{path.relative_to(ROOT)}`. Restore the experiment output and reload this dashboard.")
        st.stop()
    try:
        if kind == "json":
            return json.loads(path.read_text(encoding="utf-8"))
        if kind == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read `{path.relative_to(ROOT)}`: {exc}")
        st.stop()


def style() -> None:
    st.markdown("""
    <style>
    .stApp { background: #F3F5F9; color: #172B46; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stAppViewContainer"] { background:#F3F5F9; }
    [data-testid="stHeader"] { background: transparent; }
    .stMainBlockContainer { max-width: 1440px; padding: 2.5rem 3rem 3rem; }
    h1,h2,h3,p,label { color: #172B46; }
    .hero { background:linear-gradient(115deg,#13273F 20%,#214B60 100%); padding:40px 42px; border:1px solid #21445A; border-radius:20px; margin-bottom:16px; box-shadow:0 12px 30px rgba(19,39,63,.12); border-bottom:4px solid #49BEAD; }
    .eyebrow { color:#87E0D2; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:18px; }
    .hero h1 { color:#FFFFFF; font-size:clamp(1.7rem,3vw,2.85rem); font-weight:700; letter-spacing:-.045em; line-height:1.15; margin:0 0 18px; padding:0; max-width:1000px; }
    .hero p { color:#D5E2EE; font-size:1rem; line-height:1.65; margin:0; }
    .kpi { background:#fff; border:1px solid #E1E7EF; border-radius:14px; padding:22px 20px; min-height:138px; box-shadow:0 4px 12px rgba(27,44,73,.025); }
    .kpi-label { color:#5C6D82; font-size:.78rem; font-weight:600; line-height:1.4; min-height:3em; }
    .kpi-value { color:#172B46; font-size:clamp(1.35rem,2vw,1.85rem); font-weight:700; line-height:1.25; letter-spacing:-.035em; font-variant-numeric:tabular-nums; }
    .kpi-accent { background:#E5F5F0; border-color:#BDE3D6; }
    .kpi-accent .kpi-value { color:#087664; }
    .interpretation { background:#EAF0FB; border:1px solid #D8E2F5; border-left:3px solid #3563E9; border-radius:12px; padding:20px 24px; margin:8px 0 14px; color:#314B70; font-size:.94rem; line-height:1.7; }
    .interpretation::before { content:'KEY FINDING'; display:block; color:#3563E9; font-size:.68rem; font-weight:700; letter-spacing:.12em; margin-bottom:5px; }
    [data-testid="stPlotlyChart"] { background:white; border:1px solid #E1E7EF; border-radius:16px; overflow:hidden; box-shadow:0 4px 18px rgba(27,44,73,.025); }
    [data-testid="stSelectbox"] { max-width:340px; margin-bottom:4px; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#FFFFFF; color:#172B46; border-color:#D5DEEB; border-radius:10px; min-height:46px; }
    [data-testid="stCaptionContainer"] p { color:#697B92; font-size:.8rem; }
    [data-testid="stExpander"] { background:#fff; border:1px solid #E1E7EF; border-radius:12px; color:#172B46; }
    [data-testid="stExpander"] summary { color:#172B46; }
    .st-key-methodology-rossmann, .st-key-methodology-walmart { background:#FFFFFF; border:1px solid #E1E7EF; border-radius:16px; padding:26px; margin-top:20px; }
    .st-key-methodology-rossmann h3, .st-key-methodology-walmart h3 { font-size:1.15rem; letter-spacing:-.02em; }
    [data-testid="stAlert"] { background:#EDF3FB; color:#314B70; border-radius:12px; }
    @media (max-width: 800px) {
        .stMainBlockContainer { padding:1.5rem 1rem; }
        .hero { padding:24px; }
        .kpi { padding:16px; min-height:120px; }
    }
    </style>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, dataset: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{dataset} · Demand forecasting</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items), gap="small")
    for col, (label, value) in zip(cols, items):
        with col:
            accent = " kpi-accent" if label == "Improvement from contextual variables" else ""
            st.markdown(f'<div class="kpi{accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=NAVY),
        title_font=dict(size=16, color=NAVY), margin=dict(l=28, r=60, t=76, b=48),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_color=NAVY),
        legend=dict(font=dict(color=HIST, size=11), orientation="h", y=-.22, x=0),
        bargap=.42,
    )
    fig.update_xaxes(title_font=dict(color=HIST, size=11), tickfont=dict(color=HIST), gridcolor="#EDF1F6", zeroline=False, automargin=True)
    fig.update_yaxes(title_font=dict(color=HIST, size=11), tickfont=dict(color=HIST), gridcolor="#EDF1F6", zeroline=False, automargin=True)
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, width="stretch", theme=None, config={"displayModeBar": True, "displaylogo": False, "responsive": True, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2}})


def bottom_section(method: str, interpretation: str, key: str = "methodology") -> None:
    with st.container(key=key):
        st.markdown("### Methodology and interpretation")
        with st.expander("Methodology notes"):
            st.write(method)
        with st.expander("Interpretation"):
            st.write(interpretation)
        with st.expander("Inventory evidence limitation"):
            st.write(INVENTORY_LIMIT)
        st.caption("Source: Author’s analysis of the Rossmann and Walmart retail forecasting datasets.")
    st.markdown(
        '<footer style="text-align:center; color:#526B89; font-size:0.85rem; '
        'padding:24px 0 8px; margin-top:16px; border-top:1px solid #E1E7EF;">'
        'Research by Siddhesh Sawant</footer>',
        unsafe_allow_html=True,
    )


def render_application(default_dataset: str = "Rossmann") -> None:
    """One page with independent dataset tabs and model selections."""
    from rossmann_dashboard import render_dashboard as render_rossmann
    from walmart_dashboard import render_dashboard as render_walmart

    st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📊", layout="wide")
    style()
    rossmann_tab, walmart_tab = st.tabs(["Rossmann", "Walmart"], default=default_dataset)
    with rossmann_tab:
        render_rossmann()
    with walmart_tab:
        render_walmart()


def comparison_chart(metrics: pd.DataFrame, selected: str, dataset: str) -> go.Figure:
    models = ["xgboost", "random_forest"]
    labels, vals, colors, outlines, hover = [], [], [], [], []
    for model in models:
        for fs in ["historical_only", "historical_plus_context"]:
            row = metrics[(metrics.model == model) & (metrics.feature_set == fs)].iloc[0]
            labels.append(f"{model.replace('_',' ').title()} · {'Historical-only' if fs == 'historical_only' else 'Historical + context'}")
            vals.append(float(row.mae)); colors.append(HIST if fs == "historical_only" else ORANGE)
            outlines.append(BLUE if model == selected.lower().replace(" ", "_") else "rgba(0,0,0,0)")
            hover.append(f"Dataset: {dataset}<br>Model: {model.replace('_',' ').title()}<br>Feature specification: {'Historical-only' if fs == 'historical_only' else 'Historical + context'}<br>Metric: Validation MAE<br>MAE: {row.mae:,.2f}<br>WAPE: {row.wape_pct:.2f}%<br>Validation rows: {int(row.rows):,}")
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=colors, line=dict(color=outlines, width=[4 if c == BLUE else 0 for c in outlines])), text=[f"{v:,.2f}" for v in vals], textposition="outside", cliponaxis=False, customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(title="Validation model comparison", xaxis_title="Validation MAE (lower is better)", yaxis_title="Model and feature specification", yaxis=dict(autorange="reversed"), showlegend=False)
    return chart_layout(fig, 400)


def pair_chart(model: str, context_mae: float, hist_mae: float, dataset: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=[hist_mae, context_mae], y=["Historical-only", "Historical + context"], orientation="h", marker_color=[HIST, ORANGE], text=[f"{hist_mae:,.2f}", f"{context_mae:,.2f}"], textposition="outside", cliponaxis=False,
        customdata=[[dataset, model, "Historical-only"], [dataset, model, "Historical + context"]], hovertemplate="Dataset: %{customdata[0]}<br>Model: %{customdata[1]}<br>Feature specification: %{customdata[2]}<br>Metric: Validation MAE<br>MAE: %{x:,.2f}<extra></extra>"))
    fig.update_layout(title="Validation MAE<br><sup>Historical-only versus contextual specification</sup>", xaxis_title="Validation MAE", yaxis_title="Feature specification", yaxis=dict(autorange="reversed"), showlegend=False)
    return chart_layout(fig, 350)


def aggregate_predictions(frame: pd.DataFrame, dataset: str, model: str, period: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["TargetDate"] = pd.to_datetime(frame["TargetDate"])
    # Aggregate actual and forecast over entities for each target date.
    return frame.groupby("TargetDate", as_index=False).agg(Actual=("ActualSales", "sum"), Forecast=("PredictedSales", "sum"))


def forecast_chart(frame: pd.DataFrame, dataset: str, model: str, validation_end: str, holdout_start: str | None = None) -> go.Figure:
    if frame.empty:
        return go.Figure()
    daily = aggregate_predictions(frame, dataset, model, "")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.TargetDate, y=daily.Actual, name="Actual aggregate sales", mode="lines", line=dict(color=NAVY, width=2), hovertemplate=f"Dataset: {dataset}<br>Model: {model}<br>Metric: Aggregate sales<br>Period: %{{x|%Y-%m-%d}}<br>Actual: %{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=daily.TargetDate, y=daily.Forecast, name=f"{model} forecast", mode="lines", line=dict(color=BLUE, width=2), hovertemplate=f"Dataset: {dataset}<br>Model: {model}<br>Feature specification: Historical + context<br>Metric: Forecast aggregate sales<br>Period: %{{x|%Y-%m-%d}}<br>Forecast: %{{y:,.0f}}<extra></extra>"))
    if holdout_start:
        fig.add_vline(x=pd.Timestamp(holdout_start), line_dash="dash", line_color=ORANGE, annotation_text="Holdout starts", annotation_font_color=NAVY)
    fig.update_layout(title="Actual versus forecast aggregate sales", xaxis_title="Target date", yaxis_title="Aggregate sales", hovermode="x unified")
    return chart_layout(fig, 390)
