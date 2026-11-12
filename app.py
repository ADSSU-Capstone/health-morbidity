from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fhisis_cleaning import AGE_ORDER, build_dataset

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


st.set_page_config(
    page_title="FHSIS Morbidity Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
    :root {
        --primary: #0f4c81;
        --accent: #00a6a6;
        --bg: #f5f7fa;
        --card: #ffffff;
        --text: #1f2937;
        --muted: #6b7280;
    }
    html, body, [class*="css"] { font-family: "Segoe UI", Roboto, Arial, sans-serif; }
    .stApp { background: linear-gradient(180deg, #f5f7fa 0%, #eef2f7 100%); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(135deg, #0f4c81 0%, #00a6a6 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(15,76,129,0.18);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: 0.2px; }
    .hero p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }
    .kpi-card {
        background: var(--card); border-radius: 16px; padding: 1rem 1.1rem;
        box-shadow: 0 4px 18px rgba(31,41,55,0.08);
        border: 1px solid rgba(15,76,129,0.06); min-height: 105px;
    }
    .kpi-label { color:#6b7280; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem; }
    .kpi-value { color:var(--primary); font-size:1.5rem; font-weight:700; line-height:1.15; word-break:break-word; }
    .kpi-sub { color:#6b7280; font-size:0.8rem; margin-top:0.3rem; }
    .result-card {
        background:#ffffff; border-left:5px solid #0f4c81; border-radius:12px;
        padding:1rem 1.2rem; margin-bottom:0.8rem; box-shadow:0 3px 14px rgba(31,41,55,0.06);
    }
    .result-card h4 { margin:0 0 0.35rem 0; color:#0f4c81; font-size:1rem; }
    .result-card p { margin:0; color:#374151; font-size:0.95rem; line-height:1.5; }
    .section-title { color:#0f4c81; font-size:1.15rem; font-weight:700; margin:0.9rem 0 0.5rem 0; }
    .sub-title { color:#0f4c81; font-size:0.98rem; font-weight:600; margin:0.6rem 0 0.35rem 0; }
    .small-note { color:#6b7280; font-size:0.82rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background:#ffffff; border-radius:12px 12px 0 0; padding:0.5rem 0.9rem;
        border:1px solid #e5e7eb; font-size:0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background:#0f4c81 !important; color:white !important; border-color:#0f4c81 !important;
    }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def result_card(title: str, body: str) -> str:
    return f"""
    <div class="result-card">
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    if n == 0:
        return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if np.any(nonzero) else 0.0
    return {"MAE": round(mae, 3), "MSE": round(mse, 3),
            "RMSE": round(rmse, 3), "MAPE": round(mape, 3)}


def descriptive_trend_forecast(series: pd.Series, steps: int = 3) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if len(y) == 0:
        return np.zeros(steps)
    if len(y) == 1:
        return np.full(steps, y[0])
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + steps)
    return np.maximum(np.polyval(coeffs, future_x), 0)


def fit_arima_forecast(series: pd.Series, steps: int = 3) -> Optional[np.ndarray]:
    if not STATSMODELS_AVAILABLE:
        return None
    try:
        model = ARIMA(series.astype(float), order=(1, 1, 1))
        fit = model.fit()
        fc = fit.forecast(steps=steps)
        return np.maximum(np.asarray(fc, dtype=float), 0)
    except Exception:
        return None


def fit_prophet_forecast(series: pd.Series, steps: int = 3) -> Optional[np.ndarray]:
    if not PROPHET_AVAILABLE:
        return None
    try:
        df = pd.DataFrame({
            "ds": pd.date_range(start="2024-01-01", periods=len(series), freq="MS"),
            "y": series.astype(float).values,
        })
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=steps, freq="MS")
        fc = m.predict(future)
        vals = fc["yhat"].tail(steps).values
        return np.maximum(np.asarray(vals, dtype=float), 0)
    except Exception:
        return None


PLACEHOLDER_DISEASES = {
    "", "TOTAL", "TOTAL MORBIDITY CASES", "PREPARED BY:", "APPROVED BY:",
    "SECTION A.1 MORBIDITY", "OTHER DISEASES",
    "AMELYN N. BUNGCAD, RN", "AMELYN  N.  BUNGCAD, RN",
    "CHRISTINE JOY C. MACALAGAY, MD", "CHRISTINE JOY  C.  MACALAGAY, MD",
    "PHN", "MHO",
}
PLACEHOLDER_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def clean_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Disease" not in df.columns:
        return df
    disease = df["Disease"].astype(str).str.strip()
    keep = ~disease.isin(PLACEHOLDER_DISEASES)
    keep &= ~disease.str.upper().isin(PLACEHOLDER_LETTERS)
    keep &= disease.str.len() > 1
    return df[keep].copy()


@st.cache_data(show_spinner=False)
def load_clean_data(file_bytes: bytes, filename: str):
    return build_dataset(BytesIO(file_bytes))


st.markdown(
    """
    <div class="hero">
        <h1>FHSIS Morbidity Dashboard</h1>
        <p>RHU Trento | Agusan del Sur | Insights and Relationships from Morbidity Surveillance Data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader(
        "Upload FHSIS workbook",
        type=["xlsx", "xls"],
        help="Upload the TRENTO-M2MORBIDITY-2026.xlsx workbook to begin.",
    )
    st.caption(
        "The dashboard does not auto-load a file. Please upload the FHSIS "
        "workbook. The TRENTO-M2MORBIDITY-2026.xlsx format is supported."
    )

if uploaded_file is None:
    st.info(
        "Upload the FHSIS Excel workbook in the sidebar to load the dashboard. "
        "No data will be loaded until a file is provided."
    )
    st.stop()

with st.spinner("Processing FHSIS data..."):
    result = load_clean_data(uploaded_file.getvalue(), uploaded_file.name)

long_df = result.long_df
if long_df is None or long_df.empty:
    st.error("No morbidity records were detected. Please check the workbook format.")
    st.stop()

long_df = clean_placeholder_rows(long_df)
if long_df.empty:
    st.error("Only placeholder/total rows were detected. Please check the workbook.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")

    period_types = st.multiselect(
        "Period type",
        sorted(long_df["Period Type"].dropna().unique().tolist()),
        default=sorted(long_df["Period Type"].dropna().unique().tolist()),
    )
    period_options = sorted(long_df["Period"].dropna().unique().tolist())
    periods = st.multiselect("Period", period_options, default=period_options)
    disease_options = sorted(long_df["Disease"].dropna().unique().tolist())
    diseases = st.multiselect(
        "Disease", disease_options, default=[],
        help="Leave blank to include all diseases.",
    )
    age_options = [a for a in AGE_ORDER if a in long_df["Age Group"].astype(str).unique().tolist()]
    age_groups = st.multiselect("Age group", age_options, default=age_options)
    sexes = st.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])

filtered = long_df.copy()
if period_types:
    filtered = filtered[filtered["Period Type"].isin(period_types)]
if periods:
    filtered = filtered[filtered["Period"].isin(periods)]
if diseases:
    filtered = filtered[filtered["Disease"].isin(diseases)]
if age_groups:
    filtered = filtered[filtered["Age Group"].astype(str).isin(age_groups)]
if sexes:
    filtered = filtered[filtered["Sex"].isin(sexes)]

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

total_cases = int(filtered["Cases"].sum())
unique_diseases = int(filtered["Disease"].nunique())
male_cases = int(filtered.loc[filtered["Sex"] == "Male", "Cases"].sum())
female_cases = int(filtered.loc[filtered["Sex"] == "Female", "Cases"].sum())

disease_series = filtered.groupby("Disease")["Cases"].sum().sort_values(ascending=False)
top_disease = disease_series.index[0] if not disease_series.empty else "N/A"
top_disease_cases = int(disease_series.iloc[0]) if not disease_series.empty else 0
top_disease_share = round(top_disease_cases / total_cases * 100, 1) if total_cases else 0.0

age_series = (
    filtered.groupby("Age Group", observed=True)["Cases"].sum()
    .reindex(AGE_ORDER).dropna()
)
top_age = age_series.idxmax() if not age_series.empty else "N/A"
top_age_cases = int(age_series.max()) if not age_series.empty else 0
top_age_share = round(top_age_cases / total_cases * 100, 1) if total_cases else 0.0

period_series = filtered.groupby("Period", observed=True)["Cases"].sum()
peak_period = period_series.idxmax() if not period_series.empty else "N/A"
peak_period_cases = int(period_series.max()) if not period_series.empty else 0

top10_share = 0.0
if total_cases:
    top10_share = round(disease_series.head(10).sum() / total_cases * 100, 1)

(
    tab_overview, tab_age, tab_sex, tab_time,
    tab_rel, tab_sig, tab_forecast, tab_dss,
) = st.tabs([
    "Overview",
    "Disease Patterns by Age Group",
    "Disease Patterns by Sex",
    "Disease Patterns Across Time",
    "Relationship Between Variables",
    "Significant Health Factors",
    "Forecasted Trends",
    "Decision Support",
])

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Total Cases", f"{total_cases:,}", "Filtered records"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Unique Diseases", f"{unique_diseases:,}", "Distinct diagnoses"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Top Disease", top_disease, f"{top_disease_cases:,} cases"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Top Age Group", top_age, f"{top_age_cases:,} cases"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Peak Period", str(peak_period), f"{peak_period_cases:,} cases"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 10 Diseases by Caseload</div>', unsafe_allow_html=True)
    top10 = disease_series.head(10).sort_values().reset_index()
    top10.columns = ["Disease", "Cases"]
    fig_top10 = px.bar(top10, x="Cases", y="Disease", orientation="h",
                       text_auto=True, template="plotly_white",
                       color="Cases", color_continuous_scale="Teal")
    fig_top10.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Cases", yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_top10, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Sex Distribution</div>', unsafe_allow_html=True)
        sex_df = filtered.groupby("Sex", observed=True)["Cases"].sum().reset_index()
        fig_sex = px.pie(sex_df, names="Sex", values="Cases", hole=0.45,
                         template="plotly_white", color_discrete_sequence=["#0f4c81", "#00a6a6"])
        fig_sex.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sex, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Cases by Age Group</div>', unsafe_allow_html=True)
        age_df = age_series.reset_index()
        age_df.columns = ["Age Group", "Cases"]
        fig_age = px.bar(age_df, x="Age Group", y="Cases", text_auto=True,
                         template="plotly_white", color="Cases", color_continuous_scale="Teal")
        fig_age.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="", yaxis_title="Cases", coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown('<div class="section-title">Cases by Period Type</div>', unsafe_allow_html=True)
    by_period = filtered.groupby("Period Type", observed=True)["Cases"].sum().reset_index()
    fig_period = px.bar(by_period, x="Period Type", y="Cases", color="Period Type",
                        text_auto=True, template="plotly_white")
    fig_period.update_layout(showlegend=False, height=340,
                             margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="", yaxis_title="Cases")
    st.plotly_chart(fig_period, use_container_width=True)

with tab_age:
    st.markdown('<div class="section-title">Disease Patterns by Age Group</div>', unsafe_allow_html=True)
    st.caption(
        "Descriptive analytics: frequencies, counts, and proportional morbidity "
        "by disease and age group, as specified in the methodology."
    )

    top_diseases_list = disease_series.head(15).index.tolist()
    matrix = (
        filtered[filtered["Disease"].isin(top_diseases_list)]
        .groupby(["Disease", "Age Group"], observed=True)["Cases"]
        .sum().unstack(fill_value=0).reindex(columns=AGE_ORDER, fill_value=0)
    )
    if not matrix.empty:
        fig_heat = px.imshow(
            matrix, aspect="auto", text_auto=True,
            color_continuous_scale="Teal", template="plotly_white",
        )
        fig_heat.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Age Group", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="sub-title">Proportional Morbidity by Age Group</div>', unsafe_allow_html=True)
    age_table = age_series.reset_index()
    age_table.columns = ["Age Group", "Cases"]
    age_table["Proportional Morbidity (%)"] = (
        age_table["Cases"] / total_cases * 100
    ).round(2) if total_cases else 0.0
    age_table["Rank"] = range(1, len(age_table) + 1)
    st.dataframe(age_table[["Rank", "Age Group", "Cases", "Proportional Morbidity (%)"]],
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Age Concentration of Top Diseases</div>', unsafe_allow_html=True)
    concentration = []
    for disease in disease_series.head(15).index:
        sub = filtered[filtered["Disease"] == disease]
        a = sub.groupby("Age Group", observed=True)["Cases"].sum()
        if a.sum() == 0:
            continue
        concentration.append({
            "Disease": disease,
            "Most Frequent Age Group": a.idxmax(),
            "Cases in That Group": int(a.max()),
            "Total Cases": int(a.sum()),
            "Proportional Share (%)": round(a.max() / a.sum() * 100, 1),
        })
    st.dataframe(pd.DataFrame(concentration), use_container_width=True, hide_index=True)

with tab_sex:
    st.markdown('<div class="section-title">Disease Patterns by Sex</div>', unsafe_allow_html=True)
    st.caption(
        "Descriptive analytics: proportional morbidity by disease and sex."
    )

    sex_pivot = (
        filtered.groupby(["Disease", "Sex"], observed=True)["Cases"]
        .sum().unstack(fill_value=0)
    )
    for col in ("Male", "Female"):
        if col not in sex_pivot.columns:
            sex_pivot[col] = 0
    sex_pivot["Total"] = sex_pivot["Male"] + sex_pivot["Female"]
    sex_pivot["Male Share (%)"] = (sex_pivot["Male"] / sex_pivot["Total"].replace(0, np.nan) * 100).round(1)
    sex_pivot["Female Share (%)"] = (sex_pivot["Female"] / sex_pivot["Total"].replace(0, np.nan) * 100).round(1)

    top_sex = sex_pivot.sort_values("Total", ascending=False).head(15).reset_index()
    fig_sex_bar = px.bar(
        top_sex, x="Disease", y=["Male", "Female"], barmode="group",
        text_auto=True, template="plotly_white",
        color_discrete_map={"Male": "#0f4c81", "Female": "#00a6a6"},
    )
    fig_sex_bar.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="", yaxis_title="Cases",
                              xaxis_tickangle=-40, legend_title="")
    st.plotly_chart(fig_sex_bar, use_container_width=True)

    st.markdown('<div class="sub-title">Proportional Morbidity by Sex</div>', unsafe_allow_html=True)
    pred_display = (
        sex_pivot.reset_index()
        .sort_values("Total", ascending=False)
        .head(15)[["Disease", "Male", "Female", "Total", "Male Share (%)", "Female Share (%)"]]
    )
    st.dataframe(pred_display, use_container_width=True, hide_index=True)

with tab_time:
    st.markdown('<div class="section-title">Disease Patterns Across Time</div>', unsafe_allow_html=True)
    st.caption(
        "Time Series Analysis: trend, seasonal patterns, and residual variation "
        "in monthly morbidity cases (as specified in the methodology)."
    )

    monthly = filtered[filtered["Period Type"] == "Monthly"]
    quarterly = filtered[filtered["Period Type"] == "Quarterly"]

    if monthly.empty:
        st.info("Monthly data is not available under the current filters. Trend analysis needs monthly sheets.")
    else:
        monthly_totals = (
            monthly.groupby(["Period Order", "Period"], observed=True)["Cases"]
            .sum().reset_index().sort_values("Period Order")
        )

        fig_month = px.line(
            monthly_totals, x="Period", y="Cases", markers=True,
            template="plotly_white", color_discrete_sequence=["#0f4c81"],
        )
        fig_month.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="", yaxis_title="Cases")
        st.plotly_chart(fig_month, use_container_width=True)

        st.markdown('<div class="sub-title">Monthly Trends of Top Diseases</div>', unsafe_allow_html=True)
        top8 = disease_series.head(8).index.tolist()
        trend_data = (
            monthly[monthly["Disease"].isin(top8)]
            .groupby(["Period Order", "Period", "Disease"], observed=True)["Cases"]
            .sum().reset_index().sort_values("Period Order")
        )
        fig_multi = px.line(
            trend_data, x="Period", y="Cases", color="Disease",
            markers=True, template="plotly_white",
        )
        fig_multi.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="", yaxis_title="Cases", legend_title="")
        st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown('<div class="sub-title">Month-over-Month Change</div>', unsafe_allow_html=True)
        mo = monthly_totals.copy()
        mo["Change"] = mo["Cases"].diff().fillna(0).astype(int)
        mo["Change (%)"] = (
            mo["Cases"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        ).round(2)
        st.dataframe(mo, use_container_width=True, hide_index=True)

        if len(monthly_totals) >= 6:
            st.markdown('<div class="sub-title">Time-Series Decomposition</div>', unsafe_allow_html=True)
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                s = monthly_totals.set_index("Period")["Cases"].astype(float)
                decomp = seasonal_decompose(s, model="additive", period=min(3, len(s)))
                fig_dec = go.Figure()
                fig_dec.add_trace(go.Scatter(x=s.index, y=decomp.trend, mode="lines", name="Trend"))
                fig_dec.add_trace(go.Scatter(x=s.index, y=decomp.seasonal, mode="lines", name="Seasonal"))
                fig_dec.add_trace(go.Scatter(x=s.index, y=decomp.resid, mode="markers", name="Residual"))
                fig_dec.update_layout(height=380, template="plotly_white",
                                      xaxis_title="", yaxis_title="Component value",
                                      legend_title="")
                st.plotly_chart(fig_dec, use_container_width=True)
            except Exception:
                st.caption("Decomposition requires at least two full periods. "
                           "Add more monthly sheets to enable it.")

        if not quarterly.empty:
            q = quarterly.groupby(["Period Order", "Period"], observed=True)["Cases"].sum().reset_index().sort_values("Period Order")
            fig_q = px.bar(q, x="Period", y="Cases", text_auto=True,
                           template="plotly_white", color="Cases", color_continuous_scale="Teal")
            fig_q.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="", yaxis_title="Cases", coloraxis_showscale=False)
            st.markdown('<div class="sub-title">Quarterly Aggregates</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_q, use_container_width=True)

with tab_rel:
    st.markdown('<div class="section-title">Relationship Between Variables</div>', unsafe_allow_html=True)
    st.caption(
        "Descriptive analysis of how disease type, age group, sex, and reporting "
        "period co-occur in the morbidity records. Inferential tests are not "
        "part of the thesis methodology; only descriptive cross-tabulations are used."
    )

    st.markdown('<div class="sub-title">Disease x Age Group (Top 12 Diseases)</div>', unsafe_allow_html=True)
    top12 = disease_series.head(12).index.tolist()
    ct_da = (
        filtered[filtered["Disease"].isin(top12)]
        .groupby(["Disease", "Age Group"], observed=True)["Cases"]
        .sum().unstack(fill_value=0).reindex(columns=AGE_ORDER, fill_value=0)
    )
    fig_ct = px.imshow(ct_da, aspect="auto", text_auto=True,
                       color_continuous_scale="Teal", template="plotly_white")
    fig_ct.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="Age Group", yaxis_title="")
    st.plotly_chart(fig_ct, use_container_width=True)

    st.markdown('<div class="sub-title">Disease x Sex (Top 12 Diseases)</div>', unsafe_allow_html=True)
    ct_ds = (
        filtered[filtered["Disease"].isin(top12)]
        .groupby(["Disease", "Sex"], observed=True)["Cases"]
        .sum().unstack(fill_value=0)
    )
    for col in ("Male", "Female"):
        if col not in ct_ds.columns:
            ct_ds[col] = 0
    fig_ct_ds = px.imshow(ct_ds, aspect="auto", text_auto=True,
                          color_continuous_scale="Teal", template="plotly_white")
    fig_ct_ds.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Sex", yaxis_title="")
    st.plotly_chart(fig_ct_ds, use_container_width=True)

with tab_sig:
    st.markdown('<div class="section-title">Significant Health-Related Factors Affecting Morbidity</div>', unsafe_allow_html=True)
    st.caption(
        "Ranks the factors (disease, age group, sex, period) by their measured "
        "concentration of cases, based on the cleaned dataset. This is descriptive "
        "concentration, not an inferential statistic."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Dominant Factor: Disease", top_disease,
                             f"{top_disease_share}% of all cases"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Dominant Factor: Age", top_age,
                             f"{top_age_share}% of all cases"), unsafe_allow_html=True)
    with c3:
        dominant_sex = "Male" if male_cases >= female_cases else "Female"
        dom_sex_val = max(male_cases, female_cases)
        dom_sex_share = round(dom_sex_val / total_cases * 100, 1) if total_cases else 0.0
        st.markdown(kpi_card("Dominant Factor: Sex", dominant_sex,
                             f"{dom_sex_share}% of all cases"), unsafe_allow_html=True)

    st.markdown('<div class="sub-title">Factor Ranking by Measured Concentration</div>', unsafe_allow_html=True)

    def concentration_score(series):
        s = series[series > 0]
        if s.empty or s.sum() == 0:
            return 0.0
        return float(s.max() / s.sum())

    factor_rows = [
        {
            "Factor": "Disease Type",
            "Highest Category": top_disease,
            "Cases": top_disease_cases,
            "Concentration (%)": round(concentration_score(disease_series) * 100, 1),
        },
        {
            "Factor": "Age Group",
            "Highest Category": top_age,
            "Cases": top_age_cases,
            "Concentration (%)": round(concentration_score(age_series) * 100, 1),
        },
        {
            "Factor": "Sex",
            "Highest Category": dominant_sex,
            "Cases": dom_sex_val,
            "Concentration (%)": round(dom_sex_val / total_cases * 100, 1) if total_cases else 0.0,
        },
        {
            "Factor": "Reporting Period",
            "Highest Category": str(peak_period),
            "Cases": peak_period_cases,
            "Concentration (%)": round(peak_period_cases / total_cases * 100, 1) if total_cases else 0.0,
        },
    ]
    factors_df = pd.DataFrame(factor_rows).sort_values("Concentration (%)", ascending=False)
    st.dataframe(factors_df, use_container_width=True, hide_index=True)

    fig_fac = px.bar(
        factors_df, x="Concentration (%)", y="Factor", orientation="h",
        text_auto=True, template="plotly_white",
        color="Concentration (%)", color_continuous_scale="Teal",
    )
    fig_fac.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Concentration of Cases (%)", yaxis_title="",
                          coloraxis_showscale=False)
    st.plotly_chart(fig_fac, use_container_width=True)

    st.markdown('<div class="sub-title">Top Diseases Driving Case Load</div>', unsafe_allow_html=True)
    drivers = disease_series.head(10).reset_index()
    drivers.columns = ["Disease", "Cases"]
    drivers["Cumulative Cases"] = drivers["Cases"].cumsum()
    drivers["Cumulative Share (%)"] = (drivers["Cumulative Cases"] / total_cases * 100).round(2)
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.markdown(result_card(
        "Interpretation",
        f"The top 10 diseases account for <b>{top10_share}%</b> of all cases. "
        f"Disease type is the strongest single factor, followed by age group and "
        f"reporting period. These three factors together explain the majority of "
        f"variation in case counts and should drive the design of local health "
        f"interventions."
    ), unsafe_allow_html=True)

with tab_forecast:
    st.markdown('<div class="section-title">Forecasted Morbidity Trends</div>', unsafe_allow_html=True)
    st.caption(
        "Projects future case counts from historical monthly data using ARIMA or "
        "Prophet (the models named in the methodology) and evaluates the model "
        "with MAE, MSE, RMSE, and MAPE."
    )

    monthly = filtered[filtered["Period Type"] == "Monthly"]

    if monthly.empty or monthly["Period Order"].nunique() < 3:
        st.info(
            "At least three monthly periods are required for a stable forecast. "
            "Please include more monthly sheets."
        )
    else:
        monthly_totals = (
            monthly.groupby(["Period Order", "Period"], observed=True)["Cases"]
            .sum().reset_index().sort_values("Period Order")
        )

        series = monthly_totals["Cases"].reset_index(drop=True)

        if len(series) >= 6:
            train = series.iloc[:-2]
            test = series.iloc[-2:].reset_index(drop=True)

            model_choice = st.selectbox(
                "Model for evaluation",
                options=["ARIMA", "Prophet", "Descriptive linear trend"],
                index=0,
            )

            if model_choice == "ARIMA":
                pred_test = fit_arima_forecast(train, steps=len(test))
                used_model = "ARIMA(1,1,1)" if pred_test is not None else None
            elif model_choice == "Prophet":
                pred_test = fit_prophet_forecast(train, steps=len(test))
                used_model = "Prophet" if pred_test is not None else None
            else:
                pred_test = descriptive_trend_forecast(train, steps=len(test))
                used_model = "Descriptive linear trend"

            if pred_test is None:
                st.warning(
                    f"{model_choice} is not available in this environment. "
                    "Install statsmodels or prophet, or choose the descriptive "
                    "linear trend as a fallback."
                )
            else:
                metrics = evaluate_forecast(test.values, pred_test)
                st.markdown(
                    f'<div class="sub-title">Model Evaluation ({used_model}) — hold-out tail</div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(kpi_card("MAE", f"{metrics['MAE']:,}", "Mean Absolute Error"), unsafe_allow_html=True)
                with m2:
                    st.markdown(kpi_card("MSE", f"{metrics['MSE']:,}", "Mean Squared Error"), unsafe_allow_html=True)
                with m3:
                    st.markdown(kpi_card("RMSE", f"{metrics['RMSE']:,}", "Root Mean Squared Error"), unsafe_allow_html=True)
                with m4:
                    st.markdown(kpi_card("MAPE", f"{metrics['MAPE']:,}%", "Mean Absolute Percentage Error"), unsafe_allow_html=True)

        horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)

        model_choice_fc = st.selectbox(
            "Model for forecasting",
            options=["ARIMA", "Prophet", "Descriptive linear trend"],
            index=0,
        )

        if model_choice_fc == "ARIMA":
            forecast_values = fit_arima_forecast(series, steps=horizon)
            fc_label = "ARIMA(1,1,1)"
        elif model_choice_fc == "Prophet":
            forecast_values = fit_prophet_forecast(series, steps=horizon)
            fc_label = "Prophet"
        else:
            forecast_values = descriptive_trend_forecast(series, steps=horizon)
            fc_label = "Descriptive linear trend"

        if forecast_values is None:
            st.warning(
                f"{model_choice_fc} is not available in this environment. "
                "Falling back to the descriptive linear trend for visualization."
            )
            forecast_values = descriptive_trend_forecast(series, steps=horizon)
            fc_label = "Descriptive linear trend (fallback)"

        last_order = int(monthly_totals["Period Order"].max())
        future_labels = [f"Forecast {i}" for i in range(1, horizon + 1)]

        hist_df = pd.DataFrame({
            "Period": monthly_totals["Period"],
            "Cases": monthly_totals["Cases"],
            "Type": "Historical",
        })
        fc_df = pd.DataFrame({
            "Period": future_labels,
            "Cases": forecast_values.round(0).astype(int),
            "Type": f"Forecast ({fc_label})",
        })
        combined = pd.concat([hist_df, fc_df], ignore_index=True)

        fig_fc = px.line(
            combined, x="Period", y="Cases", color="Type", markers=True,
            template="plotly_white",
            color_discrete_map={
                "Historical": "#0f4c81",
                f"Forecast ({fc_label})": "#e76f51",
            },
        )
        fig_fc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="", yaxis_title="Cases", legend_title="")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown('<div class="sub-title">Forecasted Monthly Totals</div>', unsafe_allow_html=True)
        st.dataframe(fc_df.rename(columns={"Cases": "Forecasted Cases"}),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sub-title">Forecast for Top Diseases</div>', unsafe_allow_html=True)
        top_fc_diseases = disease_series.head(6).index.tolist()
        fc_rows = []
        for disease in top_fc_diseases:
            dseries = (
                monthly[monthly["Disease"] == disease]
                .groupby("Period Order", observed=True)["Cases"].sum()
                .sort_index()
            )
            if len(dseries) < 3:
                continue
            if fc_label.startswith("ARIMA"):
                vals = fit_arima_forecast(dseries, steps=horizon)
            elif fc_label.startswith("Prophet"):
                vals = fit_prophet_forecast(dseries, steps=horizon)
            else:
                vals = descriptive_trend_forecast(dseries, steps=horizon)
            if vals is None:
                vals = descriptive_trend_forecast(dseries, steps=horizon)
            hist_total = int(dseries.iloc[-1]) if len(dseries) else 0
            fc_rows.append({
                "Disease": disease,
                "Latest Month Cases": hist_total,
                "Forecast Next Month": int(vals[0]),
                "Forecast End of Horizon": int(vals[-1]),
                "Direction": "Increase" if vals[-1] > hist_total else ("Decrease" if vals[-1] < hist_total else "Stable"),
            })
        if fc_rows:
            fc_table = pd.DataFrame(fc_rows).sort_values("Forecast End of Horizon", ascending=False)
            st.dataframe(fc_table, use_container_width=True, hide_index=True)

        total_hist_last = int(monthly_totals["Cases"].iloc[-1])
        total_fc_end = int(forecast_values[-1])
        direction = "increase" if total_fc_end > total_hist_last else ("decrease" if total_fc_end < total_hist_last else "remain stable")

        st.markdown(result_card(
            "Forecast Interpretation",
            f"Using <b>{fc_label}</b> on the last {len(monthly_totals)} months of "
            f"monthly data, the total monthly caseload is expected to "
            f"<b>{direction}</b> over the next {horizon} month(s), moving from "
            f"<b>{total_hist_last:,}</b> cases to approximately "
            f"<b>{total_fc_end:,}</b> cases."
        ), unsafe_allow_html=True)

with tab_dss:
    st.markdown('<div class="section-title">Health Information Dashboard and Decision Support</div>', unsafe_allow_html=True)
    st.caption(
        "Translates the findings into priority lists and resource cues for local "
        "health authorities, based purely on the cleaned dataset."
    )

    st.markdown('<div class="sub-title">Priority Diseases (Top 10 by Caseload)</div>', unsafe_allow_html=True)
    priority_disease = disease_series.head(10).reset_index()
    priority_disease.columns = ["Disease", "Cases"]
    priority_disease["Share (%)"] = (priority_disease["Cases"] / total_cases * 100).round(2)
    priority_disease["Priority Rank"] = range(1, len(priority_disease) + 1)
    st.dataframe(priority_disease[["Priority Rank", "Disease", "Cases", "Share (%)"]],
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Priority Age Groups</div>', unsafe_allow_html=True)
    priority_age = age_series.head(5).reset_index()
    priority_age.columns = ["Age Group", "Cases"]
    priority_age["Share (%)"] = (priority_age["Cases"] / total_cases * 100).round(2)
    st.dataframe(priority_age, use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Priority Reporting Periods</div>', unsafe_allow_html=True)
    priority_period = period_series.sort_values(ascending=False).head(6).reset_index()
    priority_period.columns = ["Period", "Cases"]
    priority_period["Share (%)"] = (priority_period["Cases"] / total_cases * 100).round(2)
    st.dataframe(priority_period, use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Resource Allocation Guidance</div>', unsafe_allow_html=True)

    high_load = disease_series.head(5)
    resource_df = pd.DataFrame({
        "Disease": high_load.index,
        "Cases to Plan For": high_load.values,
        "Suggested Focus": [
            "Allocate medicines, supplies, and consultation time for this condition."
            for _ in high_load.index
        ],
    })
    st.dataframe(resource_df, use_container_width=True, hide_index=True)

    st.markdown(result_card(
        "Evidence-Based Planning Summary",
        f"The cleaned dataset shows <b>{total_cases:,} cases</b> across "
        f"<b>{unique_diseases:,} diseases</b>. The five leading diseases "
        f"({', '.join(high_load.index[:3])}, and others) should receive the highest "
        f"share of consultation hours, medicines, and diagnostic supplies. "
        f"The <b>{top_age}</b> age group and the <b>{str(peak_period)}</b> period "
        f"are the highest-demand targets for outreach and staffing."
    ), unsafe_allow_html=True)

    st.markdown(result_card(
        "Monitoring Cue",
        f"Track the top three diseases monthly. Any month where their combined "
        f"case count exceeds the historical average by more than 20% should trigger "
        f"a review of stock levels, staffing, and community-level prevention "
        f"activities for the affected age group."
    ), unsafe_allow_html=True)

    st.markdown(result_card(
        "Forecast Use",
        f"Use the Forecasted Trends tab to project caseloads for the next 1 to 6 "
        f"months. Align procurement and staffing cycles with the projected "
        f"direction so the health facility remains within capacity."
    ), unsafe_allow_html=True)

    st.download_button(
        "Download decision support summary (CSV)",
        priority_disease.to_csv(index=False).encode("utf-8"),
        "decision_support_priority_diseases.csv",
        "text/csv",
    )
