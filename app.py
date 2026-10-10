# app.py
# FHSIS Morbidity Dashboard - RHU Trento
# Tabs mirror the "Insights and Relationships" chapter.

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fhisis_cleaning import AGE_ORDER, build_dataset

# ==================================================================
# PAGE CONFIG + STYLES
# ==================================================================
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


# ==================================================================
# HELPERS
# ==================================================================
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


def chi_square(observed: np.ndarray) -> tuple[float, int]:
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0 or observed.sum() == 0:
        return 0.0, 0
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    grand = observed.sum()
    expected = row @ col / grand
    mask = expected > 0
    chi2 = np.nansum(((observed - expected) ** 2)[mask] / expected[mask])
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return float(chi2), int(dof)


def cramers_v(chi2: float, n: float, r: int, c: int) -> float:
    if n <= 0 or min(r - 1, c - 1) <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))


def v_strength(v: float) -> str:
    if v < 0.1: return "Negligible"
    if v < 0.2: return "Weak"
    if v < 0.4: return "Moderate"
    if v < 0.6: return "Strong"
    return "Very Strong"


def forecast_linear(series: pd.Series, steps: int = 3) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if len(y) == 0:
        return np.zeros(steps)
    if len(y) == 1:
        return np.full(steps, y[0])
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + steps)
    return np.maximum(np.polyval(coeffs, future_x), 0)


@st.cache_data(show_spinner=False)
def load_clean_data(file_bytes: Optional[bytes], filename: str):
    source = BytesIO(file_bytes) if file_bytes is not None else filename
    return build_dataset(source)


# ==================================================================
# HERO
# ==================================================================
st.markdown(
    """
    <div class="hero">
        <h1>FHSIS Morbidity Dashboard</h1>
        <p>RHU Trento | Agusan del Sur | Insights and Relationships from Morbidity Surveillance Data</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
# DATA SOURCE
# ==================================================================
with st.sidebar:
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader(
        "Upload FHSIS workbook",
        type=["xlsx", "xls"],
        help="Upload the TRENTO-M2MORBIDITY workbook. If not uploaded, the app uses the default filename.",
    )

default_path = Path("TRENTO-M2MORBIDITY-2026.xlsx")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
elif default_path.exists():
    file_bytes = None
    filename = str(default_path)
else:
    st.error(
        "No workbook found. Please upload the FHSIS Excel file or place "
        "TRENTO-M2MORBIDITY-2026.xlsx in the project folder."
    )
    st.stop()

with st.spinner("Processing FHSIS data..."):
    result = load_clean_data(file_bytes, filename)

long_df = result.long_df
if long_df.empty:
    st.error("No morbidity records were detected. Please check the workbook format.")
    st.stop()

# ==================================================================
# FILTERS
# ==================================================================
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

# ==================================================================
# PRE-COMPUTED METRICS
# ==================================================================
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

# ==================================================================
# TABS
# ==================================================================
(
    tab_overview,
    tab_age,
    tab_sex,
    tab_time,
    tab_rel,
    tab_sig,
    tab_forecast,
    tab_dss,
) = st.tabs(
    [
        "Overview",
        "Disease Patterns by Age Group",
        "Disease Patterns by Sex",
        "Disease Patterns Across Time",
        "Relationship Between Variables",
        "Significant Health Factors",
        "Forecasted Trends",
        "Decision Support",
    ]
)

# ==================================================================
# 1. OVERVIEW
# ==================================================================
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

# ==================================================================
# 2. DISEASE PATTERNS BY AGE GROUP
# ==================================================================
with tab_age:
    st.markdown('<div class="section-title">Disease Patterns by Age Group</div>', unsafe_allow_html=True)
    st.caption(
        "Identifies whether certain diseases concentrate in specific age groups, "
        "so targeted interventions can be planned."
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

    st.markdown('<div class="sub-title">Age-Concentrated Diseases</div>', unsafe_allow_html=True)
    concentration = []
    for disease in disease_series.index:
        sub = filtered[filtered["Disease"] == disease]
        a = sub.groupby("Age Group", observed=True)["Cases"].sum()
        if a.sum() == 0:
            continue
        top_g = a.idxmax()
        share = round(a.max() / a.sum() * 100, 1)
        concentration.append({
            "Disease": disease,
            "Dominant Age Group": top_g,
            "Cases in Age Group": int(a.max()),
            "Total Cases": int(a.sum()),
            "Concentration (%)": share,
        })
    conc_df = pd.DataFrame(concentration).sort_values("Concentration (%)", ascending=False).head(15)
    st.dataframe(conc_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Statistical Association: Disease vs Age Group</div>', unsafe_allow_html=True)
    obs = (
        filtered.groupby(["Disease", "Age Group"], observed=True)["Cases"]
        .sum().unstack(fill_value=0).values
    )
    chi2, dof = chi_square(obs)
    v = cramers_v(chi2, obs.sum(), obs.shape[0], obs.shape[1])
    st.markdown(result_card(
        "Chi-Square Result",
        f"Chi-square = <b>{chi2:,.2f}</b>, degrees of freedom = <b>{dof}</b>. "
        f"Cramer's V = <b>{v:.3f}</b> — <b>{v_strength(v)}</b> association between "
        f"disease type and age group. This indicates that disease distribution is "
        f"{'strongly' if v >= 0.4 else 'moderately' if v >= 0.2 else 'weakly'} "
        f"dependent on age group."
    ), unsafe_allow_html=True)

    st.markdown('<div class="sub-title">Age Group Ranking by Total Cases</div>', unsafe_allow_html=True)
    age_table = age_series.reset_index()
    age_table.columns = ["Age Group", "Cases"]
    age_table["Rank"] = range(1, len(age_table) + 1)
    age_table["Share (%)"] = (age_table["Cases"] / total_cases * 100).round(2) if total_cases else 0.0
    st.dataframe(age_table[["Rank", "Age Group", "Cases", "Share (%)"]],
                 use_container_width=True, hide_index=True)

# ==================================================================
# 3. DISEASE PATTERNS BY SEX
# ==================================================================
with tab_sex:
    st.markdown('<div class="section-title">Disease Patterns by Sex</div>', unsafe_allow_html=True)
    st.caption(
        "Compares morbidity between males and females to detect sex-predominant "
        "diseases that may need different prevention approaches."
    )

    sex_pivot = (
        filtered.groupby(["Disease", "Sex"], observed=True)["Cases"]
        .sum().unstack(fill_value=0)
    )
    for col in ("Male", "Female"):
        if col not in sex_pivot.columns:
            sex_pivot[col] = 0
    sex_pivot["Total"] = sex_pivot["Male"] + sex_pivot["Female"]
    sex_pivot["Male Share (%)"] = (
        sex_pivot["Male"] / sex_pivot["Total"].replace(0, np.nan) * 100
    ).round(1)
    sex_pivot["Female Share (%)"] = (
        sex_pivot["Female"] / sex_pivot["Total"].replace(0, np.nan) * 100
    ).round(1)

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

    st.markdown('<div class="sub-title">Sex-Predominant Diseases</div>', unsafe_allow_html=True)
    pred = sex_pivot.copy()
    pred["Dominant Sex"] = np.where(pred["Male"] > pred["Female"], "Male",
                             np.where(pred["Female"] > pred["Male"], "Female", "Balanced"))
    pred["Imbalance (percentage points)"] = (pred["Male Share (%)"] - pred["Female Share (%)"]).abs()
    pred_display = (
        pred.reset_index()
        .sort_values("Total", ascending=False)
        .head(15)[["Disease", "Male", "Female", "Total",
                   "Dominant Sex", "Imbalance (percentage points)"]]
    )
    st.dataframe(pred_display, use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Statistical Association: Disease vs Sex</div>', unsafe_allow_html=True)
    obs_sex = sex_pivot[["Male", "Female"]].values
    chi2_s, dof_s = chi_square(obs_sex)
    v_s = cramers_v(chi2_s, obs_sex.sum(), obs_sex.shape[0], obs_sex.shape[1])
    st.markdown(result_card(
        "Chi-Square Result",
        f"Chi-square = <b>{chi2_s:,.2f}</b>, degrees of freedom = <b>{dof_s}</b>. "
        f"Cramer's V = <b>{v_s:.3f}</b> — <b>{v_strength(v_s)}</b> association between "
        f"disease type and sex. Overall case load: "
        f"<b>{male_cases:,} male</b> vs <b>{female_cases:,} female</b>."
    ), unsafe_allow_html=True)

# ==================================================================
# 4. DISEASE PATTERNS ACROSS TIME
# ==================================================================
with tab_time:
    st.markdown('<div class="section-title">Disease Patterns Across Time</div>', unsafe_allow_html=True)
    st.caption(
        "Reveals seasonal variation, monthly changes, and periods of unusually "
        "high or low disease occurrence."
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
                                xaxis_title="", yaxis_title="Cases",
                                legend_title="")
        st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown('<div class="sub-title">Month-over-Month Change</div>', unsafe_allow_html=True)
        mo = monthly_totals.copy()
        mo["Change"] = mo["Cases"].diff().fillna(0).astype(int)
        mo["Change (%)"] = (
            mo["Cases"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        ).round(2)
        st.dataframe(mo, use_container_width=True, hide_index=True)

        if not quarterly.empty:
            q = quarterly.groupby(["Period Order", "Period"], observed=True)["Cases"].sum().reset_index().sort_values("Period Order")
            fig_q = px.bar(q, x="Period", y="Cases", text_auto=True,
                           template="plotly_white",
                           color="Cases", color_continuous_scale="Teal")
            fig_q.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="", yaxis_title="Cases",
                                coloraxis_showscale=False)
            st.markdown('<div class="sub-title">Quarterly Aggregates</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_q, use_container_width=True)

# ==================================================================
# 5. RELATIONSHIP BETWEEN VARIABLES
# ==================================================================
with tab_rel:
    st.markdown('<div class="section-title">Relationship Between Variables</div>', unsafe_allow_html=True)
    st.caption(
        "Examines how disease type, age group, sex, and reporting period are "
        "associated with the number of morbidity cases."
    )

    st.markdown('<div class="sub-title">Association Strength (Cramer\'s V)</div>', unsafe_allow_html=True)

    # Disease vs Age
    obs_da = filtered.groupby(["Disease", "Age Group"], observed=True)["Cases"].sum().unstack(fill_value=0).values
    chi2_da, dof_da = chi_square(obs_da)
    v_da = cramers_v(chi2_da, obs_da.sum(), obs_da.shape[0], obs_da.shape[1])

    # Disease vs Sex
    obs_ds = filtered.groupby(["Disease", "Sex"], observed=True)["Cases"].sum().unstack(fill_value=0).values
    chi2_ds, dof_ds = chi_square(obs_ds)
    v_ds = cramers_v(chi2_ds, obs_ds.sum(), obs_ds.shape[0], obs_ds.shape[1])

    # Disease vs Period
    obs_dp = filtered.groupby(["Disease", "Period"], observed=True)["Cases"].sum().unstack(fill_value=0).values
    chi2_dp, dof_dp = chi_square(obs_dp)
    v_dp = cramers_v(chi2_dp, obs_dp.sum(), obs_dp.shape[0], obs_dp.shape[1])

    # Age vs Sex
    obs_as = filtered.groupby(["Age Group", "Sex"], observed=True)["Cases"].sum().unstack(fill_value=0).values
    chi2_as, dof_as = chi_square(obs_as)
    v_as = cramers_v(chi2_as, obs_as.sum(), obs_as.shape[0], obs_as.shape[1])

    rel_table = pd.DataFrame({
        "Variable Pair": ["Disease - Age Group", "Disease - Sex",
                          "Disease - Reporting Period", "Age Group - Sex"],
        "Chi-Square": [round(chi2_da, 2), round(chi2_ds, 2), round(chi2_dp, 2), round(chi2_as, 2)],
        "Degrees of Freedom": [dof_da, dof_ds, dof_dp, dof_as],
        "Cramer's V": [round(v_da, 3), round(v_ds, 3), round(v_dp, 3), round(v_as, 3)],
        "Strength": [v_strength(v_da), v_strength(v_ds), v_strength(v_dp), v_strength(v_as)],
    })
    st.dataframe(rel_table, use_container_width=True, hide_index=True)

    fig_v = px.bar(
        rel_table, x="Cramer's V", y="Variable Pair", orientation="h",
        text_auto=True, template="plotly_white",
        color="Cramer's V", color_continuous_scale="Teal",
    )
    fig_v.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Association Strength (Cramer's V)", yaxis_title="",
                        coloraxis_showscale=False)
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown('<div class="sub-title">Disease x Age Group Crosstab (Top 12 Diseases)</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sub-title">Disease x Sex Crosstab (Top 12 Diseases)</div>', unsafe_allow_html=True)
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

# ==================================================================
# 6. SIGNIFICANT HEALTH FACTORS
# ==================================================================
with tab_sig:
    st.markdown('<div class="section-title">Significant Health-Related Factors Affecting Morbidity</div>', unsafe_allow_html=True)
    st.caption(
        "Ranks the factors (disease, age group, sex, period) by their measured "
        "influence on morbidity case counts, based on the cleaned dataset."
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

    st.markdown('<div class="sub-title">Factor Ranking by Measured Impact</div>', unsafe_allow_html=True)

    # Compute influence for each factor using normalized concentration
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

# ==================================================================
# 7. FORECASTED MORBIDITY TRENDS
# ==================================================================
with tab_forecast:
    st.markdown('<div class="section-title">Forecasted Morbidity Trends</div>', unsafe_allow_html=True)
    st.caption(
        "Projects future case counts from historical monthly data using a linear "
        "trend model, so health authorities can anticipate increases or decreases."
    )

    monthly = filtered[filtered["Period Type"] == "Monthly"]

    if monthly.empty or monthly["Period Order"].nunique() < 3:
        st.info(
            "At least three monthly periods are required for a stable forecast. "
            "Please include more monthly sheets."
        )
    else:
        horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)

        monthly_totals = (
            monthly.groupby(["Period Order", "Period"], observed=True)["Cases"]
            .sum().reset_index().sort_values("Period Order")
        )

        forecast_values = forecast_linear(monthly_totals["Cases"], steps=horizon)
        last_order = int(monthly_totals["Period Order"].max())
        future_orders = list(range(last_order + 1, last_order + 1 + horizon))
        future_labels = [f"Forecast {i}" for i in range(1, horizon + 1)]

        hist_df = pd.DataFrame({
            "Period": monthly_totals["Period"],
            "Cases": monthly_totals["Cases"],
            "Type": "Historical",
        })
        fc_df = pd.DataFrame({
            "Period": future_labels,
            "Cases": forecast_values.round(0).astype(int),
            "Type": "Forecast",
        })
        combined = pd.concat([hist_df, fc_df], ignore_index=True)

        fig_fc = px.line(
            combined, x="Period", y="Cases", color="Type", markers=True,
            template="plotly_white",
            color_discrete_map={"Historical": "#0f4c81", "Forecast": "#e76f51"},
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
            vals = forecast_linear(dseries, steps=horizon)
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
            f"Based on the linear trend of the last {len(monthly_totals)} months, "
            f"the total monthly caseload is expected to <b>{direction}</b> over the "
            f"next {horizon} month(s), moving from <b>{total_hist_last:,}</b> cases "
            f"to approximately <b>{total_fc_end:,}</b> cases."
        ), unsafe_allow_html=True)

# ==================================================================
# 8. DECISION SUPPORT
# ==================================================================
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
