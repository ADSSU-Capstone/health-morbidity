# app.py
# FHSIS Morbidity Dashboard - RHU Trento
# Requires: fhisis_cleaning.py in the same folder

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

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FHSIS Morbidity Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM UI
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary: #0f4c81;
            --accent: #00a6a6;
            --bg: #f5f7fa;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
        }
        html, body, [class*="css"] {
            font-family: "Segoe UI", Roboto, Arial, sans-serif;
        }
        .stApp {
            background: linear-gradient(180deg, #f5f7fa 0%, #eef2f7 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0f4c81 0%, #00a6a6 100%);
            padding: 1.6rem 2rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(15,76,129,0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            opacity: 0.92;
            font-size: 0.95rem;
        }
        .kpi-card {
            background: var(--card);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 18px rgba(31,41,55,0.08);
            border: 1px solid rgba(15,76,129,0.06);
            min-height: 105px;
        }
        .kpi-label {
            color: #6b7280;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            color: var(--primary);
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .kpi-sub {
            color: #6b7280;
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }
        .section-title {
            color: #0f4c81;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.8rem 0 0.5rem 0;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.82rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 12px 12px 0 0;
            padding: 0.55rem 1rem;
            border: 1px solid #e5e7eb;
        }
        .stTabs [aria-selected="true"] {
            background: #0f4c81 !important;
            color: white !important;
            border-color: #0f4c81 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #0f4c81;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean_data(file_bytes: Optional[bytes], filename: str):
    source = BytesIO(file_bytes) if file_bytes is not None else filename
    return build_dataset(source)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>FHSIS Morbidity Dashboard</h1>
        <p>RHU Trento | Agusan del Sur | Annual, Quarterly and Monthly Morbidity Surveillance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# SIDEBAR - SOURCE
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader(
        "Upload FHSIS workbook",
        type=["xlsx", "xls"],
        help="Upload the TRENTO-M2MORBIDITY workbook. If not uploaded, the app looks for the default filename.",
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

with st.spinner("Cleaning and consolidating FHSIS sheets..."):
    result = load_clean_data(file_bytes, filename)

long_df = result.long_df

if long_df.empty:
    st.error("No morbidity records were detected after cleaning. Please check the workbook format.")
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR - FILTERS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")

    period_types = st.multiselect(
        "Period type",
        sorted(long_df["Period Type"].dropna().unique().tolist()),
        default=sorted(long_df["Period Type"].dropna().unique().tolist()),
    )

    period_options = sorted(long_df["Period"].dropna().unique().tolist())
    periods = st.multiselect(
        "Period",
        period_options,
        default=period_options,
    )

    disease_options = sorted(long_df["Disease"].dropna().unique().tolist())
    diseases = st.multiselect(
        "Disease",
        disease_options,
        default=[],
        help="Leave blank to include all diseases.",
    )

    age_options = [a for a in AGE_ORDER if a in long_df["Age Group"].astype(str).unique().tolist()]
    age_groups = st.multiselect(
        "Age group",
        age_options,
        default=age_options,
    )

    sexes = st.multiselect(
        "Sex",
        ["Male", "Female"],
        default=["Male", "Female"],
    )

    st.markdown("---")
    st.caption(
        "Recommendations are not shown in this dashboard. "
        "Use the manuscript for recommendations."
    )

# ------------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_overview, tab_disease, tab_demo, tab_quality, tab_log, tab_framework, tab_ch3 = st.tabs(
    [
        "Overview",
        "Disease Analysis",
        "Demographics",
        "Data Quality",
        "Cleaning Log",
        "Conceptual Framework",
        "Chapter 3 Expected Outcomes",
    ]
)

# ------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------
with tab_overview:
    total_cases = int(filtered["Cases"].sum())
    unique_diseases = int(filtered["Disease"].nunique())

    top_disease_series = filtered.groupby("Disease")["Cases"].sum()
    top_disease = top_disease_series.idxmax() if not top_disease_series.empty else "N/A"
    top_disease_cases = int(top_disease_series.max()) if not top_disease_series.empty else 0

    top_age_series = filtered.groupby("Age Group", observed=True)["Cases"].sum()
    top_age = top_age_series.idxmax() if not top_age_series.empty else "N/A"

    quality_score = result.quality_score

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Total Cases", f"{total_cases:,}", "Filtered records"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Unique Diseases", f"{unique_diseases:,}", "Distinct diagnoses"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Top Disease", f"{top_disease}", f"{top_disease_cases:,} cases"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Top Age Group", f"{top_age}", "Highest case count"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Data Quality", f"{quality_score}%", "Cleaning score"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cases by Period Type</div>', unsafe_allow_html=True)
    by_period = filtered.groupby("Period Type", observed=True)["Cases"].sum().reset_index()
    fig_period = px.bar(
        by_period,
        x="Period Type",
        y="Cases",
        color="Period Type",
        text_auto=True,
        template="plotly_white",
    )
    fig_period.update_layout(
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="Cases",
    )
    st.plotly_chart(fig_period, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top 10 Diseases</div>', unsafe_allow_html=True)
        top10 = (
            filtered.groupby("Disease")["Cases"]
            .sum()
            .nlargest(10)
            .sort_values()
            .reset_index()
        )
        fig_top10 = px.bar(
            top10,
            x="Cases",
            y="Disease",
            orientation="h",
            text_auto=True,
            template="plotly_white",
        )
        fig_top10.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Cases",
            yaxis_title="",
        )
        st.plotly_chart(fig_top10, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Sex Distribution</div>', unsafe_allow_html=True)
        sex_df = filtered.groupby("Sex", observed=True)["Cases"].sum().reset_index()
        fig_sex = px.pie(
            sex_df,
            names="Sex",
            values="Cases",
            hole=0.45,
            template="plotly_white",
        )
        fig_sex.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sex, use_container_width=True)

    st.markdown('<div class="section-title">Cases by Age Group</div>', unsafe_allow_html=True)
    age_df = (
        filtered.groupby("Age Group", observed=True)["Cases"]
        .sum()
        .reindex(AGE_ORDER)
        .dropna()
        .reset_index()
    )
    fig_age = px.bar(
        age_df,
        x="Age Group",
        y="Cases",
        text_auto=True,
        template="plotly_white",
    )
    fig_age.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="Cases",
    )
    st.plotly_chart(fig_age, use_container_width=True)

# ------------------------------------------------------------------
# DISEASE ANALYSIS
# ------------------------------------------------------------------
with tab_disease:
    st.markdown('<div class="section-title">Disease Trend and Age Distribution</div>', unsafe_allow_html=True)

    disease_list = sorted(filtered["Disease"].dropna().unique().tolist())

    if not disease_list:
        st.info("No disease data under the current filters.")
    else:
        selected_disease = st.selectbox("Select disease", disease_list)
        ddf = filtered[filtered["Disease"] == selected_disease]

        trend = (
            ddf.groupby(["Period Type", "Period Order", "Period"], observed=True)["Cases"]
            .sum()
            .reset_index()
            .sort_values(["Period Type", "Period Order"])
        )
        fig_trend = px.line(
            trend,
            x="Period",
            y="Cases",
            color="Period Type",
            markers=True,
            template="plotly_white",
        )
        fig_trend.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="",
            yaxis_title="Cases",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        age_sex = (
            ddf.groupby(["Age Group", "Sex"], observed=True)["Cases"]
            .sum()
            .reset_index()
        )
        fig_age_sex = px.bar(
            age_sex,
            x="Age Group",
            y="Cases",
            color="Sex",
            barmode="group",
            text_auto=True,
            template="plotly_white",
        )
        fig_age_sex.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="",
            yaxis_title="Cases",
        )
        st.plotly_chart(fig_age_sex, use_container_width=True)

        st.markdown("#### Detailed Records")
        st.dataframe(
            ddf[["Period", "Age Group", "Sex", "Cases"]]
            .sort_values(["Period", "Age Group", "Sex"]),
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------------
# DEMOGRAPHICS
# ------------------------------------------------------------------
with tab_demo:
    st.markdown('<div class="section-title">Demographic Breakdown</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_age = st.selectbox(
            "Age group",
            [a for a in AGE_ORDER if a in filtered["Age Group"].astype(str).unique().tolist()],
        )
        selected_sex = st.selectbox("Sex", ["All", "Male", "Female"])

    with col2:
        ddf = filtered[filtered["Age Group"].astype(str) == selected_age]
        if selected_sex != "All":
            ddf = ddf[ddf["Sex"] == selected_sex]

        top = (
            ddf.groupby("Disease")["Cases"]
            .sum()
            .nlargest(15)
            .sort_values()
            .reset_index()
        )
        fig_demo = px.bar(
            top,
            x="Cases",
            y="Disease",
            orientation="h",
            text_auto=True,
            template="plotly_white",
        )
        fig_demo.update_layout(
            height=540,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Cases",
            yaxis_title="",
        )
        st.plotly_chart(fig_demo, use_container_width=True)

# ------------------------------------------------------------------
# DATA QUALITY
# ------------------------------------------------------------------
with tab_quality:
    st.markdown('<div class="section-title">Data Quality Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Data Quality Score", f"{result.quality_score}%")
    c2.metric("Total Cases", f"{int(filtered['Cases'].sum()):,}")
    c3.metric("Cleaning Issues", f"{len(result.quality_log):,}")

    st.markdown("#### Sheet Summary")
    st.dataframe(result.sheet_summary, use_container_width=True, hide_index=True)

    if not result.quality_log.empty:
        sev = (
            result.quality_log.groupby("Severity")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        fig_sev = px.bar(
            sev,
            x="Severity",
            y="Count",
            color="Severity",
            text_auto=True,
            template="plotly_white",
        )
        fig_sev.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Issues",
        )
        st.plotly_chart(fig_sev, use_container_width=True)

        itype = (
            result.quality_log.groupby("Issue Type")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        st.markdown("#### Issue Types")
        st.dataframe(itype, use_container_width=True, hide_index=True)

    st.download_button(
        "Download cleaned long-format CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "fhisis_cleaned_long.csv",
        "text/csv",
    )

# ------------------------------------------------------------------
# CLEANING LOG
# ------------------------------------------------------------------
with tab_log:
    st.markdown('<div class="section-title">Cleaning Log</div>', unsafe_allow_html=True)

    qlog = result.quality_log.copy()

    if qlog.empty:
        st.success("No cleaning issues were logged.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            sev_filter = st.multiselect(
                "Severity",
                sorted(qlog["Severity"].dropna().unique().tolist()),
                default=sorted(qlog["Severity"].dropna().unique().tolist()),
            )
        with col2:
            type_filter = st.multiselect(
                "Issue type",
                sorted(qlog["Issue Type"].dropna().unique().tolist()),
                default=sorted(qlog["Issue Type"].dropna().unique().tolist()),
            )
        with col3:
            sheet_filter = st.multiselect(
                "Sheet",
                sorted(qlog["Sheet"].dropna().unique().tolist()),
                default=sorted(qlog["Sheet"].dropna().unique().tolist()),
            )

        view = qlog[
            qlog["Severity"].isin(sev_filter)
            & qlog["Issue Type"].isin(type_filter)
            & qlog["Sheet"].isin(sheet_filter)
        ]
        st.dataframe(view, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# CONCEPTUAL FRAMEWORK
# ------------------------------------------------------------------
with tab_framework:
    st.markdown('<div class="section-title">Conceptual Framework - Written Output</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **Input**
        - FHSIS morbidity reports from RHU Trento
        - Monthly, quarterly, and annual disease surveillance data
        - Age group and sex disaggregation

        **Process**
        - Data cleaning and validation
        - Consolidation of monthly and quarterly sheets
        - Disease ranking and demographic analysis
        - Data quality assessment

        **Output**
        - Cleaned morbidity dataset
        - Interactive dashboard
        - Disease trend and demographic reports

        **Outcome**
        - Improved visibility of morbidity patterns
        - Evidence-based health program planning
        - Faster identification of priority diseases

        **Impact**
        - Strengthened local health surveillance
        - Better allocation of health resources
        - Improved community health outcomes
        """
    )
    st.info(
        "Replace this placeholder with your actual conceptual framework narrative "
        "from the manuscript."
    )

# ------------------------------------------------------------------
# CHAPTER 3 EXPECTED OUTCOMES
# ------------------------------------------------------------------
with tab_ch3:
    st.markdown('<div class="section-title">Chapter 3 - Expected Outcomes</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **Expected Outcome 1**
        A cleaned and validated FHSIS morbidity dataset for RHU Trento.

        **Expected Outcome 2**
        An interactive dashboard that visualizes disease trends by period, age group, and sex.

        **Expected Outcome 3**
        Identification of the leading causes of morbidity in the catchment population.

        **Expected Outcome 4**
        A data quality report documenting cleaning issues, missing values, and inconsistencies.

        **Expected Outcome 5**
        Evidence to support local health planning, resource allocation, and disease prevention programs.
        """
    )
    st.info(
        "Replace this placeholder with the exact expected outcomes written in your Chapter 3."
    )