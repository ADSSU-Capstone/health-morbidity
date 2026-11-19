from __future__ import annotations

from io import BytesIO
from typing import Optional
import re

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

PALETTE = {
    "purple": "#6A1B9A",
    "purple_dark": "#4A148C",
    "purple_light": "#9C4DCC",
    "yellow": "#F5C518",
    "yellow_dark": "#C9A000",
    "yellow_light": "#FFF2B3",
    "bg": "#FAF7FF",
    "card": "#FFFFFF",
    "text": "#2A1A3A",
    "muted": "#6B5B7B",
}

STYLES = f"""
<style>
    html, body, [class*="css"] {{ font-family: "Segoe UI", Roboto, Arial, sans-serif; }}
    .stApp {{ background: linear-gradient(180deg, #FAF7FF 0%, #FFF9E0 100%); }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
    .hero {{
        background: linear-gradient(135deg, {PALETTE['purple']} 0%, {PALETTE['yellow']} 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(106,27,154,0.25);
    }}
    .hero h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: 0.2px; }}
    .hero p {{ margin: 0.35rem 0 0 0; opacity: 0.95; font-size: 0.95rem; }}
    .kpi-card {{
        background: {PALETTE['card']}; border-radius: 16px; padding: 1rem 1.1rem;
        box-shadow: 0 4px 18px rgba(74,20,140,0.10);
        border: 1px solid rgba(106,27,154,0.10); min-height: 105px;
        border-top: 4px solid {PALETTE['yellow']};
    }}
    .kpi-label {{ color:{PALETTE['muted']}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem; }}
    .kpi-value {{ color:{PALETTE['purple']}; font-size:1.5rem; font-weight:700; line-height:1.15; word-break:break-word; }}
    .kpi-sub {{ color:{PALETTE['muted']}; font-size:0.8rem; margin-top:0.3rem; }}
    .result-card {{
        background:{PALETTE['card']}; border-left:5px solid {PALETTE['yellow']};
        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem;
        box-shadow:0 3px 14px rgba(74,20,140,0.08);
    }}
    .result-card h4 {{ margin:0 0 0.35rem 0; color:{PALETTE['purple']}; font-size:1rem; }}
    .result-card p {{ margin:0; color:{PALETTE['text']}; font-size:0.95rem; line-height:1.5; }}
    .section-title {{ color:{PALETTE['purple']}; font-size:1.15rem; font-weight:700; margin:0.9rem 0 0.5rem 0; }}
    .sub-title {{ color:{PALETTE['purple']}; font-size:0.98rem; font-weight:600; margin:0.6rem 0 0.35rem 0; }}
    .small-note {{ color:{PALETTE['muted']}; font-size:0.82rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab"] {{
        background:{PALETTE['card']}; border-radius:12px 12px 0 0; padding:0.5rem 0.9rem;
        border:1px solid #EADDF5; font-size:0.88rem;
    }}
    .stTabs [aria-selected="true"] {{
        background:{PALETTE['purple']} !important; color:white !important;
        border-color:{PALETTE['purple']} !important;
    }}
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
    if n == 0: return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if np.any(nonzero) else 0.0
    return {"MAE": round(mae, 3), "MSE": round(mse, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 3)}


def descriptive_trend_forecast(series: pd.Series, steps: int = 3) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    if len(y) == 0: return np.zeros(steps)
    if len(y) == 1: return np.full(steps, y[0])
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + steps)
    return np.maximum(np.polyval(coeffs, future_x), 0)


def fit_arima_forecast(series: pd.Series, steps: int = 3) -> Optional[np.ndarray]:
    if not STATSMODELS_AVAILABLE: return None
    try:
        model = ARIMA(series.astype(float), order=(1, 1, 1))
        fit = model.fit()
        fc = fit.forecast(steps=steps)
        return np.maximum(np.asarray(fc, dtype=float), 0)
    except Exception:
        return None


def fit_prophet_forecast(series: pd.Series, steps: int = 3) -> Optional[np.ndarray]:
    if not PROPHET_AVAILABLE: return None
    try:
        df = pd.DataFrame({
            "ds": pd.date_range(start="2024-01-01", periods=len(series), freq="MS"),
            "y": series.astype(float).values,
        })
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=steps, freq="MS")
        fc = m.predict(future)
        return np.maximum(np.asarray(fc["yhat"].tail(steps).values, dtype=float), 0)
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
    if df.empty or "Disease" not in df.columns: return df
    disease = df["Disease"].astype(str).str.strip()
    keep = ~disease.isin(PLACEHOLDER_DISEASES)
    keep &= ~disease.str.upper().isin(PLACEHOLDER_LETTERS)
    keep &= disease.str.len() > 1
    return df[keep].copy()


@st.cache_data(show_spinner=False)
def load_clean_data(file_bytes_list: tuple, filenames: tuple):
    frames = []
    for fb, name in zip(file_bytes_list, filenames):
        res = build_dataset(BytesIO(fb))
        df = res.long_df if hasattr(res, "long_df") else res
        df = clean_placeholder_rows(df)
        if df is not None and not df.empty:
            df = df.copy()
            df["Source File"] = name
            year_match = re.search(r'20\d{2}', name)
            df["Year"] = year_match.group(0) if year_match else None
            frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


st.markdown(
    f"""
    <div class="hero">
        <h1>FHSIS Morbidity Dashboard</h1>
        <p>RHU Trento | Agusan del Sur | Insights and Relationships from Morbidity Surveillance Data</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Data Source")
    uploaded_files = st.file_uploader(
        "Upload one or more FHSIS workbooks",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Select one or more FHSIS workbooks. Each file is processed and the records are combined.",
    )
    st.caption("The dashboard does not auto-load a file. Please upload at least one FHSIS workbook to begin.")

if not uploaded_files:
    st.info("Upload at least one FHSIS Excel workbook in the sidebar to load the dashboard. No data will be loaded until a file is provided.")
    st.stop()

file_bytes_list = tuple(f.getvalue() for f in uploaded_files)
filenames = tuple(f.name for f in uploaded_files)

with st.spinner("Processing FHSIS data..."):
    long_df = load_clean_data(file_bytes_list, filenames)

if long_df is None or long_df.empty:
    st.error("No morbidity records were detected in the uploaded workbook(s). Please check the workbook format.")
    st.stop()


with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")
    source_files = sorted(long_df["Source File"].unique().tolist()) if "Source File" in long_df.columns else []
    if len(source_files) > 1:
        selected_sources = st.multiselect("Source file", source_files, default=source_files, help="Include records from one or more uploaded workbooks.")
        if selected_sources: long_df = long_df[long_df["Source File"].isin(selected_sources)]

    period_options = sorted(long_df["Period"].dropna().unique().tolist())
    periods = st.multiselect("Period", period_options, default=period_options)
    disease_options = sorted(long_df["Disease"].dropna().unique().tolist())
    diseases = st.multiselect("Disease", disease_options, default=[], help="Leave blank to include all diseases.")
    age_options = [a for a in AGE_ORDER if a in long_df["Age Group"].astype(str).unique().tolist()]
    age_groups = st.multiselect("Age group", age_options, default=age_options)
    sexes = st.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])

filtered = long_df.copy()
if periods: filtered = filtered[filtered["Period"].isin(periods)]
if diseases: filtered = filtered[filtered["Disease"].isin(diseases)]
if age_groups: filtered = filtered[filtered["Age Group"].astype(str).isin(age_groups)]
if sexes: filtered = filtered[filtered["Sex"].isin(sexes)]

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

filtered = filtered.dropna(subset=["Year"])

total_cases = int(filtered["Cases"].sum())
unique_diseases = int(filtered["Disease"].nunique())
male_cases = int(filtered.loc[filtered["Sex"] == "Male", "Cases"].sum())
female_cases = int(filtered.loc[filtered["Sex"] == "Female", "Cases"].sum())

disease_series = filtered.groupby("Disease")["Cases"].sum().sort_values(ascending=False)
top_disease = disease_series.index[0] if not disease_series.empty else "N/A"
top_disease_cases = int(disease_series.iloc[0]) if not disease_series.empty else 0
top_disease_share = round(top_disease_cases / total_cases * 100, 1) if total_cases else 0.0

age_series = filtered.groupby("Age Group", observed=True)["Cases"].sum().reindex(AGE_ORDER).dropna()
top_age = age_series.idxmax() if not age_series.empty else "N/A"
top_age_cases = int(age_series.max()) if not age_series.empty else 0
top_age_share = round(top_age_cases / total_cases * 100, 1) if total_cases else 0.0

period_series = filtered.groupby("Period", observed=True)["Cases"].sum()
peak_period = period_series.idxmax() if not period_series.empty else "N/A"
peak_period_cases = int(period_series.max()) if not period_series.empty else 0
top10_share = round(disease_series.head(10).sum() / total_cases * 100, 1) if total_cases else 0.0


(
    tab_overview,
    tab_age,
    tab_sex,
    tab_time,
    tab_rel,
    tab_forecast,
    tab_dss,
) = st.tabs([
    "Overview",
    "Disease Patterns by Age Group",
    "Disease Patterns by Sex",
    "Disease Patterns Across Time",
    "Relationship Between Variables",
    "Forecasted Trends",
    "Decision Support",
])


with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(kpi_card("Total Cases", f"{total_cases:,}", "Filtered records"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Unique Diseases", f"{unique_diseases:,}", "Distinct diagnoses"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Top Disease", top_disease, f"{top_disease_cases:,} cases"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Top Age Group", top_age, f"{top_age_cases:,} cases"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Peak Period", str(peak_period), f"{peak_period_cases:,} cases"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 10 Diseases by Caseload (By Year)</div>', unsafe_allow_html=True)
    top10_year = filtered[filtered["Disease"].isin(disease_series.head(10).index)].groupby(["Disease", "Year"], observed=True)["Cases"].sum().reset_index()
    top10_year = top10_year.sort_values(by=["Year", "Cases"], ascending=[True, False])
    
    fig_top10 = px.bar(
        top10_year, x="Cases", y="Disease", color="Year", orientation="h", 
        barmode="group", text_auto=True, template="plotly_white",
        color_discrete_sequence=[PALETTE["purple"], PALETTE["yellow"], PALETTE["purple_light"]]
    )
    fig_top10.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Cases", yaxis_title="", legend_title="Year")
    st.plotly_chart(fig_top10, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Sex Distribution (By Year)</div>', unsafe_allow_html=True)
        sex_df = filtered.groupby(["Sex", "Year"], observed=True)["Cases"].sum().reset_index()
        fig_sex = px.bar(
            sex_df, x="Sex", y="Cases", color="Year", barmode="group", text_auto=True, 
            template="plotly_white", color_discrete_sequence=[PALETTE["purple"], PALETTE["yellow"], PALETTE["purple_light"]]
        )
        fig_sex.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Cases", legend_title="Year")
        st.plotly_chart(fig_sex, use_container_width=True)
        
    with col2:
        st.markdown('<div class="section-title">Cases by Age Group (By Year)</div>', unsafe_allow_html=True)
        age_year_df = filtered.groupby(["Age Group", "Year"], observed=True)["Cases"].sum().reset_index()
        age_year_df = age_year_df.sort_values(by=["Year", "Age Group"])
        
        fig_age = px.bar(
            age_year_df, x="Age Group", y="Cases", color="Year", barmode="group",
            text_auto=True, template="plotly_white",
            color_discrete_sequence=[PALETTE["purple"], PALETTE["yellow"], PALETTE["purple_light"]]
        )
        fig_age.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="", yaxis_title="Cases", legend_title="Year"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown('<div class="section-title">Total Cases by Year (2024-2026)</div>', unsafe_allow_html=True)
    year_df = filtered.dropna(subset=["Year"])
    if not year_df.empty:
        by_year = year_df.groupby("Year", observed=True)["Cases"].sum().reset_index().sort_values("Year")
        fig_period = px.bar(
            by_year, x="Year", y="Cases", text_auto=True, template="plotly_white", color="Cases",
            color_continuous_scale=[[0, PALETTE["yellow_light"]], [0.5, PALETTE["yellow"]], [1, PALETTE["purple"]]]
        )
        fig_period.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_period.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Year", yaxis_title="Cases", coloraxis_showscale=False)
        fig_period.update_xaxes(type='category')
        st.plotly_chart(fig_period, use_container_width=True)


with tab_age:
    st.markdown('<div class="section-title">Disease Patterns by Age Group (By Year)</div>', unsafe_allow_html=True)
    st.caption("Descriptive analytics: frequencies, counts, and proportional morbidity broken down by year.")

    st.markdown('<div class="sub-title">Heatmap: Age Group vs Disease (Faceted by Year)</div>', unsafe_allow_html=True)
    top_diseases_list = disease_series.head(10).index.tolist()
    
    # FIX: Group by Year, Disease, and Age Group directly without complex pivoting
    matrix_data = filtered[filtered["Disease"].isin(top_diseases_list)].groupby(["Year", "Disease", "Age Group"], observed=True)["Cases"].sum().reset_index()
    
    if not matrix_data.empty:
        fig_heat = px.density_heatmap(
            matrix_data, x="Age Group", y="Disease", z="Cases", facet_col="Year", 
            facet_col_wrap=1, text_auto=True, template="plotly_white",
            color_continuous_scale=[[0, PALETTE["yellow_light"]], [0.5, PALETTE["yellow"]], [1, PALETTE["purple"]]],
            category_orders={"Age Group": AGE_ORDER}
        )
        fig_heat.update_layout(height=1000, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Age Group", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="sub-title">Age Concentration of Top Diseases (By Year)</div>', unsafe_allow_html=True)
    concentration = []
    for disease in disease_series.head(10).index:
        for year in ["2024", "2025", "2026"]:
            sub = filtered[(filtered["Disease"] == disease) & (filtered["Year"] == year)]
            a = sub.groupby("Age Group", observed=True)["Cases"].sum()
            if a.sum() > 0:
                concentration.append({
                    "Disease": disease, "Year": year, "Most Frequent Age Group": a.idxmax(), 
                    "Cases in That Group": int(a.max()), "Total Cases": int(a.sum()), 
                    "Proportional Share (%)": round(a.max() / a.sum() * 100, 1),
                })
    st.dataframe(pd.DataFrame(concentration), use_container_width=True, hide_index=True)


with tab_sex:
    st.markdown('<div class="section-title">Disease Patterns by Sex (By Year)</div>', unsafe_allow_html=True)
    st.caption("Descriptive analytics: proportional morbidity by disease and sex, broken down by year.")

    sex_year_df = filtered.groupby(["Disease", "Sex", "Year"], observed=True)["Cases"].sum().reset_index()
    top_sex_diseases = disease_series.head(8).index.tolist()
    sex_year_df = sex_year_df[sex_year_df["Disease"].isin(top_sex_diseases)]

    fig_sex_bar = px.bar(
        sex_year_df, x="Disease", y="Cases", color="Year", facet_col="Sex", barmode="group",
        text_auto=True, template="plotly_white",
        color_discrete_sequence=[PALETTE["purple"], PALETTE["yellow"], PALETTE["purple_light"]]
    )
    fig_sex_bar.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10), 
        xaxis_title="", yaxis_title="Cases", xaxis_tickangle=-40, legend_title="Year"
    )
    st.plotly_chart(fig_sex_bar, use_container_width=True)


with tab_time:
    st.markdown('<div class="section-title">Disease Patterns Across Time</div>', unsafe_allow_html=True)
    st.caption("Time Series Analysis: trend and year-over-year morbidity cases for the top diseases.")

    year_filtered = filtered.dropna(subset=["Year"]).copy()
    
    if year_filtered.empty:
        st.info("No year data is available. Please ensure your file names contain a 4-digit year (e.g., 2024).")
    else:
        st.markdown('<div class="sub-title">Total Cases by Year</div>', unsafe_allow_html=True)
        yearly_totals = year_filtered.groupby("Year", observed=True)["Cases"].sum().reset_index().sort_values("Year")
        
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            fig_year_line = px.line(yearly_totals, x="Year", y="Cases", markers=True, template="plotly_white", color_discrete_sequence=[PALETTE["purple"]])
            fig_year_line.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Year", yaxis_title="Total Cases")
            fig_year_line.update_xaxes(type='category') 
            st.plotly_chart(fig_year_line, use_container_width=True)
        with col_y2:
            yoy = yearly_totals.copy()
            yoy["Change"] = yoy["Cases"].diff().fillna(0).astype(int)
            yoy["Change (%)"] = (yoy["Cases"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100).round(2)
            st.dataframe(yoy, use_container_width=True, hide_index=True)

        st.markdown('<div class="sub-title">Monthly Trends of Top Diseases (Broken Down by Year)</div>', unsafe_allow_html=True)
        monthly = filtered[filtered["Period Type"] == "Monthly"]
        
        if monthly.empty:
            st.info("Monthly data is not available under the current filters.")
        else:
            top8 = disease_series.head(8).index.tolist()
            trend_data = monthly[monthly["Disease"].isin(top8)].groupby(["Year", "Period Order", "Period", "Disease"], observed=True)["Cases"].sum().reset_index().sort_values("Period Order")
            
            colors = [PALETTE["purple"], PALETTE["yellow"], PALETTE["purple_light"], PALETTE["yellow_dark"], "#3E0A5C", "#FFE066", "#B39DDB", "#8D6E63"]
            
            tab_24, tab_25, tab_26 = st.tabs(["2024 Monthly Trends", "2025 Monthly Trends", "2026 Monthly Trends"])
            
            with tab_24:
                df24 = trend_data[trend_data["Year"] == "2024"]
                if df24.empty: st.info("No monthly data for 2024.")
                else:
                    fig24 = px.line(df24, x="Period", y="Cases", color="Disease", markers=True, template="plotly_white", color_discrete_sequence=colors)
                    fig24.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month", yaxis_title="Cases", legend_title="")
                    st.plotly_chart(fig24, use_container_width=True)

            with tab_25:
                df25 = trend_data[trend_data["Year"] == "2025"]
                if df25.empty: st.info("No monthly data for 2025.")
                else:
                    fig25 = px.line(df25, x="Period", y="Cases", color="Disease", markers=True, template="plotly_white", color_discrete_sequence=colors)
                    fig25.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month", yaxis_title="Cases", legend_title="")
                    st.plotly_chart(fig25, use_container_width=True)

            with tab_26:
                df26 = trend_data[trend_data["Year"] == "2026"]
                if df26.empty: st.info("No monthly data for 2026.")
                else:
                    fig26 = px.line(df26, x="Period", y="Cases", color="Disease", markers=True, template="plotly_white", color_discrete_sequence=colors)
                    fig26.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month", yaxis_title="Cases", legend_title="")
                    st.plotly_chart(fig26, use_container_width=True)

        st.markdown('<div class="sub-title">Overall Yearly Trends of Top Diseases</div>', unsafe_allow_html=True)
        trend_data_yearly = year_filtered[year_filtered["Disease"].isin(top8)].groupby(["Year", "Disease"], observed=True)["Cases"].sum().reset_index().sort_values("Year")
        
        fig_multi = px.line(
            trend_data_yearly, x="Year", y="Cases", color="Disease", markers=True, template="plotly_white",
            color_discrete_sequence=colors
        )
        fig_multi.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Year", yaxis_title="Cases", legend_title="")
        fig_multi.update_xaxes(type='category')
        st.plotly_chart(fig_multi, use_container_width=True)


with tab_rel:
    st.markdown('<div class="section-title">Relationship Between Variables (By Year)</div>', unsafe_allow_html=True)
    st.caption("Descriptive analysis of how disease type, age group, sex, and reporting period co-occur, broken down by year.")

    top12 = disease_series.head(12).index.tolist()

    st.markdown('<div class="sub-title">Disease x Age Group (Faceted by Year)</div>', unsafe_allow_html=True)
    ct_da = (filtered[filtered["Disease"].isin(top12)].groupby(["Disease", "Age Group", "Year"], observed=True)["Cases"].sum().reset_index())
    
    fig_ct = px.density_heatmap(
        ct_da, x="Age Group", y="Disease", z="Cases", facet_col="Year", 
        facet_col_wrap=1, text_auto=True, template="plotly_white",
        color_continuous_scale=[[0, PALETTE["yellow_light"]], [0.5, PALETTE["yellow"]], [1, PALETTE["purple"]]],
        category_orders={"Age Group": AGE_ORDER}
    )
    fig_ct.update_layout(height=1000, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Age Group", yaxis_title="")
    st.plotly_chart(fig_ct, use_container_width=True)

    st.markdown('<div class="sub-title">Disease x Sex (Faceted by Year)</div>', unsafe_allow_html=True)
    ct_ds = (filtered[filtered["Disease"].isin(top12)].groupby(["Disease", "Sex", "Year"], observed=True)["Cases"].sum().reset_index())
    
    fig_ct_ds = px.density_heatmap(
        ct_ds, x="Sex", y="Disease", z="Cases", facet_col="Year", 
        facet_col_wrap=1, text_auto=True, template="plotly_white",
        color_continuous_scale=[[0, PALETTE["yellow_light"]], [0.5, PALETTE["yellow"]], [1, PALETTE["purple"]]]
    )
    fig_ct_ds.update_layout(height=1000, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Sex", yaxis_title="")
    st.plotly_chart(fig_ct_ds, use_container_width=True)


with tab_forecast:
    st.markdown('<div class="section-title">Forecasted Morbidity Trends</div>', unsafe_allow_html=True)
    st.caption("Projects future case counts from historical monthly data using predictive models.")

    monthly = filtered[filtered["Period Type"] == "Monthly"]
    if monthly.empty or monthly["Period Order"].nunique() < 3:
        st.info("At least three monthly periods are required for a stable forecast.")
    else:
        monthly_totals = (monthly.groupby(["Period Order", "Period"], observed=True)["Cases"].sum().reset_index().sort_values("Period Order"))
        series = monthly_totals["Cases"].reset_index(drop=True)

        model_choice = st.selectbox("Predictive model", options=["ARIMA", "Prophet", "Descriptive trend (fallback)"], index=0)

        if model_choice == "ARIMA" and not STATSMODELS_AVAILABLE:
            st.warning("statsmodels is not installed; falling back to the descriptive trend.")
            model_choice = "Descriptive trend (fallback)"
        if model_choice == "Prophet" and not PROPHET_AVAILABLE:
            st.warning("prophet is not installed; falling back to the descriptive trend.")
            model_choice = "Descriptive trend (fallback)"

        if len(series) >= 6:
            train = series.iloc[:-2]
            test = series.iloc[-2:].reset_index(drop=True)
            if model_choice == "ARIMA": pred_test = fit_arima_forecast(train, steps=len(test))
            elif model_choice == "Prophet": pred_test = fit_prophet_forecast(train, steps=len(test))
            else: pred_test = descriptive_trend_forecast(train, steps=len(test))
            if pred_test is None: pred_test = descriptive_trend_forecast(train, steps=len(test))
            metrics = evaluate_forecast(test.values, pred_test)

            st.markdown('<div class="sub-title">Model Evaluation (hold-out tail)</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(kpi_card("MAE", f"{metrics['MAE']:,}", "Mean Absolute Error"), unsafe_allow_html=True)
            with m2: st.markdown(kpi_card("MSE", f"{metrics['MSE']:,}", "Mean Squared Error"), unsafe_allow_html=True)
            with m3: st.markdown(kpi_card("RMSE", f"{metrics['RMSE']:,}", "Root Mean Squared Error"), unsafe_allow_html=True)
            with m4: st.markdown(kpi_card("MAPE", f"{metrics['MAPE']:,}%", "Mean Absolute Percentage Error"), unsafe_allow_html=True)

        horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)

        if model_choice == "ARIMA": forecast_values = fit_arima_forecast(series, steps=horizon)
        elif model_choice == "Prophet": forecast_values = fit_prophet_forecast(series, steps=horizon)
        else: forecast_values = None
        if forecast_values is None:
            forecast_values = descriptive_trend_forecast(series, steps=horizon)
            if model_choice != "Descriptive trend (fallback)":
                st.info("The selected model could not be fitted; showing the descriptive trend.")

        future_labels = [f"Forecast {i}" for i in range(1, horizon + 1)]
        hist_df = pd.DataFrame({"Period": monthly_totals["Period"], "Cases": monthly_totals["Cases"], "Type": "Historical"})
        fc_df = pd.DataFrame({"Period": future_labels, "Cases": forecast_values.round(0).astype(int), "Type": "Forecast"})
        combined = pd.concat([hist_df, fc_df], ignore_index=True)

        fig_fc = px.line(combined, x="Period", y="Cases", color="Type", markers=True, template="plotly_white", color_discrete_map={"Historical": PALETTE["purple"], "Forecast": PALETTE["yellow"]})
        fig_fc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Cases", legend_title="")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown('<div class="sub-title">Forecasted Monthly Totals</div>', unsafe_allow_html=True)
        st.dataframe(fc_df.rename(columns={"Cases": "Forecasted Cases"}), use_container_width=True, hide_index=True)


with tab_dss:
    st.markdown('<div class="section-title">Health Information Dashboard and Decision Support</div>', unsafe_allow_html=True)
    st.caption("Translates the findings into priority lists and resource cues for local health authorities, based purely on the cleaned dataset.")

    st.markdown('<div class="sub-title">Priority Diseases (Top 10 by Caseload)</div>', unsafe_allow_html=True)
    priority_disease = disease_series.head(10).reset_index()
    priority_disease.columns = ["Disease", "Cases"]
    priority_disease["Share (%)"] = (priority_disease["Cases"] / total_cases * 100).round(2)
    priority_disease["Priority Rank"] = range(1, len(priority_disease) + 1)
    st.dataframe(priority_disease[["Priority Rank", "Disease", "Cases", "Share (%)"]], use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Priority Age Groups (By Year)</div>', unsafe_allow_html=True)
    priority_age_year = filtered.groupby(["Age Group", "Year"], observed=True)["Cases"].sum().reset_index()
    priority_age_year["Share (%)"] = (priority_age_year["Cases"] / total_cases * 100).round(2)
    st.dataframe(priority_age_year.sort_values(by=["Year", "Cases"], ascending=[True, False]), use_container_width=True, hide_index=True)

    st.markdown('<div class="sub-title">Resource Allocation Guidance</div>', unsafe_allow_html=True)
    high_load = disease_series.head(5)
    resource_df = pd.DataFrame({
        "Disease": high_load.index,
        "Cases to Plan For": high_load.values,
        "Suggested Focus": ["Allocate medicines, supplies, and consultation time for this condition." for _ in high_load.index],
    })
    st.dataframe(resource_df, use_container_width=True, hide_index=True)

    st.markdown(
        result_card(
            "Evidence-Based Planning Summary",
            f"The cleaned dataset shows <b>{total_cases:,} cases</b> across <b>{unique_diseases:,} diseases</b>. "
            f"The five leading diseases ({', '.join(high_load.index[:3])}, and others) should receive the highest "
            f"share of consultation hours, medicines, and diagnostic supplies. "
            f"The <b>{top_age}</b> age group and the <b>{str(peak_period)}</b> period "
            f"are the highest-demand targets for outreach and staffing."
        ), unsafe_allow_html=True
    )

    st.markdown(
        result_card(
            "Monitoring Cue",
            f"Track the top three diseases monthly. Any month where their combined "
            f"case count exceeds the historical average by more than 20% should trigger "
            f"a review of stock levels, staffing, and community-level prevention "
            f"activities for the affected age group."
        ), unsafe_allow_html=True
    )

    st.markdown(
        result_card(
            "Forecast Use",
            f"Use the Forecasted Trends tab to project caseloads for the next 1 to 6 "
            f"months. Align procurement and staffing cycles with the projected "
            f"direction so the health facility remains within capacity."
        ), unsafe_allow_html=True
    )

    st.download_button("Download decision support summary (CSV)", priority_disease.to_csv(index=False).encode("utf-8"), "decision_support_priority_diseases.csv", "text/csv")
