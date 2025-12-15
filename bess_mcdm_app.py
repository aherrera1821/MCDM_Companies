import streamlit as st
import pandas as pd
import numpy as np

# ------------- App config -------------
st.set_page_config(
    page_title="BESS – AI/Data Center MCDM",
    layout="wide"
)

st.title("MCDM Tool - Portfolio Companies Overview")

st.markdown(
    """
1. Upload the Excel file with company scores.
2. Adjust the weights for each criterion.
3. Review rankings by **company** and aggregated by **sector**.
"""
)

# ------------- File upload -------------
uploaded_file = st.file_uploader(
    "Upload the Excel file (e.g. `bess_mcdm_companies.xlsx`)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("⬆️ Upload the Excel file to begin.")
    st.stop()

# Load data
try:
    df = pd.read_excel(uploaded_file, sheet_name="Companies")
except Exception as e:
    st.error(f"Could not read `Companies` sheet: {e}")
    st.stop()

# Basic validation
required_cols = [
    "Company",
    "Sector",
    "AI_DC_Exposure_Score",
    "BESS_Synergy_Technical",
    "BESS_Synergy_Commercial",
    "Growth_Outlook",
    "Risk_Adjusted_Quality",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in Excel file: {missing}")
    st.stop()

# ------------- Criteria & weights -------------
st.sidebar.header("MCDM Weights")

criteria = {
    "AI_DC_Exposure_Score": "AI / Data Center Exposure",
    "BESS_Synergy_Technical": "Technical BESS Synergy",
    "BESS_Synergy_Commercial": "Commercial BESS Synergy",
    "Growth_Outlook": "Growth Outlook",
    "Risk_Adjusted_Quality": "Risk-Adjusted Quality",
}

weights = {}
for col, label in criteria.items():
    weights[col] = st.sidebar.slider(
        label,
        min_value=0.0,
        max_value=5.0,
        value=3.0,  # default
        step=0.1,
    )

weight_values = np.array(list(weights.values()))
if weight_values.sum() == 0:
    st.warning("All weights are zero. Increase at least one weight to see rankings.")
    st.stop()

# Normalized weights
norm_weights = weight_values / weight_values.sum()

st.sidebar.markdown("**Normalized weights:**")
for (col, label), w in zip(criteria.items(), norm_weights):
    st.sidebar.markdown(f"- {label}: {w:.2f}")

# ------------- Compute scores -------------
score_cols = list(criteria.keys())

# Ensure numeric
for c in score_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

score_matrix = df[score_cols].to_numpy(dtype=float)
# Simple scaling: assume all scores on similar scale (1–5). Normalize to 0–1.
score_matrix_norm = score_matrix / 5.0

overall_scores = (score_matrix_norm * norm_weights).sum(axis=1)
df["Overall_Score"] = overall_scores

# For convenience show as 0–100 index
df["Overall_Index_0_100"] = (df["Overall_Score"] * 100).round(1)

# ------------- Filters -------------
st.sidebar.header("Filters")

sectors = sorted(df["Sector"].dropna().unique())
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector",
    options=sectors,
    default=sectors
)

filtered_df = df[df["Sector"].isin(selected_sectors)].copy()

# ------------- Company rankings -------------
st.subheader("Company Ranking")

ranked = filtered_df.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
ranked["Rank"] = ranked.index + 1

cols_to_show = [
    "Rank",
    "Company",
    "Sector",
    "Overall_Index_0_100",
] + score_cols

st.dataframe(
    ranked[cols_to_show],
    use_container_width=True,
    hide_index=True
)

# ------------- Sector rankings -------------
st.subheader("Sector Ranking (Average Score)")

sector_scores = (
    filtered_df
    .groupby("Sector")["Overall_Score"]
    .mean()
    .sort_values(ascending=False)
)

sector_df = sector_scores.reset_index()
sector_df["Overall_Index_0_100"] = (sector_df["Overall_Score"] * 100).round(1)

st.dataframe(sector_df, use_container_width=True, hide_index=True)

st.bar_chart(
    sector_df.set_index("Sector")["Overall_Index_0_100"],
    use_container_width=True
)

# ------------- Detail view -------------
st.subheader("Compare Selected Companies")

selected_companies = st.multiselect(
    "Pick companies to compare (optional)",
    options=list(ranked["Company"]),
)

if selected_companies:
    comp_df = ranked[ranked["Company"].isin(selected_companies)]

    st.write("### Detailed criteria comparison")
    st.dataframe(
        comp_df[["Company", "Sector", "Overall_Index_0_100"] + score_cols],
        use_container_width=True,
        hide_index=True
    )

# ------------- Raw data -------------
with st.expander("Show raw data from Excel"):
    st.dataframe(df, use_container_width=True)