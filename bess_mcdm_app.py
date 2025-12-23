import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Companies MCDM for BESS", layout="wide")
st.title("BESS Partner & Investment Attractiveness – MCDM Tool")

st.markdown(
    """
Upload one Excel file that contains a **Companies** sheet.
"""
)

# ---------------- Helpers ----------------
OLD_SCHEMA = {
    "name": "Original schema (AI/DC exposure)",
    "criteria": [
        "AI_DC_Exposure_Score",
        "BESS_Synergy_Technical",
        "BESS_Synergy_Commercial",
        "Growth_Outlook",
        "Risk_Adjusted_Quality",
    ],
}

NEW_SCHEMA = {
    "name": "New schema (Sector Growth + BESS synergy)",
    "criteria": [
        "Sector_Growth",
        "BESS_Technical_Synergy",
        "BESS_Commercial_Synergy",
        "Company_Growth",
        "Risk_Adjusted_Score",
    ],
}

def load_companies_sheet(xlsx_file) -> pd.DataFrame:
    df = pd.read_excel(xlsx_file, sheet_name="Companies")
    # normalize common fields
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    if "Company" not in df.columns:
        raise ValueError("Missing required column: Company")
    return df

def detect_available_criteria(df: pd.DataFrame) -> list[str]:
    # numeric-like columns excluding obvious non-criteria fields
    exclude = {
        "Company","Sector","Subsector","PublicPrivate","HQRegion","Scale","Notes",
        "Initial_Overall_Score","Overall_Score","Overall_Index_0_100","Rank","Source_File"
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # accept if convertible to numeric for most rows
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.6:  # >=60% numeric values
            cols.append(c)
    return cols

def normalize_series(s: pd.Series, method: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if method == "assume_1_to_5":
        return (s / 5.0).clip(0, 1)
    elif method == "min_max":
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(np.nan, index=s.index)
        return (s - mn) / (mx - mn)
    else:  # auto
        mx = s.max(skipna=True)
        mn = s.min(skipna=True)
        # If it looks like a 1–5 score, scale by 5; else min-max.
        if (mn >= 0) and (mx <= 5):
            return (s / 5.0).clip(0, 1)
        return normalize_series(s, "min_max")

# ---------------- Upload ----------------
files = st.file_uploader(
    "Upload one or more Excel files (.xlsx) with a Companies sheet",
    type=["xlsx"],
    accept_multiple_files=True
)

if not files:
    st.info("Upload at least one Excel file to begin.")
    st.stop()

dfs = []
for f in files:
    try:
        df_i = load_companies_sheet(f)
        df_i["Source_File"] = getattr(f, "name", "uploaded.xlsx")
        dfs.append(df_i)
    except Exception as e:
        st.error(f"Failed to read {getattr(f, 'name', '')}: {e}")
        st.stop()

df = pd.concat(dfs, ignore_index=True)

# ---------------- Criteria selection ----------------
st.sidebar.header("Criteria selection")

available = detect_available_criteria(df)

has_old = all(c in df.columns for c in OLD_SCHEMA["criteria"])
has_new = all(c in df.columns for c in NEW_SCHEMA["criteria"])

preset_options = ["Custom (choose columns)"]
if has_old:
    preset_options.insert(0, OLD_SCHEMA["name"])
if has_new:
    preset_options.insert(0, NEW_SCHEMA["name"])

preset = st.sidebar.selectbox("Preset", preset_options)

if preset == OLD_SCHEMA["name"]:
    criteria = OLD_SCHEMA["criteria"]
elif preset == NEW_SCHEMA["name"]:
    criteria = NEW_SCHEMA["criteria"]
else:
    criteria = st.sidebar.multiselect(
        "Pick criteria columns",
        options=sorted(available),
        default=sorted([c for c in NEW_SCHEMA["criteria"] if c in available]) or sorted(available)[:5]
    )

if not criteria:
    st.warning("Select at least one criterion.")
    st.stop()

# ---------------- Scaling options ----------------
st.sidebar.header("Scaling")
scale_method = st.sidebar.selectbox(
    "Normalize criteria values",
    options=[
        ("auto", "Auto (1–5 → /5, else Min-Max)"),
        ("assume_1_to_5", "Assume all criteria are 1–5 scores"),
        ("min_max", "Min-Max normalize (0–1)"),
    ],
    format_func=lambda x: x[1]
)[0]

# ---------------- Weights ----------------
st.sidebar.header("Weights")
weights = []
for c in criteria:
    w = st.sidebar.slider(f"Weight: {c}", 0.0, 5.0, 3.0, 0.1)
    weights.append(w)

weights = np.array(weights, dtype=float)
if weights.sum() == 0:
    st.warning("All weights are zero. Increase at least one weight.")
    st.stop()

weights = weights / weights.sum()

# ---------------- Filters ----------------
st.sidebar.header("Filters")
sectors = sorted(df["Sector"].dropna().unique())
selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)

filtered = df[df["Sector"].isin(selected_sectors)].copy()

# ---------------- Compute MCDM ----------------
norm = pd.DataFrame(index=filtered.index)
for c in criteria:
    norm[c] = normalize_series(filtered[c], scale_method)

# Drop rows with too many NaNs in selected criteria
valid_mask = norm[criteria].notna().mean(axis=1) >= 0.6
filtered = filtered[valid_mask].copy()
norm = norm.loc[filtered.index].copy()

overall = (norm[criteria].fillna(0).to_numpy() * weights).sum(axis=1)
filtered["Overall_Score"] = overall
filtered["Overall_Index_0_100"] = (filtered["Overall_Score"] * 100).round(1)

# ---------------- Outputs ----------------
st.subheader("Company Ranking")
ranked = filtered.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
ranked["Rank"] = ranked.index + 1

show_cols = ["Rank", "Company", "Sector", "Overall_Index_0_100", "Source_File"] + criteria
st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)

st.subheader("Sector Ranking (Average)")
sector_df = (
    ranked.groupby("Sector", as_index=False)["Overall_Score"]
    .mean()
    .sort_values("Overall_Score", ascending=False)
)
sector_df["Overall_Index_0_100"] = (sector_df["Overall_Score"] * 100).round(1)
st.dataframe(sector_df[["Sector", "Overall_Index_0_100"]], use_container_width=True, hide_index=True)
st.bar_chart(sector_df.set_index("Sector")["Overall_Index_0_100"], use_container_width=True)

st.subheader("Compare Selected Companies")
choices = st.multiselect("Select companies", options=list(ranked["Company"].unique()))
if choices:
    comp = ranked[ranked["Company"].isin(choices)]
    st.dataframe(comp[["Company","Sector","Overall_Index_0_100","Source_File"] + criteria],
                 use_container_width=True, hide_index=True)

with st.expander("Show combined raw input table"):
    st.dataframe(df, use_container_width=True)
