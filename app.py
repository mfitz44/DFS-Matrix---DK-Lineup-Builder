# app.py — DFS Matrix • Flexible Lineup Builder (DK-ready)
# - Upload your GTO scorecard CSV (from the Scalar Resonance app or any source)
# - App auto-maps columns (Adj_GTO_% -> GTO_Ownership%, PO_% -> Projected_Ownership%, RG_ceil -> Ceiling, etc.)
# - Set include toggles + target ownerships (or Use GTO)
# - Choose salary range + number of lineups
# - Generate DK-ready CSV (PlayerIDs only, no headers)
# - See exposure summary (Name, PlayerID, Salary, GTO%, Target%, Exposure%, Avg/Min/Max lineup salary)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DFS Matrix — Flexible Lineup Builder", layout="wide")
st.title("🏌️ DFS Matrix — Flexible GTO Lineup Builder")

# -------------------------------
# Upload scorecard
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload your scorecard CSV to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# -------------------------------
# Normalize / Shim the scorecard
# -------------------------------
# 1) Ensure Name column
if "Name" not in df_raw and "Player" in df_raw:
    df_raw["Name"] = df_raw["Player"]

# 2) Map GTO (prefer private Adj_GTO_% if present)
if "GTO_Ownership%" not in df_raw:
    if "Adj_GTO_%" in df_raw:
        df_raw["GTO_Ownership%"] = df_raw["Adj_GTO_%"]
    elif "GTO_%" in df_raw:
        df_raw["GTO_Ownership%"] = df_raw["GTO_%"]

# 3) Map Projected Ownership (public / field)
if "Projected_Ownership%" not in df_raw:
    if "PO_%" in df_raw:
        df_raw["Projected_Ownership%"] = df_raw["PO_%"]
    elif "RG_proj_own" in df_raw:
        df_raw["Projected_Ownership%"] = df_raw["RG_proj_own"]

# 4) Map Ceiling
if "Ceiling" not in df_raw:
    if "RG_ceil" in df_raw:
        df_raw["Ceiling"] = df_raw["RG_ceil"]
    else:
        # fallback if RG_ceil absent: try to synthesize from projection/floor
        if "RG_fpts" in df_raw and "RG_floor" in df_raw:
            df_raw["Ceiling"] = df_raw["RG_fpts"] + (df_raw["RG_fpts"] - df_raw["RG_floor"]).clip(lower=0)
        else:
            df_raw["Ceiling"] = np.nan

# 5) Coerce to numerics where needed
for c in ["GTO_Ownership%","Projected_Ownership%","Ceiling","Salary"]:
    if c in df_raw:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# 6) Rescale 600-slot ownerships to 0–100 if necessary (builder uses % of lineups)
def _maybe_to_percent(s: pd.Series) -> pd.Series:
    if s is None or s.isna().all():
        return s
    s_sum = s.fillna(0).sum()
    # If totals look like ~600 (slot-based), convert to ~100 scale
    return (s / 6.0) if s_sum > 200 else s

df_raw["GTO_Ownership%"]      = _maybe_to_percent(df_raw.get("GTO_Ownership%", pd.Series(dtype=float)))
df_raw["Projected_Ownership%"] = _maybe_to_percent(df_raw.get("Projected_Ownership%", pd.Series(dtype=float)))

# 7) Ensure PlayerID is string (needed for DK export)
if "PlayerID" in df_raw:
    df_raw["PlayerID"] = df_raw["PlayerID"].astype(str)

# 8) Minimal presence check
required = ["Name","Salary","GTO_Ownership%","Ceiling"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Scorecard is missing required columns: {missing}")
    st.stop()

# -------------------------------
# Prepare player pool
# -------------------------------
df_pool = df_raw.dropna(subset=["Name","Salary","GTO_Ownership%"]).reset_index(drop=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Player Pool", "⚙️ Builder Settings", "🎲 Lineups", "📊 Summary"
])

# -------------------------------
# Player Pool Tab
# -------------------------------
with tab1:
    st.subheader("Player Pool (toggle include / set targets)")
    controls = []
    for i, row in df_pool.iterrows():
        col1, col2, col3, col4 = st.columns([3,1,1,2])
        with col1:
            include = st.checkbox(row["Name"], value=True, key=f"use_{i}")
        with col2:
            use_gto = st.checkbox("Use GTO", key=f"gto_{i}")
        with col3:
            tgt_val = row["GTO_Ownership%"] if use_gto else ""
            target = st.text_input("%", value=f"{tgt_val:.2f}" if tgt_val != "" else "", key=f"tgt_{i}")
        with col4:
            st.write(f"${int(row['Salary'])}")
        controls.append({
            "Name": row["Name"],
            "PlayerID": row.get("PlayerID", ""),
            "Include": include,
            "Target%": float(target) if str(target).strip() not in ["", "None"] else None,
            "Salary": float(row["Salary"]),
            "GTO%": float(row["GTO_Ownership%"])
        })
    df_controls = pd.DataFrame(controls)

# -------------------------------
# Settings Tab
# -------------------------------
with tab2:
    st.subheader("Builder Settings")
    total_lineups = st.slider("Number of Lineups", 1, 150, 150)
    # reasonable salary slider bounds based on pool salaries
    min_possible = int(df_pool["Salary"].min()) * 6
    max_possible = int(df_pool["Salary"].max()) * 6
    default_min, default_max = 49700, 50000
    # clamp defaults to computed bounds
    default_min = max(min_possible, min(default_min, max_possible))
    default_max = max(min_possible, min(default_max, max_possible))
    min_sal, max_sal = st.slider(
        "Target Salary Range",
        min_value=min_possible, max_value=max_possible,
        value=(default_min, default_max), step=100
    )
    st.markdown(f"**Lineups to Generate:** {total_lineups}  \n"
                f"**Salary Range:** ${min_sal:,} – ${max_sal:,}")

# -------------------------------
# Lineup generator
# -------------------------------
def build_lineups(df_ctrl: pd.DataFrame, num_lineups: int, sal_range: tuple[int,int]):
    pool = df_ctrl[df_ctrl["Include"]].copy()
    if pool.empty:
        return []

    names    = pool["Name"].tolist()
    ids_map  = dict(zip(pool["Name"], pool["PlayerID"]))
    sal_map  = dict(zip(pool["Name"], pool["Salary"]))

    # Weights from Target% (default 1.0 if blank)
    raw_w = []
    for _, r in pool.iterrows():
        w = r["Target%"] if r["Target%"] is not None else 1.0
        raw_w.append(max(w, 0.000001))
    weights = np.array(raw_w) / np.sum(raw_w)

    lineups = []
    seen = set()
    attempts = 0
    max_attempts = num_lineups * 10000  # generous to find valid uniques

    while len(lineups) < num_lineups and attempts < max_attempts:
        attempts += 1
        cand = list(np.random.choice(names, 6, replace=False, p=weights))
        total_salary = int(sum(sal_map[n] for n in cand))
        key = tuple(sorted(cand))
        if sal_range[0] <= total_salary <= sal_range[1] and key not in seen:
            seen.add(key)
            lineup_ids = [ids_map.get(n, "") for n in cand]
            lineups.append({"names": cand, "ids": lineup_ids, "salary": total_salary})

    return lineups

# Keep lineups in session
if "lineups" not in st.session_state:
    st.session_state["lineups"] = []

# -------------------------------
# Lineups Tab
# -------------------------------
with tab3:
    st.subheader("Generated Lineups")
    if st.button("Run Builder"):
        with st.spinner("Generating lineups…"):
            lus = build_lineups(df_controls, total_lineups, (min_sal, max_sal))
            st.session_state["lineups"] = lus

            if lus:
                # Display names view
                df_names = pd.DataFrame(
                    [{f"P{i+1}": p for i, p in enumerate(lu["names"])} | {"Salary": lu["salary"]}
                     for lu in lus]
                )
                st.dataframe(df_names, use_container_width=True)

                # DK CSV: PlayerIDs only, no headers
                dk_df = pd.DataFrame([lu["ids"] for lu in lus])
                st.download_button(
                    "📥 Download DraftKings CSV (PlayerIDs only)",
                    dk_df.to_csv(index=False, header=False),
                    file_name="dfs_matrix_lineups.csv"
                )
            else:
                st.warning("No valid lineups found with current settings. Try widening salary range or adjusting targets.")
    else:
        st.info("Configure targets and click **Run Builder**.")

# -------------------------------
# Summary Tab
# -------------------------------
with tab4:
    st.subheader("Exposure Summary")
    lus = st.session_state["lineups"]
    if lus:
        # Count exposures by PlayerID
        all_ids = [pid for lu in lus for pid in lu["ids"]]
        counts = pd.Series(all_ids).value_counts().rename_axis("PlayerID").reset_index(name="Count")
        # Merge back names/salaries/targets/GTO
        base = df_controls[["Name","PlayerID","Salary","GTO%","Target%"]].copy()
        summary = base.merge(counts, on="PlayerID", how="left").fillna({"Count":0})
        # Exposure% scaled to 600 like GTO scale
        # (Each lineup has 6 golfers → exposures in "slots"; scale to 600 to compare with 600-slot GTO if desired)
        total_lineups_safe = max(len(lus), 1)
        summary["Exposure%"] = (summary["Count"] / total_lineups_safe) * 100 / 6 * 600
        summary = summary[["Name","PlayerID","Salary","GTO%","Target%","Exposure%"]].sort_values("Exposure%", ascending=False)

        st.dataframe(summary, use_container_width=True)

        # Lineup salary stats
        salaries = [lu["salary"] for lu in lus]
        avg_salary = np.mean(salaries)
        min_salary = np.min(salaries)
        max_salary = np.max(salaries)
        st.markdown(f"**Average Lineup Salary:** ${avg_salary:,.0f}  \n"
                    f"**Min Salary:** ${min_salary:,}  \n"
                    f"**Max Salary:** ${max_salary:,}")

        st.download_button(
            "📥 Download Exposure Summary",
            summary.to_csv(index=False),
            file_name="dfs_matrix_exposures.csv"
        )
    else:
        st.info("No lineups built yet.")
