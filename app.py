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
for c in ["GTO_Ownership%","Projected_Ownership%","Ceiling","Salary","RealScore","Leverage_%"]:
    if c in df_raw:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# 6) Rescale 600-slot ownerships to 0–100 if necessary (builder uses % of lineups)
def _maybe_to_percent(s: pd.Series) -> pd.Series:
    if s is None or s.isna().all():
        return s
    s_sum = s.fillna(0).sum()
    # If totals look like ~600 (slot-based), convert to ~100 scale
    return (s / 6.0) if s_sum > 200 else s

df_raw["GTO_Ownership%"]       = _maybe_to_percent(df_raw.get("GTO_Ownership%", pd.Series(dtype=float)))
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

# Sidebar — old rules as toggles
st.sidebar.header("Rules & Settings")
total_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 150)

# Salary slider hard-capped at 50,000
min_possible = int(df_pool["Salary"].min()) * 6
max_possible = 50000  # hard cap
default_min, default_max = 49700, 50000
default_min = max(min_possible, min(default_min, max_possible))
default_max = max(min_possible, min(default_max, max_possible))
min_sal, max_sal = st.sidebar.slider(
    "Target Salary Range",
    min_value=min_possible, max_value=max_possible,
    value=(default_min, default_max), step=100
)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range", True)

# Exposure cap
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", False)
max_exposure_pct = st.sidebar.slider("Max Exposure (%)", 1.0, 100.0, 26.5, step=0.5)

# Singleton
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule (each included at least once)", False)

# Double-punt
enforce_double = st.sidebar.checkbox("Limit Double-Punt Lineups", False)
double_threshold = st.sidebar.slider("Punt threshold: Projected_Ownership% <", 0.0, 10.0, 2.75, step=0.1)
max_double = st.sidebar.slider("Max Double-Punt Lineups", 0, 150, 15)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Player Pool", "📋 Builder Settings", "🎲 Lineups", "📊 Summary"
])

# -------------------------------
# Player Pool Tab
# -------------------------------
with tab1:
    st.subheader("Player Pool (toggle include / set targets)")
    master_use_gto = st.checkbox("✅ Use GTO for ALL", value=True, help="When ON, any blank targets default to GTO%")
    # reference table on top
    ref_cols = [c for c in ["Name","Salary","PlayerID","GTO_Ownership%","Projected_Ownership%","Ceiling","RealScore","Leverage_%"] if c in df_pool.columns]
    with st.expander("Reference: full scorecard columns", expanded=False):
        st.dataframe(df_pool[ref_cols], use_container_width=True)

    controls = []
    for i, row in df_pool.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3,1,1,1.2,2])
        with col1:
            include = st.checkbox(row["Name"], value=True, key=f"use_{i}")
        with col2:
            use_gto = st.checkbox("Use", value=False, key=f"gto_{i}", help="Use row GTO for this golfer")
        with col3:
            tgt_val = row["GTO_Ownership%"] if (use_gto or master_use_gto) else ""
            target = st.text_input("%", value=f"{tgt_val:.2f}" if tgt_val != "" else "", key=f"tgt_{i}")
        with col4:
            st.write(f"💰 ${int(row['Salary'])}")
        with col5:
            info_bits = []
            if "Projected_Ownership%" in df_pool.columns and not np.isnan(row["Projected_Ownership%"]):
                info_bits.append(f"PO {row['Projected_Ownership%']:.1f}%")
            if "Ceiling" in df_pool.columns and not np.isnan(row["Ceiling"]):
                info_bits.append(f"Ceil {row['Ceiling']:.1f}")
            if "Leverage_%":  # safe
                lev = row.get("Leverage_%", np.nan)
                if not pd.isna(lev):
                    info_bits.append(f"Lev {lev:+.1f}")
            st.caption(" • ".join(info_bits))
        controls.append({
            "Name": row["Name"],
            "PlayerID": row.get("PlayerID", ""),
            "Include": include,
            "UseGTO": use_gto,
            "Target%": float(target) if str(target).strip() not in ["", "None"] else None,
            "Salary": float(row["Salary"]),
            "GTO%": float(row["GTO_Ownership%"]),
            "PO%": float(row["Projected_Ownership%"]) if "Projected_Ownership%" in df_pool.columns else np.nan
        })
    df_controls = pd.DataFrame(controls)

# -------------------------------
# Settings Tab (echo)
# -------------------------------
with tab2:
    st.subheader("Builder Settings (echo)")
    st.markdown(
        f"- **Lineups:** {total_lineups}\n"
        f"- **Salary Range:** ${min_sal:,} – ${max_sal:,} {'(ENFORCED)' if enforce_salary else '(ignored)'}\n"
        f"- **Exposure Cap:** {'ON' if enforce_cap else 'OFF'} @ {max_exposure_pct:.1f}%\n"
        f"- **Singleton:** {'ON' if enforce_singleton else 'OFF'}\n"
        f"- **Double-Punt:** {'ON' if enforce_double else 'OFF'} (PO% < {double_threshold:.2f}%, max {max_double})"
    )

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
    po_map   = dict(zip(pool["Name"], pool["PO%"])) if "PO%" in pool.columns else {}

    # weights from Target% or UseGTO or master_use_gto
    raw_w = []
    for _, r in pool.iterrows():
        if r["Target%"] is not None:
            w = r["Target%"]
        elif r["UseGTO"] or master_use_gto:
            w = r["GTO%"]
        else:
            w = 1.0
        raw_w.append(max(w, 1e-6))
    weights = np.array(raw_w) / np.sum(raw_w)

    # constraints bookkeeping
    lineups = []
    seen = set()
    exposure = {n: 0 for n in names}
    attempts = 0
    max_attempts = num_lineups * 15000  # generous to find valid uniques

    # helpers
    def lineup_salary(lu_names):
        return int(sum(sal_map[n] for n in lu_names))

    def is_double_punt(lu_names):
        if not enforce_double: return False
        low = [n for n in lu_names if po_map.get(n, np.nan) < double_threshold]
        return len(low) >= 2

    def under_cap(lu_names):
        if not enforce_cap: return True
        max_cnt = int(np.floor(num_lineups * (max_exposure_pct/100.0)))
        return all(exposure[n] < max_cnt for n in lu_names)

    # singleton pass (optional)
    if enforce_singleton:
        singles = set(names)
        while singles and len(lineups) < num_lineups and attempts < max_attempts:
            attempts += 1
            seed = singles.pop()
            rest = [n for n in names if n != seed]
            cand = [seed] + list(np.random.choice(rest, 5, replace=False, p=[weights[names.index(x)] for x in rest]))
            s = lineup_salary(cand)
            key = tuple(sorted(cand))
            if (not enforce_salary or (sal_range[0] <= s <= sal_range[1])) and (key not in seen) and under_cap(cand) and (not is_double_punt(cand) or len(lineups) < max_double):
                # accept
                seen.add(key)
                lineups.append({"names":cand, "ids":[ids_map.get(n,"") for n in cand], "salary": s})
                for n in cand: exposure[n] += 1

    # fill remaining
    while len(lineups) < num_lineups and attempts < max_attempts:
        attempts += 1
        cand = list(np.random.choice(names, 6, replace=False, p=weights))
        s = lineup_salary(cand)
        key = tuple(sorted(cand))
        if enforce_salary and not (sal_range[0] <= s <= sal_range[1]): 
            continue
        if key in seen: 
            continue
        if not under_cap(cand):
            continue
        if enforce_double and is_double_punt(cand):
            # count how many built so far are double punt
            double_count = sum(
                1 for lu in lineups if sum(1 for p in lu["names"] if po_map.get(p, np.nan) < double_threshold) >= 2
            )
            if double_count >= max_double:
                continue

        # accept
        seen.add(key)
        lineups.append({"names":cand, "ids":[ids_map.get(n,"") for n in cand], "salary": s})
        for n in cand: exposure[n] += 1

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
        st.info("Set your targets and rules, then click **Run Builder**.")

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

        base_cols = ["Name","PlayerID","Salary","GTO%","Target%","PO%"]
        base_cols = [c for c in base_cols if c in df_controls.columns]
        base = df_controls[base_cols].copy()

        summary = base.merge(counts, on="PlayerID", how="left").fillna({"Count":0})
        # Exposure% scaled to 600 like GTO
        total_lineups_safe = max(len(lus), 1)
        summary["Exposure%"] = (summary["Count"] / total_lineups_safe) * 100 / 6 * 600
        keep = [c for c in ["Name","PlayerID","Salary","GTO%","Target%","PO%","Exposure%"] if c in summary.columns]
        summary = summary[keep].sort_values("Exposure%", ascending=False)
        st.dataframe(summary, use_container_width=True)

        # Lineup salary stats
        salaries = [lu["salary"] for lu in lus]
        avg_salary = float(np.mean(salaries))
        min_salary = int(np.min(salaries))
        max_salary = int(np.max(salaries))
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


