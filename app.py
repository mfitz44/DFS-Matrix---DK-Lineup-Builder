import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DFS Matrix — Targeted Monte Carlo Builder", layout="wide")
st.title("🏌️ DFS Matrix — Targeted Monte Carlo Builder")

# ==============================
# Upload scorecard
# ==============================
uploaded_file = st.sidebar.file_uploader("Upload GTO Scorecard CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload your scorecard CSV to begin.")
    st.stop()
df_raw = pd.read_csv(uploaded_file)

# ==============================
# Normalize / Shim the scorecard
# ==============================
# Ensure Name
if "Name" not in df_raw and "Player" in df_raw:
    df_raw["Name"] = df_raw["Player"]

# Map GTO (prefer Adj_GTO_%)
if "GTO_Ownership%" not in df_raw:
    if "Adj_GTO_%" in df_raw:
        df_raw["GTO_Ownership%"] = df_raw["Adj_GTO_%"]
    elif "GTO_%" in df_raw:
        df_raw["GTO_Ownership%"] = df_raw["GTO_%"]

# Map PO (public)
if "Projected_Ownership%" not in df_raw:
    if "PO_%" in df_raw:
        df_raw["Projected_Ownership%"] = df_raw["PO_%"]
    elif "RG_proj_own" in df_raw:
        df_raw["Projected_Ownership%"] = df_raw["RG_proj_own"]

# Map Ceiling
if "Ceiling" not in df_raw:
    if "RG_ceil" in df_raw:
        df_raw["Ceiling"] = df_raw["RG_ceil"]
    elif "RG_fpts" in df_raw and "RG_floor" in df_raw:
        df_raw["Ceiling"] = df_raw["RG_fpts"] + (df_raw["RG_fpts"] - df_raw["RG_floor"]).clip(lower=0)
    else:
        df_raw["Ceiling"] = np.nan

# Coerce numerics we use
for c in ["GTO_Ownership%","Projected_Ownership%","Ceiling","Salary","RealScore","Leverage_%"]:
    if c in df_raw:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

# Promote 100-scale ownerships to 600 (so input can be either)
def ensure_600_scale(s: pd.Series) -> pd.Series:
    if s is None or s.isna().all():
        return s
    total = s.fillna(0).sum()
    if 80 <= total <= 120:  # likely 100-scale
        return s * 6.0
    return s

df_raw["GTO_Ownership%"]       = ensure_600_scale(df_raw.get("GTO_Ownership%", pd.Series(dtype=float)))
df_raw["Projected_Ownership%"] = ensure_600_scale(df_raw.get("Projected_Ownership%", pd.Series(dtype=float)))

# Require basics
required = ["Name","Salary","GTO_Ownership%"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Scorecard is missing required columns: {missing}")
    st.stop()

# ==============================
# Prepare pool
# ==============================
df_pool = df_raw.dropna(subset=["Name","Salary","GTO_Ownership%"]).reset_index(drop=True)
if "PlayerID" in df_pool:
    df_pool["PlayerID"] = df_pool["PlayerID"].astype(str)

# ==============================
# Sidebar rules & tuning
# ==============================
st.sidebar.header("Rules & Settings")
L = st.sidebar.slider("Number of Lineups (L)", 1, 150, 150)

# Salary range hard-capped at 50,000
min_possible = int(df_pool["Salary"].min()) * 6
max_possible = 50000
default_min, default_max = 49700, 50000
default_min = max(min_possible, min(default_min, max_possible))
default_max = max(min_possible, min(default_max, max_possible))
sal_min, sal_max = st.sidebar.slider(
    "Salary Range", min_value=min_possible, max_value=max_possible,
    value=(default_min, default_max), step=100
)
enforce_salary = st.sidebar.checkbox("Enforce Salary Range", True)

# Exposure cap (% of lineups)
enforce_cap = st.sidebar.checkbox("Enforce Exposure Cap", False)
cap_pct = st.sidebar.slider("Max Exposure (% of lineups)", 1.0, 100.0, 26.5, step=0.5)

# Singleton
enforce_singleton = st.sidebar.checkbox("Enforce Singleton Rule (each included ≥ 1 lineup)", False)

# Double-punt: threshold is % of lineups; PO_600 / 6
enforce_double = st.sidebar.checkbox("Limit Double-Punt Lineups", False)
double_threshold = st.sidebar.slider("Punt threshold: Projected_Ownership% of lineups <", 0.0, 10.0, 2.75, step=0.1)
max_double = st.sidebar.slider("Max Double-Punt Lineups", 0, 150, 15)

st.sidebar.header("Ownership Tolerance")
tol_pp = st.sidebar.slider("Absolute tolerance (p.p.)", 0.0, 5.0, 2.0, step=0.5)
rel_tol = st.sidebar.slider("Relative tolerance (% of target)", 0.0, 30.0, 10.0, step=1.0)
kappa = st.sidebar.slider("Sampling Tightness (Dirichlet κ)", 20, 400, 120, step=10)
backoff_limit = st.sidebar.slider("Max backoff rounds", 0, 3, 2)

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Player Pool", "📋 Builder Settings", "🎲 Lineups", "📊 Summary"
])

# ==============================
# Player Pool tab (controls)
# ==============================
with tab1:
    st.subheader("Player Pool (toggle include / set targets)")
    master_use_gto = st.checkbox("✅ Use GTO for ALL (targets on 600 scale)", value=True)

    ref_cols = [c for c in ["Name","Salary","PlayerID","GTO_Ownership%","Projected_Ownership%","Ceiling","RealScore","Leverage_%"] if c in df_pool.columns]
    with st.expander("Reference: full scorecard columns", expanded=False):
        st.dataframe(df_pool[ref_cols], use_container_width=True)

    controls = []
    for i, row in df_pool.iterrows():
        c1, c2, c3, c4, c5 = st.columns([3,1,1,1.2,2])
        with c1:
            include = st.checkbox(row["Name"], value=True, key=f"inc_{i}")
        with c2:
            use_gto = st.checkbox("Use", value=False, key=f"usegto_{i}", help="Use GTO for this golfer")
        with c3:
            tgt_val = row["GTO_Ownership%"] if (use_gto or master_use_gto) else ""
            target = st.text_input("% (600)", value=f"{tgt_val:.2f}" if tgt_val != "" else "", key=f"tgt_{i}")
        with c4:
            st.write(f"💰 ${int(row['Salary'])}")
        with c5:
            info_bits = []
            if "Projected_Ownership%" in df_pool.columns and not np.isnan(row["Projected_Ownership%"]):
                info_bits.append(f"PO {row['Projected_Ownership%']/6.0:.1f}%")
            if "Ceiling" in df_pool.columns and not np.isnan(row["Ceiling"]):
                info_bits.append(f"Ceil {row['Ceiling']:.1f}")
            lev = row.get("Leverage_%", np.nan)
            if not pd.isna(lev):
                info_bits.append(f"Lev {lev:+.1f}")
            st.caption(" • ".join(info_bits))

        controls.append({
            "Name": row["Name"],
            "PlayerID": row.get("PlayerID", ""),
            "Include": include,
            "UseGTO": use_gto,
            "Target600": float(target) if str(target).strip() not in ["","None"] else None,  # stored on 600 scale
            "Salary": float(row["Salary"]),
            "GTO600": float(row["GTO_Ownership%"]),
            "PO600": float(row["Projected_Ownership%"]) if "Projected_Ownership%" in df_pool.columns and pd.notna(row["Projected_Ownership%"]) else np.nan
        })
    df_controls = pd.DataFrame(controls)

# ==============================
# Settings tab (echo)
# ==============================
with tab2:
    st.subheader("Builder Settings")
    st.markdown(
        f"- **Lineups (L):** {L}\n"
        f"- **Salary Range:** ${sal_min:,}–${sal_max:,} {'(ENFORCED)' if enforce_salary else '(ignored)'}\n"
        f"- **Exposure Cap:** {'ON' if enforce_cap else 'OFF'} @ {cap_pct:.1f}%\n"
        f"- **Singleton:** {'ON' if enforce_singleton else 'OFF'}\n"
        f"- **Double-Punt:** {'ON' if enforce_double else 'OFF'} (PO% of lineups < {double_threshold:.2f}%, max {max_double})\n"
        f"- **Tolerance:** {tol_pp:.1f} p.p. & {rel_tol:.0f}% of target; Dirichlet κ = {kappa}; Backoff rounds = {backoff_limit}"
    )

# ==============================
# Builder Core
# ==============================
def compute_bands(df_ctrl, L, tol_pp, rel_tol):
    """Compute lower/upper counts per player from targets (600-scale → % of lineups)."""
    # target % of lineups
    t_pct = []
    for _, r in df_ctrl.iterrows():
        if r["Target600"] is not None:
            t = r["Target600"] / 6.0
        elif r["UseGTO"] or master_use_gto:
            t = r["GTO600"] / 6.0
        else:
            t = 100.0 / len(df_ctrl)  # uniform fallback
        t_pct.append(max(t, 0.0))

    df = df_ctrl.copy()
    df["t_pct"] = t_pct
    df["T"] = np.rint(L * df["t_pct"] / 100.0).astype(int)

    A = int(np.rint(L * tol_pp / 100.0))  # absolute band in counts
    rel = np.ceil((rel_tol/100.0) * df["T"]).astype(int)
    band = np.maximum(A, np.maximum(rel, 1))
    df["lower"] = np.maximum(0, df["T"] - band)
    df["upper"] = np.minimum(L, df["T"] + band)
    return df[["Name","t_pct","T","lower","upper"]]

def draw_probs(df_ctrl):
    """Dirichlet draw around normalized target vector (on % of lineups)."""
    # base weights from target %
    base = []
    for _, r in df_ctrl.iterrows():
        if r["Target600"] is not None:
            w = r["Target600"]/6.0
        elif r["UseGTO"] or master_use_gto:
            w = r["GTO600"]/6.0
        else:
            w = 100.0/len(df_ctrl)
        base.append(max(w, 1e-8))
    base = np.array(base, dtype=float)
    base = base / base.sum()  # normalize to 1
    alpha = kappa * base
    return np.random.dirichlet(alpha)

def build_lineups(df_ctrl, L, sal_range, bands, attempts_limit, backoff_limit):
    pool = df_ctrl[df_ctrl["Include"]].copy()
    if pool.empty:
        return [], {}

    names = pool["Name"].tolist()
    idx_map = {n:i for i,n in enumerate(names)}
    id_map  = dict(zip(pool["Name"], pool["PlayerID"])) if "PlayerID" in pool else {}
    sal_map = dict(zip(pool["Name"], pool["Salary"]))
    po_map  = dict(zip(pool["Name"], pool["PO600"])) if "PO600" in pool else {}

    bands = bands.set_index("Name").to_dict(orient="index")
    counts = {n:0 for n in names}
    lineups, seen = [], set()

    backoff = 0
    attempts = 0

    def under_cap(cand):
        if not enforce_cap: return True
        max_cnt = int(np.floor(L * (cap_pct/100.0)))
        return all(counts[n] < max_cnt for n in cand)

    def double_punt(cand):
        if not enforce_double: return False
        # PO600 / 6 → % of lineups
        low = [n for n in cand if (po_map.get(n, np.nan) / 6.0) < double_threshold]
        return len(low) >= 2

    def salary_ok(cand):
        s = int(sum(sal_map[n] for n in cand))
        return (not enforce_salary) or (sal_range[0] <= s <= sal_range[1])

    def add_lineup(cand):
        lineups.append({
            "names": cand,
            "ids": [id_map.get(n, "") for n in cand],
            "salary": int(sum(sal_map[n] for n in cand))
        })
        for n in cand:
            counts[n] += 1

    # Singleton pass (optional)
    if enforce_singleton:
        for n in names:
            attempts += 1
            if attempts > attempts_limit: break
            # sample 5 others with current probs
            probs = draw_probs(pool)
            rest = [m for m in names if m != n]
            rest_probs = np.array([probs[idx_map[m]] for m in rest])
            rest_probs = rest_probs/rest_probs.sum() if rest_probs.sum() > 0 else np.ones_like(rest_probs)/len(rest_probs)
            cand = [n] + list(np.random.choice(rest, 5, replace=False, p=rest_probs))
            key = tuple(sorted(cand))
            if key in seen: continue
            if not salary_ok(cand): continue
            # upper cap
            if any(counts[p] >= bands[p]["upper"] for p in cand): continue
            if not under_cap(cand): continue
            if double_punt(cand):
                # allow only if we haven't exceeded global double-punt count
                dp_so_far = sum(1 for lu in lineups if double_punt(lu["names"]))
                if dp_so_far >= max_double: continue
            seen.add(key)
            add_lineup(cand)
            if len(lineups) >= L: break

    # Main sampling loop
    while len(lineups) < L and backoff <= backoff_limit:
        probs = draw_probs(pool)
        tries_this_round = 0
        while len(lineups) < L and attempts < attempts_limit and tries_this_round < attempts_limit//5:
            attempts += 1
            tries_this_round += 1
            cand = list(np.random.choice(names, 6, replace=False, p=probs))
            key = tuple(sorted(cand))
            if key in seen: continue
            if not salary_ok(cand): continue
            # don't overshoot any upper bound
            if any(counts[p] >= bands[p]["upper"] for p in cand): continue
            if not under_cap(cand): continue
            if double_punt(cand):
                dp_so_far = sum(1 for lu in lineups if double_punt(lu["names"]))
                if dp_so_far >= max_double: continue
            seen.add(key)
            add_lineup(cand)

        # if we stalled, widen bands by +1 lineup (backoff)
        if len(lineups) < L and attempts >= attempts_limit:
            backoff += 1
            attempts = 0
            for p in bands:
                bands[p]["upper"] = min(L, bands[p]["upper"] + 1)
                bands[p]["lower"] = max(0, bands[p]["lower"] - 0)  # keep lower; we fix with repair
            st.info(f"Backoff widening bands (+1). Round {backoff}/{backoff_limit}")

    # --------- Repair pass to meet lower bounds ----------
    if len(lineups) == 0:
        return lineups, counts

    # Build indices
    lu_by_player = {n:set() for n in names}
    for li, lu in enumerate(lineups):
        for p in lu["names"]:
            lu_by_player[p].add(li)

    need = [p for p in names if counts[p] < bands[p]["lower"]]
    surplus = [p for p in names if counts[p] > bands[p]["upper"]]

    # Greedy swaps
    for u in need:
        needed = bands[u]["lower"] - counts[u]
        if needed <= 0: continue
        # find candidate lineups to inject u
        for _ in range(needed):
            # choose a surplus player lineup to swap out
            found = False
            # refresh surplus each time
            surplus = [p for p in names if counts[p] > bands[p]["upper"]]
            for s in surplus:
                # try each lineup that currently has s and not u
                for li in list(lu_by_player[s]):
                    cand_names = lineups[li]["names"][:]
                    if u in cand_names or s not in cand_names: continue
                    # swap s -> u
                    new_names = cand_names[:]
                    new_names[new_names.index(s)] = u
                    # salary & constraints
                    s_new = int(sum(sal_map[n] for n in new_names))
                    if enforce_salary and not (sal_min <= s_new <= sal_max):
                        continue
                    # exposure cap (preview)
                    if enforce_cap:
                        max_cnt = int(np.floor(L * (cap_pct/100.0)))
                        if counts[u] >= max_cnt:
                            continue
                    # double-punt check
                    if enforce_double:
                        low = [n for n in new_names if (po_map.get(n, np.nan) / 6.0) < double_threshold]
                        # count double-punts overall if we replace
                        new_dp = sum(1 for lu in lineups if len([p for p in lu["names"] if (po_map.get(p, np.nan) / 6.0) < double_threshold]) >= 2)
                        if len(low) >= 2 and new_dp >= max_double:
                            continue

                    # apply swap
                    # update lineups
                    lineups[li]["names"] = new_names
                    lineups[li]["ids"]   = [id_map.get(n, "") for n in new_names] if "PlayerID" in df_pool else [""]*6
                    lineups[li]["salary"]= s_new
                    # update counts & indexes
                    counts[s] -= 1
                    counts[u] += 1
                    lu_by_player[s].remove(li)
                    lu_by_player[u].add(li)
                    found = True
                    break
                if found: break
            # if we couldn't repair further, break
            if not found:
                break

    return lineups, counts

# ==============================
# Run builder
# ==============================
if "lineups" not in st.session_state:
    st.session_state["lineups"] = []
    st.session_state["counts"] = {}

attempts_limit = L * 8000  # per backoff round

with tab3:
    st.subheader("Lineups")
    if st.button("Run Builder"):
        with st.spinner("Sampling and repairing to hit ownership bands…"):
            bands = compute_bands(df_controls, L, tol_pp, rel_tol)
            lus, counts = build_lineups(df_controls[df_controls["Include"]], L, (sal_min, sal_max), bands, attempts_limit, backoff_limit)
            st.session_state["lineups"], st.session_state["counts"] = lus, counts

            if lus:
                df_names = pd.DataFrame(
                    [{f"P{i+1}": p for i,p in enumerate(lu["names"])} | {"Salary": lu["salary"]} for lu in lus]
                )
                st.dataframe(df_names, use_container_width=True)
                # DK CSV: PlayerIDs only, no headers
                ids_df = pd.DataFrame([lu["ids"] for lu in lus])
                st.download_button("📥 Download DraftKings CSV (PlayerIDs only)",
                                   ids_df.to_csv(index=False, header=False),
                                   file_name="dfs_matrix_lineups.csv")
            else:
                st.warning("No valid lineups found. Try widening salary, increasing tolerances, or reducing constraints.")
    else:
        st.info("Adjust targets & rules, then click **Run Builder**.")

# ==============================
# Summary (true % of lineups)
# ==============================
with tab4:
    st.subheader("Exposure Summary (true % of lineups)")
    lus = st.session_state["lineups"]
    counts = st.session_state["counts"]
    if lus:
        # Build summary
        expo = pd.Series(counts, name="Count").rename_axis("Name").reset_index()
        expo["Exposure%"] = (expo["Count"] / L) * 100.0

        base = df_controls[["Name","PlayerID","Salary","GTO600","Target600","PO600"]].copy()
        base = base.rename(columns={"GTO600":"GTO% (600)","Target600":"Target% (600)","PO600":"PO% (600)"})
        summary = base.merge(expo, on="Name", how="left").fillna({"Count":0,"Exposure%":0.0})
        # Optional leverage on 600 scale if PO available
        if "PO% (600)" in summary.columns:
            summary["Leverage (GTO-PO)"] = summary["GTO% (600)"] - summary["PO% (600)"]

        cols = [c for c in ["Name","PlayerID","Salary","GTO% (600)","Target% (600)","PO% (600)","Exposure%","Leverage (GTO-PO)"] if c in summary.columns]
        summary = summary[cols].sort_values("Exposure%", ascending=False)
        st.dataframe(summary, use_container_width=True)

        # Salary stats
        sals = [lu["salary"] for lu in lus]
        st.markdown(f"**Average Lineup Salary:** ${np.mean(sals):,.0f}  \n"
                    f"**Min Salary:** ${np.min(sals):,}  \n"
                    f"**Max Salary:** ${np.max(sals):,}")

        st.download_button("📥 Download Exposure Summary",
                           summary.to_csv(index=False),
                           file_name="dfs_matrix_exposures.csv")
    else:
        st.info("No lineups built yet.")







