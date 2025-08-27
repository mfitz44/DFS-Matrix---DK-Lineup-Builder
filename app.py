# app.py  —  DFS Matrix Lineup Builder (Beta)
# -------------------------------------------------------------
# Guided flow (Steps 1–7) + Bubble Theory math in Step 7:
# - BGTO% (baseline): salary-aware, geometry-aware utility with quality, risk,
#   and value/ceiling per $1k — NO ownership term.
# - AGTO% (adjusted): distort baseline ONLY by ownership spin using geometry α.
# Includes: robust imputation by salary bins + λ bisection to hit spend target,
#           DK CSV export and exposure summary, and a debug panel.

import streamlit as st
import pandas as pd
import numpy as np
import re

# -------------------------------------------------------------
# Streamlit page
# -------------------------------------------------------------
st.set_page_config(page_title="DFS Matrix — Lineup Builder (Beta)", layout="wide")
st.title("🏗️ DFS Matrix — Lineup Builder (Beta)")

# -------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------
def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9\-\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def first_present(df: pd.DataFrame, cols) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c].astype(float)
    return pd.Series(1.0, index=df.index)

def probify(s: pd.Series) -> pd.Series:
    """Convert to fractional probability 0..1 (accept % or 0..1)."""
    s = s.astype(float)
    if s.max() > 1.5:
        return (s / 100.0).clip(0.0, 1.0)
    return s.clip(0.0, 1.0)

def dedup_and_canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one canonical Name/PlayerID/Salary; keep other unique columns."""
    cols = df.columns.tolist()
    # Name
    name_cols = [c for c in cols if c.lower() in ["name", "player", "player name", "player_name"]]
    if name_cols:
        keep = name_cols[0]
        if keep != "Name":
            df.rename(columns={keep: "Name"}, inplace=True)
        for c in name_cols:
            if c != "Name":
                df.drop(columns=[c], inplace=True, errors="ignore")
    # PlayerID
    id_cols = [c for c in df.columns if c.lower() in ["playerid", "player id", "dk_id", "dkid", "id"]]
    if id_cols:
        keep = id_cols[0]
        if keep != "PlayerID":
            df.rename(columns={keep: "PlayerID"}, inplace=True)
        for c in id_cols:
            if c != "PlayerID":
                df.drop(columns=[c], inplace=True, errors="ignore")
    # Salary
    sal_cols = [c for c in df.columns if c.lower() in ["salary", "dk salary"]]
    if sal_cols:
        keep = sal_cols[0]
        if keep != "Salary":
            df.rename(columns={keep: "Salary"}, inplace=True)
        for c in sal_cols:
            if c != "Salary":
                df.drop(columns=[c], inplace=True, errors="ignore")
    return df

# -------------------------------------------------------------
# Imputation by salary bins (prevents flat outputs)
# -------------------------------------------------------------
def fill_by_salary_bins(df: pd.DataFrame, cols, bin_size: int = 500) -> pd.DataFrame:
    """Fill NaNs by median within salary bins, then global median."""
    if "Salary" not in df.columns:
        return df
    sal = df["Salary"].astype(float)
    # Build bins that actually cover values
    lo = int(np.floor(sal.min() / bin_size) * bin_size)
    hi = int(np.ceil(sal.max() / bin_size) * bin_size) + bin_size
    bins = np.arange(lo, hi + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([lo, lo + bin_size])
    df["_sal_bin"] = pd.cut(sal, bins=bins, include_lowest=True)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        s = s.fillna(df.groupby("_sal_bin")[c].transform("median"))
        s = s.fillna(s.median())
        df[c] = s
    df.drop(columns=["_sal_bin"], inplace=True)
    return df

# -------------------------------------------------------------
# Coherence + value terms for golf
# -------------------------------------------------------------
def pick_prob(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return probify(df[c])
    return pd.Series(0.0, index=df.index)

def coherence_terms(df: pd.DataFrame):
    # DataGolf odds (or equivalent)
    Win    = pick_prob(df, ["DG_Win%","win","Win%"])
    Top20  = pick_prob(df, ["DG_Top20%","top_20","Top20%"])
    MakeCt = pick_prob(df, ["DG_MakeCut%","make_cut","MakeCut%"])
    # Real coherence (quality) & risk proxy
    ReC = 2.0*Win + 0.5*Top20 - 0.5*(1.0 - MakeCt)

    Proj = first_present(df, ["Proj_Points","RG_Proj","fpts"]).clip(lower=0)
    Ceil = first_present(df, ["Ceiling","RG_Ceil","ceil"]).fillna(Proj)
    vol  = (Ceil - Proj)
    vol  = (vol - vol.mean()) / (vol.std(ddof=0) if vol.std(ddof=0)>0 else 1.0)
    Risk = (1.0 - MakeCt) + vol.clip(lower=0)  # miss-cut + excess vol
    return ReC.fillna(0.0), Risk.fillna(0.0)

def value_terms(df: pd.DataFrame):
    Proj = first_present(df, ["Proj_Points","RG_Proj","fpts"]).clip(lower=0)
    Ceil = first_present(df, ["Ceiling","RG_Ceil","ceil"]).fillna(Proj)
    Sal  = df["Salary"].astype(float).clip(lower=1.0)
    eta  = Proj / (Sal/1000.0)   # value per $1k
    chi  = Ceil / (Sal/1000.0)   # ceiling per $1k
    eta_z = (eta - eta.mean()) / (eta.std(ddof=0) if eta.std(ddof=0)>0 else 1.0)
    chi_z = (chi - chi.mean()) / (chi.std(ddof=0) if chi.std(ddof=0)>0 else 1.0)
    return eta_z.fillna(0.0), chi_z.fillna(0.0)

# -------------------------------------------------------------
# Geometry weights (baseline, no spin)
# -------------------------------------------------------------
def baseline_geometry_weights(contest_type: str, field_size: int, payout_shape: str):
    N = int(max(1, field_size))
    field_bucket = "small" if N <= 1000 else ("mid" if N <= 10000 else "large")
    # weights: w1 proj, w2 ceil, beta quality, gamma risk, delta value/$, rho spend factor
    if contest_type == "Double-Up":
        return dict(w1=1.00, w2=0.20, beta=0.25, gamma=0.10, delta=0.30, rho=0.995)
    if field_bucket == "small" and payout_shape in ["flat","balanced"]:
        return dict(w1=0.85, w2=0.30, beta=0.30, gamma=0.15, delta=0.25, rho=0.990)
    if field_bucket == "mid" and payout_shape in ["balanced","topheavy"]:
        return dict(w1=0.75, w2=0.45, beta=0.35, gamma=0.18, delta=0.20, rho=0.985)
    if field_bucket == "large" and payout_shape in ["topheavy","ultratop"]:
        return dict(w1=0.65, w2=0.60, beta=0.40, gamma=0.20, delta=0.15, rho=0.975)
    return dict(w1=0.75, w2=0.40, beta=0.35, gamma=0.18, delta=0.20, rho=0.985)

# -------------------------------------------------------------
# BGTO% — salary-aware, geometry-aware (NO ownership)
# -------------------------------------------------------------
def bgto_baseline(df: pd.DataFrame, contest_type: str, field_size: int, payout_shape: str,
                  cap: int = 50000, roster: int = 6):
    # Track NaNs for debug
    impute_cols = [
        "Salary","Proj_Points","RG_Proj","fpts",
        "Ceiling","RG_Ceil","ceil",
        "DG_Win%","win","DG_Top20%","top_20","DG_MakeCut%","make_cut"
    ]
    nan_before = {c: (df[c].isna().sum() if c in df.columns else None) for c in impute_cols}
    df = fill_by_salary_bins(df, cols=impute_cols)
    nan_after  = {c: (df[c].isna().sum() if c in df.columns else None) for c in impute_cols}

    Sal  = df["Salary"].astype(float).clip(lower=0)
    Proj = first_present(df, ["Proj_Points","RG_Proj","fpts"]).clip(lower=0)
    Ceil = first_present(df, ["Ceiling","RG_Ceil","ceil"]).fillna(Proj)

    g    = baseline_geometry_weights(contest_type, field_size, payout_shape)
    ReC, Risk     = coherence_terms(df)
    eta_z, chi_z  = value_terms(df)

    # Utility (no ownership): w1*Proj + w2*Ceil + beta*ReC - gamma*Risk + delta*(eta_z + 0.5*chi_z) - λ*(Salary/1000)
    def U_of_lambda(lam):
        U = (g["w1"]*Proj +
             g["w2"]*Ceil +
             g["beta"]*ReC -
             g["gamma"]*Risk +
             g["delta"]*(eta_z + 0.5*chi_z) -
             lam * (Sal/1000.0))
        return U.clip(lower=0.0)

    target_lineup_spend = g["rho"] * cap
    lam_lo, lam_hi = 0.0, 8.0
    slots, implied_spend = None, None
    for _ in range(14):
        lam = 0.5*(lam_lo + lam_hi)
        U   = U_of_lambda(lam)
        if U.sum() <= 0:
            lam_hi = lam
            continue
        slots = (U / U.sum()) * 600.0                    # 600-slot scale
        implied_spend = float((slots * Sal).sum() / 100) # per-lineup spend
        if implied_spend > target_lineup_spend:
            lam_lo = lam
        else:
            lam_hi = lam

    if slots is None or slots.sum() <= 0:
        U = U_of_lambda(0.0)
        slots = (U / U.sum()) * 600.0 if U.sum() > 0 else pd.Series(600.0/len(df), index=df.index)
        implied_spend = float((slots * Sal).sum() / 100)

    debug = {
        "nan_before": nan_before,
        "nan_after": nan_after,
        "target_spend": target_lineup_spend,
        "implied_spend": implied_spend,
        "rho": g["rho"]
    }
    return slots, debug

# -------------------------------------------------------------
# AGTO% — distort BGTO by ownership spin ONLY (Bubble Theory spin)
# -------------------------------------------------------------
def agto_from_spin(df: pd.DataFrame, BGTO: pd.Series, contest_type: str, field_size: int, payout_shape: str):
    Own = first_present(df, ["Proj_Ownership%","RG_ProjOwn","proj_own"])
    # impute ownership by salary bin if partially missing
    df = fill_by_salary_bins(df, cols=[Own.name] if Own.name in df.columns else [])
    Own = first_present(df, [Own.name]) if Own.name in df.columns else pd.Series((BGTO/6.0).median(), index=df.index)
    Own = Own.fillna((BGTO/6.0).median()).clip(lower=0)

    BGTO_pct = BGTO / 6.0
    spin = (BGTO_pct - Own) / 100.0   # convert p.p. to fractions

    # geometry spin weight α
    N = int(max(1, field_size))
    field_bucket = "small" if N <= 1000 else ("mid" if N <= 10000 else "large")
    if contest_type == "Double-Up":
        alpha = 0.12
    elif field_bucket == "large" and payout_shape in ["topheavy","ultratop"]:
        alpha = 0.55
    elif field_bucket == "mid" and payout_shape in ["balanced","topheavy"]:
        alpha = 0.35
    else:
        alpha = 0.25

    expo = (alpha * spin).clip(-1.5, 1.5)
    AGTO_raw = (BGTO * np.exp(expo)).clip(lower=0.0)
    s = AGTO_raw.sum()
    AGTO = AGTO_raw * (600.0 / s) if s > 0 else BGTO

    debug = {"alpha": alpha, "spin_mean": float(spin.mean()), "spin_std": float(spin.std())}
    return AGTO, debug

# -------------------------------------------------------------
# Minimal sampler (demo)
# -------------------------------------------------------------
def sample_lineups(df: pd.DataFrame, L=50, salary_cap=50000, roster_size=6,
                   use_agto=True, enforce_unique=True):
    weights_col = "AGTO%" if use_agto and "AGTO%" in df.columns else "BGTO%"
    w = df[weights_col].fillna(0.0).values
    if w.sum() <= 0:
        w = np.ones_like(w)
    p = w / w.sum()
    names   = df["Name"].tolist()
    playerid= df["PlayerID"].tolist() if "PlayerID" in df.columns else [""]*len(names)
    salary  = df["Salary"].fillna(0).astype(int).tolist() if "Salary" in df.columns else [0]*len(names)

    lineups, seen = [], set()
    tries, max_tries = 0, L*8000
    while len(lineups) < L and tries < max_tries:
        tries += 1
        if len(names) < roster_size:
            break
        idx = np.random.choice(len(names), size=roster_size, replace=False, p=p)
        c_names = [names[i] for i in idx]
        c_ids   = [playerid[i] for i in idx]
        c_sal   = sum(salary[i] for i in idx)
        if c_sal > salary_cap:
            continue
        key = tuple(sorted(c_ids if any(c_ids) else c_names))
        if enforce_unique and key in seen:
            continue
        seen.add(key)
        lineups.append({"names": c_names, "ids": c_ids, "salary": int(c_sal)})
    return lineups

# =============================================================================
# Guided Flow
# =============================================================================

# Step 1 — Platform
st.header("Step 1: Platform")
c1, c2 = st.columns(2)
with c1:
    if st.button("DraftKings", type="primary"):
        st.session_state["site"] = "DraftKings"
with c2:
    st.button("FanDuel (Coming Soon)", disabled=True)
if "site" not in st.session_state:
    st.stop()
st.success(f"Selected: {st.session_state['site']}")
st.divider()

# Step 2 — Sport
st.header("Step 2: Sport")
cols = st.columns(6)
with cols[0]:
    if st.button("Golf", type="primary"):
        st.session_state["sport"] = "Golf"
with cols[1]: st.button("MLB (Coming Soon)", disabled=True)
with cols[2]: st.button("NBA (Coming Soon)", disabled=True)
with cols[3]: st.button("NFL (Coming Soon)", disabled=True)
with cols[4]: st.button("NHL (Coming Soon)", disabled=True)
with cols[5]: st.button("Soccer (Coming Soon)", disabled=True)
if "sport" not in st.session_state:
    st.stop()
st.success(f"Selected: {st.session_state['sport']}")
st.divider()

# Step 3 — Contest Type
st.header("Step 3: Contest Type")
cc1, cc2 = st.columns(2)
with cc1:
    if st.button("GPP", type=("primary" if st.session_state.get("contest_type")=="GPP" else "secondary")):
        st.session_state["contest_type"] = "GPP"
with cc2:
    if st.button("Double-Up", type=("primary" if st.session_state.get("contest_type")=="Double-Up" else "secondary")):
        st.session_state["contest_type"] = "Double-Up"
if "contest_type" not in st.session_state:
    st.stop()
st.success(f"Selected: {st.session_state['contest_type']}")
st.divider()

# Step 4 — Contest Size
st.header("Step 4: Contest Size")
field_size = st.number_input("Enter field size", min_value=1, value=5000, step=100, format="%d")
presets = [(50, "1–100"), (500, "100–1,000"), (2500, "1,000–5,000"),
           (7500, "5,000–10,000"), (25000, "10,000–50,000"), (75000, "50,000+")]
pcols = st.columns(6)
for i, (val, label) in enumerate(presets):
    if pcols[i].button(label):
        field_size = val
st.session_state["field_size"] = int(max(1, field_size))
st.info(f"Field size set to {st.session_state['field_size']:,}")
st.divider()

# Step 5 — Max Entries per User
st.header("Step 5: Max Entries per User")
max_entries = st.number_input("Enter max entries per user", min_value=1, value=20, step=1, format="%d")
mvals = [1,3,20,50,150]
mcols = st.columns(5)
for i, val in enumerate(mvals):
    if mcols[i].button(str(val)):
        max_entries = val
st.session_state["max_entries"] = int(max(1, max_entries))
st.info(f"Max entries set to {st.session_state['max_entries']}")
st.divider()

# Step 6 — Payout Structure
st.header("Step 6: Payout Structure")
payout_options = {
    "flat": "Flat (50/50 style)",
    "balanced": "Balanced (mid-field GPP)",
    "topheavy": "Top-Heavy (large-field GPP)",
    "ultratop": "Ultra Top-Heavy (Milly style)"
}
cA, cB, cC, cD = st.columns(4)
if st.session_state["contest_type"] == "Double-Up":
    st.session_state["payout_shape"] = "flat"
    with cA: st.button(payout_options["flat"] + " • Selected", disabled=True)
    with cB: st.button(payout_options["balanced"], disabled=True)
    with cC: st.button(payout_options["topheavy"], disabled=True)
    with cD: st.button(payout_options["ultratop"], disabled=True)
else:
    for shape, label in payout_options.items():
        if st.button(label, type=("primary" if st.session_state.get("payout_shape")==shape else "secondary")):
            st.session_state["payout_shape"] = shape

st.text_area("Paste custom payout table (optional)", height=100)
if "payout_shape" not in st.session_state:
    st.stop()
st.success(f"Payout shape: {payout_options[st.session_state['payout_shape']]}")
st.divider()

# Defaults (DK Golf)
salary_cap  = 50000
roster_size = 6
st.caption("**Contest Summary**")
st.code(f"{st.session_state['site']} • {st.session_state['sport']} • {st.session_state['contest_type']} • {st.session_state['field_size']:,} entries • Max {st.session_state['max_entries']}/user • {st.session_state['payout_shape']} • Cap ${salary_cap:,} • Roster {roster_size}")
st.divider()

# Step 7 — Slate & Data (Bubble Theory math here)
st.header("Step 7: Slate & Data")
prep_file = st.file_uploader("Upload GTO Scorecard Prep File (CSV)", type=["csv"])
if not prep_file:
    st.stop()

df = pd.read_csv(prep_file)
df = dedup_and_canonicalize(df)

# BGTO — salary/geometry aware (no ownership)
df["BGTO%"], dbg_bgto = bgto_baseline(
    df,
    contest_type=st.session_state["contest_type"],
    field_size=st.session_state["field_size"],
    payout_shape=st.session_state["payout_shape"],
    cap=salary_cap,
    roster=roster_size
)

# AGTO — Bubble Theory spin
df["AGTO%"], dbg_agto = agto_from_spin(
    df, df["BGTO%"],
    contest_type=st.session_state["contest_type"],
    field_size=st.session_state["field_size"],
    payout_shape=st.session_state["payout_shape"]
)

# Show table
st.caption("Computed **BGTO%** (salary/geometry-aware baseline) and **AGTO%** (ownership-based distortion). Sorted by AGTO%.")
disp_cols = [c for c in df.columns if c not in ["BGTO%","AGTO%"]] + ["BGTO%","AGTO%"]
st.dataframe(df.sort_values("AGTO%", ascending=False)[disp_cols], use_container_width=True)

# Debug panel
with st.expander("🔎 Debug Info: Step 7 Bubble Theory"):
    st.write("**Imputation — NaN counts (before/after):**")
    st.json(dbg_bgto["nan_before"])
    st.json(dbg_bgto["nan_after"])
    st.write(f"Target lineup spend: {dbg_bgto['target_spend']:.1f}")
    st.write(f"Implied lineup spend: {dbg_bgto['implied_spend']:.1f}")
    st.write(f"ρ (spend factor): {dbg_bgto['rho']:.3f}")
    st.write(f"α (spin weight): {dbg_agto['alpha']}")
    st.write(f"Spin mean: {dbg_agto['spin_mean']:.4f}  |  Spin std: {dbg_agto['spin_std']:.4f}")

# -------------------------------------------------------------
# Build Lineups (demo sampler)
# -------------------------------------------------------------
st.divider()
st.header("Build Lineups")
L = st.slider("Number of Lineups", 1, 150, 50)
enforce_unique = st.checkbox("Enforce unique lineups", True)
use_agto = st.checkbox("Use AGTO% weights", True)

if st.button("Run Builder", type="primary"):
    with st.spinner("Building lineups…"):
        lineups = sample_lineups(df, L=L, salary_cap=salary_cap, roster_size=roster_size,
                                 use_agto=use_agto, enforce_unique=enforce_unique)
        if not lineups:
            st.error("No valid lineups under these settings.")
        else:
            st.success(f"Generated {len(lineups)} lineups.")
            df_names = pd.DataFrame([{f"P{i+1}": p for i,p in enumerate(lu["names"])} | {"Salary": lu["salary"]} for lu in lineups])
            st.subheader("Lineups")
            st.dataframe(df_names, use_container_width=True)

            ids_df = pd.DataFrame([lu["ids"] for lu in lineups])
            st.download_button("📥 Download DraftKings CSV (PlayerIDs)",
                               ids_df.to_csv(index=False, header=False).encode("utf-8"),
                               file_name="dfs_matrix_lineups.csv")

            names_flat = [n for lu in lineups for n in lu["names"]]
            expo = pd.Series(names_flat).value_counts(normalize=True)*100
            merge_cols = ["Name","PlayerID","Salary","BGTO%","AGTO%"]
            if "Proj_Ownership%" in df.columns:
                merge_cols.append("Proj_Ownership%")
            expo_df = pd.DataFrame({"Name": expo.index, "Exposure%": expo.values}).merge(
                df[[c for c in merge_cols if c in df.columns]],
                on="Name", how="left"
            ).sort_values("Exposure%", ascending=False)
            st.subheader("Exposure Summary")
            st.dataframe(expo_df, use_container_width=True)
