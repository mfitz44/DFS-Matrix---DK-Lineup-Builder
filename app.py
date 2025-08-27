import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="DFS Matrix — Lineup Builder (Beta)", layout="wide")
st.title("🏗️ DFS Matrix — Lineup Builder (Beta)")

# --------------------------
# Utility helpers
# --------------------------
def norm_name(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9\-\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def probify(s: pd.Series) -> pd.Series:
    """Ensure probabilities (DG odds) are fractional 0..1."""
    if s.max() is np.nan:
        return s
    s = s.astype(float)
    if s.max() <= 1.5:
        return s.clip(0, 1)
    return (s / 100.0).clip(0, 1)

def bucket_field_size(n):
    try:
        n = int(n)
    except:
        return "mid"
    if n <= 1000:
        return "small"
    elif n <= 10000:
        return "mid"
    else:
        return "large"

def dedup_and_canonicalize(df):
    """Keep one canonical Name/PlayerID/Salary; do not drop other unique columns."""
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

# --------------------------
# Geometry weights
# --------------------------
def geometry_weights(contest_type: str, field_size: int, payout_shape: str):
    """Return alpha (spin), beta (real), gamma (imag), delta (value) and rho (spend target factor)."""
    # field bucket
    N = int(max(1, field_size))
    field_bucket = "small" if N <= 1000 else ("mid" if N <= 10000 else "large")

    # defaults
    alpha, beta, gamma, delta = 0.35, 0.40, 0.20, 0.15
    rho = 0.985  # fraction of cap per lineup (geometry spend target)

    if contest_type == "Double-Up":
        alpha, beta, gamma, delta = 0.12, 0.25, 0.10, 0.25
        rho = 0.995
        return alpha, beta, gamma, delta, rho

    # GPP
    if field_bucket == "small" and payout_shape in ["flat", "balanced"]:
        alpha, beta, gamma, delta = 0.25, 0.35, 0.15, 0.20
        rho = 0.990
    elif field_bucket == "mid" and payout_shape in ["balanced", "topheavy"]:
        alpha, beta, gamma, delta = 0.35, 0.40, 0.20, 0.15
        rho = 0.985
    elif field_bucket == "large" and payout_shape in ["topheavy", "ultratop"]:
        alpha, beta, gamma, delta = 0.55, 0.45, 0.25, 0.10
        rho = 0.975
    return alpha, beta, gamma, delta, rho

# --------------------------
# Salary-aware BGTO (bisection on lambda)
# --------------------------
def compute_bgto_salary_aware(df, contest_type: str, field_size: int, payout_shape: str,
                              cap_per_lineup=50000, roster_size=6):
    """BGTO% on 600-slot scale with shadow price λ calibrated to match target spend."""
    # Choose projection & ceiling
    proj_cols = ["Proj_Points", "RG_Proj", "fpts"]
    for c in proj_cols:
        if c in df.columns:
            Proj = df[c].astype(float).clip(lower=0)
            break
    else:
        Proj = pd.Series(1.0, index=df.index)

    ceil_cols = ["Ceiling", "RG_Ceil", "ceil"]
    for c in ceil_cols:
        if c in df.columns:
            Ceil = df[c].astype(float).fillna(Proj)
            break
    else:
        Ceil = Proj

    Salary = df["Salary"].astype(float).clip(lower=0).fillna(0)

    # Contest-type weights for projection vs ceiling
    if contest_type == "Double-Up":
        w1, w2 = 1.0, 0.2
    else:
        w1, w2 = 0.7, 0.6

    # Geometry spend factor rho
    _, _, _, _, rho = geometry_weights(contest_type, field_size, payout_shape)
    target_lineup_spend = rho * cap_per_lineup  # e.g. 49,750 for GPP mid

    # Bisection on λ
    lam_lo, lam_hi = 0.0, 6.0
    bgto = None
    for _ in range(14):
        lam = 0.5 * (lam_lo + lam_hi)
        U = (w1 * Proj + w2 * Ceil - lam * (Salary / 1000.0)).clip(lower=0)
        if U.sum() <= 0:
            lam_hi = lam
            continue
        slots = (U / U.sum()) * 600.0  # 600-slot scale
        # implied per-lineup spend: sum(salary * slots) / 100
        lineup_spend = float((slots * Salary).sum() / 100.0)
        if lineup_spend > target_lineup_spend:
            lam_lo = lam  # spending too much -> increase penalty
            bgto = slots
        else:
            lam_hi = lam
            bgto = slots

    if bgto is None or bgto.sum() <= 0:
        # fallback proportional to projection if bisection fails
        x = Proj.clip(lower=0)
        bgto = (x / x.sum()) * 600.0 if x.sum() > 0 else pd.Series(600.0/len(df), index=df.index)
    return bgto

# --------------------------
# Bubble Theory AGTO (spin + real – imag + value)
# --------------------------
def compute_agto_bubble(df, contest_type: str, field_size: int, payout_shape: str):
    """AGTO% from BGTO% with Bubble Theory tilt terms, then renorm to 600."""
    BGTO = df["BGTO%"].astype(float).fillna(0.0)
    Salary = df["Salary"].astype(float).clip(lower=1.0)

    # ownership (best available)
    own_cols = ["Proj_Ownership%","RG_ProjOwn","proj_own"]
    for c in own_cols:
        if c in df.columns:
            Own = df[c].astype(float).clip(lower=0).fillna(BGTO.mean()/6.0)
            break
    else:
        Own = pd.Series((BGTO/6.0).median(), index=df.index)

    # spin (fractional units)
    BGTO_pct = BGTO / 6.0   # % of lineups
    spin = (BGTO_pct - Own) / 100.0

    # DG odds → real coherence proxy (fractions)
    win_cols = ["DG_Win%","win","Win%"]
    top20_cols = ["DG_Top20%","top_20","Top20%"]
    mc_cols = ["DG_MakeCut%","make_cut","MakeCut%"]

    def pick_prob(cols):
        for c in cols:
            if c in df.columns:
                return probify(df[c])
        return pd.Series(0.0, index=df.index)

    Win = pick_prob(win_cols)
    Top20 = pick_prob(top20_cols)
    MakeCut = pick_prob(mc_cols)

    # real coherence term Re(C): favors quality, penalizes MC risk
    ReC = 2.0 * Win + 0.5 * Top20 - 0.5 * (1.0 - MakeCut)

    # imaginary (oscillation risk) — default to |spin|
    ImC = spin.abs()

    # value per $1k (z-scored)
    proj_cols = ["Proj_Points", "RG_Proj", "fpts"]
    for c in proj_cols:
        if c in df.columns:
            Proj = df[c].astype(float).clip(lower=0)
            break
    else:
        Proj = pd.Series(1.0, index=df.index)

    eta = Proj / (Salary / 1000.0)
    eta_z = (eta - eta.mean()) / (eta.std(ddof=0) if eta.std(ddof=0) > 0 else 1.0)

    # geometry weights
    alpha, beta, gamma, delta, _ = geometry_weights(contest_type, field_size, payout_shape)

    # AGTO tilt
    expo = (alpha * spin + beta * ReC - gamma * ImC + delta * eta_z)
    factor = np.exp(expo.clip(-1.5, 1.5))  # guardrails on exponent
    AGTO_raw = (BGTO * factor).clip(lower=0.0)
    s = AGTO_raw.sum()
    if s <= 0:
        return BGTO
    AGTO = AGTO_raw * (600.0 / s)

    # Optional KL guardrail (light): if too far from BGTO in flat/small contests, shrink a touch
    if contest_type == "Double-Up" or (bucket_field_size(field_size) == "small" and payout_shape in ["flat","balanced"]):
        p = (BGTO / 600.0).clip(1e-12, 1)
        q = (AGTO / 600.0).clip(1e-12, 1)
        kl = float((p * np.log(p / q)).sum())
        if kl > 0.08:
            t = 0.5
            AGTO = (t * AGTO + (1 - t) * BGTO)
            AGTO = AGTO * (600.0 / AGTO.sum())
    return AGTO

# --------------------------
# Sampler (demo)
# --------------------------
def sample_lineups(df, L=50, salary_cap=50000, roster_size=6, use_agto=True, enforce_unique=True):
    weights_col = "AGTO%" if use_agto and "AGTO%" in df.columns else "BGTO%"
    w = df[weights_col].fillna(0.0).values
    if w.sum() <= 0:
        w = np.ones_like(w)
    p = w / w.sum()
    names = df["Name"].tolist()
    playerid = df["PlayerID"].tolist() if "PlayerID" in df.columns else [""] * len(names)
    salary = df["Salary"].fillna(0).astype(int).tolist() if "Salary" in df.columns else [0] * len(names)

    lineups = []
    seen = set()
    tries, max_tries = 0, L * 8000

    while len(lineups) < L and tries < max_tries:
        tries += 1
        if len(names) < roster_size:
            break
        idx = np.random.choice(len(names), size=roster_size, replace=False, p=p)
        c_names = [names[i] for i in idx]
        c_ids = [playerid[i] for i in idx]
        c_sal = sum(salary[i] for i in idx)
        if c_sal > salary_cap:
            continue
        key = tuple(sorted(c_ids if any(c_ids) else c_names))
        if enforce_unique and key in seen:
            continue
        seen.add(key)
        lineups.append({"names": c_names, "ids": c_ids, "salary": int(c_sal)})
    return lineups

# ============================================
# Guided Flow
# ============================================

# Step 1: Platform
st.header("Step 1: Platform")
c1, c2 = st.columns(2)
with c1:
    if st.button("DraftKings", type="primary"):
        st.session_state["site"] = "DraftKings"
with c2:
    st.button("FanDuel (Coming Soon)", disabled=True)

if "site" not in st.session_state: st.stop()
st.success(f"Selected: {st.session_state['site']}")
st.divider()

# Step 2: Sport
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

if "sport" not in st.session_state: st.stop()
st.success(f"Selected: {st.session_state['sport']}")
st.divider()

# Step 3: Contest Type
st.header("Step 3: Contest Type")
cc1, cc2 = st.columns(2)
with cc1:
    if st.button("GPP", type=("primary" if st.session_state.get("contest_type")=="GPP" else "secondary")):
        st.session_state["contest_type"] = "GPP"
with cc2:
    if st.button("Double-Up", type=("primary" if st.session_state.get("contest_type")=="Double-Up" else "secondary")):
        st.session_state["contest_type"] = "Double-Up"

if "contest_type" not in st.session_state: st.stop()
st.success(f"Selected: {st.session_state['contest_type']}")
st.divider()

# Step 4: Contest Size
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

# Step 5: Max Entries per User
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

# Step 6: Payout Structure
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
    with cA: st.button(payout_options["flat"]+" • Selected", disabled=True)
    with cB: st.button(payout_options["balanced"], disabled=True)
    with cC: st.button(payout_options["topheavy"], disabled=True)
    with cD: st.button(payout_options["ultratop"], disabled=True)
else:
    for shape, label in payout_options.items():
        if st.button(label, type=("primary" if st.session_state.get("payout_shape")==shape else "secondary")):
            st.session_state["payout_shape"] = shape

st.text_area("Paste custom payout table (optional)", height=100)
if "payout_shape" not in st.session_state: st.stop()
st.success(f"Payout shape: {payout_options[st.session_state['payout_shape']]}")
st.divider()

# Defaults (DK Golf)
salary_cap = 50000
roster_size = 6
st.caption("**Contest Summary**")
st.code(f"{st.session_state['site']} • {st.session_state['sport']} • {st.session_state['contest_type']} • {st.session_state['field_size']:,} entries • Max {st.session_state['max_entries']}/user • {st.session_state['payout_shape']} • Cap ${salary_cap:,} • Roster {roster_size}")
st.divider()

# Step 7: Slate & Data (Bubble Theory computation here)
st.header("Step 7: Slate & Data")
prep_file = st.file_uploader("Upload GTO Scorecard Prep File", type=["csv"])
if not prep_file:
    st.stop()

df = pd.read_csv(prep_file)
df = dedup_and_canonicalize(df)

# ---- Compute BGTO (salary-aware) ----
df["BGTO%"] = compute_bgto_salary_aware(
    df,
    contest_type=st.session_state["contest_type"],
    field_size=st.session_state["field_size"],
    payout_shape=st.session_state["payout_shape"],
    cap_per_lineup=salary_cap,
    roster_size=roster_size
)

# ---- Compute AGTO (Bubble Theory tilt) ----
df["AGTO%"] = compute_agto_bubble(
    df,
    contest_type=st.session_state["contest_type"],
    field_size=st.session_state["field_size"],
    payout_shape=st.session_state["payout_shape"]
)

st.caption("Computed **BGTO%** (ownership-blind, salary-aware equilibrium) and **AGTO%** (ownership- & geometry-aware Bubble Theory tilt).")

# Display player pool
disp_cols = [c for c in df.columns if c not in ["BGTO%","AGTO%"]] + ["BGTO%","AGTO%"]
st.subheader("Player Pool")
st.dataframe(df.sort_values("AGTO%", ascending=False)[disp_cols], use_container_width=True)

# --------------------------
# Build Lineups (Demo)
# --------------------------
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
            st.download_button("📥 Download DraftKings CSV (PlayerIDs)", ids_df.to_csv(index=False, header=False).encode("utf-8"), file_name="dfs_matrix_lineups.csv")

            names_flat = [n for lu in lineups for n in lu["names"]]
            expo = pd.Series(names_flat).value_counts(normalize=True)*100
            merge_cols = ["Name","PlayerID","Salary","BGTO%","AGTO%"]
            if "Proj_Ownership%":
                merge_cols.append("Proj_Ownership%") if "Proj_Ownership%" in df.columns else None
            expo_df = pd.DataFrame({"Name": expo.index, "Exposure%": expo.values}).merge(
                df[[c for c in merge_cols if c in df.columns]],
                on="Name", how="left"
            ).sort_values("Exposure%", ascending=False)
            st.subheader("Exposure Summary")
            st.dataframe(expo_df, use_container_width=True)
