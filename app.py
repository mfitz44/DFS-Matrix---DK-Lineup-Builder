import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="DFS Matrix — New Lineup Builder (Beta)", layout="wide")
st.title("🏗️ DFS Matrix — Lineup Builder (Beta)")

# --------------------------
# Helpers
# --------------------------
def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\b(jr|sr|iii|ii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9\-\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

def compute_bgto(df, proj_col="Proj_Points"):
    x = df[proj_col].fillna(df[proj_col].median()) if proj_col in df.columns else pd.Series(1.0, index=df.index)
    x = x.clip(lower=0)
    if x.sum() <= 0:
        x = pd.Series(1.0, index=df.index)
    bgto = (x / x.sum()) * 600.0
    return bgto

def compute_agto(df, bgto_col="BGTO%", own_col="Proj_Ownership%", contest_type="GPP", field_bucket="mid", payout_shape="balanced"):
    base = 0.35
    if contest_type == "Double-Up":
        base = 0.15
    else:
        if payout_shape in ["topheavy", "ultratop"]:
            base += 0.25
        if field_bucket == "large":
            base += 0.20
        elif field_bucket == "small":
            base -= 0.10
    base = float(np.clip(base, 0.05, 0.85))

    bgto = df[bgto_col].fillna(0.0)
    bgto_pct = bgto / 6.0

    if own_col in df.columns:
        proj_own = df[own_col].fillna(bgto_pct.median())
    else:
        proj_own = pd.Series(bgto_pct.median(), index=df.index)

    d = (bgto_pct - proj_own) / 100.0
    factor = 1.0 - base * d
    factor = factor.clip(lower=0.1, upper=2.0)

    agto_raw = bgto * factor
    s = agto_raw.sum()
    if s > 0:
        agto = agto_raw * (600.0 / s)
    else:
        agto = bgto
    agto = agto.clip(lower=0.0)
    return agto

def dedup_and_canonicalize(df):
    # canonicalize Name, PlayerID, Salary
    cols = df.columns.tolist()
    name_cols = [c for c in cols if c.lower() in ["name", "player", "player name", "player_name"]]
    if name_cols:
        keep = name_cols[0]
        if keep != "Name":
            df.rename(columns={keep: "Name"}, inplace=True)
        for c in name_cols:
            if c != "Name":
                df.drop(columns=[c], inplace=True, errors="ignore")
    id_cols = [c for c in df.columns if c.lower() in ["playerid", "player id", "dk_id", "dkid", "id"]]
    if id_cols:
        keep = id_cols[0]
        if keep != "PlayerID":
            df.rename(columns={keep: "PlayerID"}, inplace=True)
        for c in id_cols:
            if c != "PlayerID":
                df.drop(columns=[c], inplace=True, errors="ignore")
    sal_cols = [c for c in df.columns if c.lower() in ["salary", "dk salary"]]
    if sal_cols:
        keep = sal_cols[0]
        if keep != "Salary":
            df.rename(columns={keep: "Salary"}, inplace=True)
        for c in sal_cols:
            if c != "Salary":
                df.drop(columns=[c], inplace=True, errors="ignore")
    return df

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
    tries = 0
    max_tries = L * 8000
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

# --------------------------
# Step 1 — Platform
# --------------------------
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

# --------------------------
# Step 2 — Sport
# --------------------------
st.header("Step 2: Sport")
cols = st.columns(6)
with cols[0]:
    if st.button("Golf", type="primary"):
        st.session_state["sport"] = "Golf"
with cols[1]:
    st.button("MLB (Coming Soon)", disabled=True)

if "sport" not in st.session_state:
    st.stop()
st.success(f"Selected: {st.session_state['sport']}")
st.divider()

# --------------------------
# Step 3 — Contest Type
# --------------------------
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

# --------------------------
# Step 4 — Contest Size
# --------------------------
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

# --------------------------
# Step 5 — Max Entries
# --------------------------
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

# --------------------------
# Step 6 — Payout Structure
# --------------------------
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

if "payout_shape" not in st.session_state:
    st.stop()
st.success(f"Payout shape: {payout_options[st.session_state['payout_shape']]}")
st.divider()

# Defaults
st.session_state["salary_cap"] = 50000
st.session_state["roster_size"] = 6
st.caption("**Contest Summary**")
st.code(f"{st.session_state['site']} • {st.session_state['sport']} • {st.session_state['contest_type']} • {st.session_state['field_size']:,} entries • Max {st.session_state['max_entries']}/user • {st.session_state['payout_shape']} • Cap $50,000 • Roster 6")
st.divider()

# --------------------------
# Step 7 — Upload Prep File
# --------------------------
st.header("Step 7: Slate & Data")
prep_file = st.file_uploader("Upload GTO Scorecard Prep File", type=["csv"])
if not prep_file:
    st.stop()

df = pd.read_csv(prep_file)
df = dedup_and_canonicalize(df)
df["BGTO%"] = compute_bgto(df, proj_col="Proj_Points")
field_bucket = bucket_field_size(st.session_state["field_size"])
df["AGTO%"] = compute_agto(df, bgto_col="BGTO%", own_col="Proj_Ownership%",
                           contest_type=st.session_state["contest_type"], field_bucket=field_bucket,
                           payout_shape=st.session_state["payout_shape"])

st.subheader("Player Pool")
st.caption("All prep columns preserved; BGTO% and AGTO% appended. Sorted by AGTO%.")
disp_cols = [c for c in df.columns if c not in ["BGTO%","AGTO%"]] + ["BGTO%","AGTO%"]
st.dataframe(df.sort_values("AGTO%", ascending=False)[disp_cols], use_container_width=True)

st.divider()
st.header("Build Lineups")
L = st.slider("Number of Lineups", 1, 150, 50)
enforce_unique = st.checkbox("Enforce unique lineups", True)
use_agto = st.checkbox("Use AGTO% weights", True)

if st.button("Run Builder", type="primary"):
    with st.spinner("Building lineups…"):
        lineups = sample_lineups(df, L=L, salary_cap=st.session_state["salary_cap"], roster_size=st.session_state["roster_size"],
                                 use_agto=use_agto, enforce_unique=enforce_unique)
        if not lineups:
            st.error("No valid lineups under these settings.")
        else:
            st.success(f"Generated {len(lineups)} lineups.")
            df_names = pd.DataFrame([{f"P{i+1}": p for i,p in enumerate(lu["names"])} | {"Salary": lu["salary"]} for lu in lineups])
            st.dataframe(df_names, use_container_width=True)
            ids_df = pd.DataFrame([lu["ids"] for lu in lineups])
            st.download_button("📥 Download DraftKings CSV (PlayerIDs)", ids_df.to_csv(index=False, header=False).encode("utf-8"), file_name="dfs_matrix_lineups.csv")

            names_flat = [n for lu in lineups for n in lu["names"]]
            expo = pd.Series(names_flat).value_counts(normalize=True)*100
            expo_df = pd.DataFrame({"Name": expo.index, "Exposure%": expo.values}).merge(
                df[["Name","PlayerID","Salary","BGTO%","AGTO%","Proj_Ownership%"]] if "Proj_Ownership%" in df.columns else df[["Name","PlayerID","Salary","BGTO%","AGTO%"]],
                on="Name", how="left"
            ).sort_values("Exposure%", ascending=False)
            st.subheader("Exposure Summary")
            st.dataframe(expo_df, use_container_width=True)
