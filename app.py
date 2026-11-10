import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
from groq_service import (
    GROQ_MODEL,
    GROQ_SOURCE,
    get_api_key,
    lookup_requirements,
)
from uni_data import (
    IELTS_REACH_TOLERANCE,
    SAT_REACH_TOLERANCE,
    STRICTNESS_REACH,
    STRICTNESS_SAFETY,
    fetch_scorecard_universities,
    find_matches,
    sample_universities,
)

st.set_page_config(page_title="University Finder", page_icon="🎓", layout="wide")

METRICS = {
    "ar_score": "Academic reputation",
    "er_score": "Employer reputation",
    "fsr_score": "Faculty/student ratio",
    "cpf_score": "Citations per faculty",
    "ifr_score": "International faculty",
    "isr_score": "International students",
    "irn_score": "Intl research network",
    "ger_score": "Employment outcomes",
    "sus_score": "Sustainability",
}
METRIC_COLS = list(METRICS)

PRESETS = {
    "Balanced": dict(ar_score=5, er_score=5, fsr_score=5, cpf_score=5, ifr_score=5,
                     isr_score=5, irn_score=5, ger_score=5, sus_score=5),
    "I want a good job after graduation": dict(ar_score=5, er_score=10, fsr_score=3, cpf_score=2,
                                              ifr_score=2, isr_score=3, irn_score=2, ger_score=10, sus_score=1),
    "I want to become a researcher": dict(ar_score=9, er_score=2, fsr_score=5, cpf_score=10,
                                          ifr_score=4, isr_score=3, irn_score=8, ger_score=2, sus_score=1),
    "Small classes and international life": dict(ar_score=4, er_score=4, fsr_score=10, cpf_score=2,
                                                 ifr_score=8, isr_score=9, irn_score=5, ger_score=4, sus_score=3),
}

RADAR_LABELS = [METRICS[c] for c in METRIC_COLS]


# ---------------------------------------------------------------
# 1. LOAD AND CLEAN DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    path = "qs_2024.xlsx" if os.path.exists("qs_2024.xlsx") else "data/qs_2024.xlsx"
    cols = ["rank_2024", "rank_2023", "institution", "country_code", "country",
            "size", "focus", "research", "status",
            "ar_score", "ar_rank", "er_score", "er_rank", "fsr_score", "fsr_rank",
            "cpf_score", "cpf_rank", "ifr_score", "ifr_rank", "isr_score", "isr_rank",
            "irn_score", "irn_rank", "ger_score", "ger_rank", "sus_score", "sus_rank",
            "overall_score"]
    df = pd.read_excel(path, header=None, skiprows=4, names=cols)

    df["institution"] = df["institution"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df = df.drop_duplicates(subset="institution")

    # ranks are text ("=76", "1001-1200", "1401+"): keep the label, make a number
    df["rank_label"] = df["rank_2024"].astype(str)
    df["rank_num"] = df["rank_label"].str.extract(r"(\d+)")[0].astype(int)

    # scores use "-" for missing: turn into numbers / NaN
    for c in [c for c in df.columns if c.endswith("_score")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["size_label"] = df["size"].map({"XL": "Extra large", "L": "Large", "M": "Medium", "S": "Small"})
    df["focus_label"] = df["focus"].map({"FC": "Full comprehensive", "CO": "Comprehensive",
                                         "FO": "Focused", "SP": "Specialist"})
    df["research_label"] = df["research"].map({"VH": "Very high", "HI": "High",
                                               "MD": "Medium", "LO": "Low"})

    keep = (["institution", "country", "rank_label", "rank_num",
             "size_label", "focus_label", "research_label"]
            + METRIC_COLS + ["overall_score"])
    return df[keep].reset_index(drop=True)


@st.cache_data
def standardized_profiles(data):
    X = data[METRIC_COLS].copy()
    X = X.fillna(X.median())
    Z = ((X - X.mean()) / X.std()).fillna(0)
    return Z


def find_universities(data, weights, countries, sizes, focus, research, max_rank, min_coverage=0.6):
    pool = data[data["rank_num"] <= max_rank].copy()
    if countries:
        pool = pool[pool["country"].isin(countries)]
    if sizes:
        pool = pool[pool["size_label"].isin(sizes)]
    if focus:
        pool = pool[pool["focus_label"].isin(focus)]
    if research:
        pool = pool[pool["research_label"].isin(research)]
    if pool.empty:
        return pool

    w = pd.Series(weights, dtype=float)
    scores = pool[w.index]
    has = scores.notna()
    num = (scores.fillna(0) * w).sum(axis=1)
    den = (has * w).sum(axis=1)
    pool["coverage"] = den / w.sum()
    pool["match_score"] = (num / den.replace(0, np.nan)).round(1)
    pool = pool[pool["coverage"] >= min_coverage]
    return pool.sort_values("match_score", ascending=False)


def radar(rows, title):
    fig = go.Figure()
    for _, r in rows.iterrows():
        vals = r[METRIC_COLS].fillna(0).astype(float).tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=RADAR_LABELS + RADAR_LABELS[:1],
                                      fill="toself", opacity=0.45, name=r["institution"][:40]))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 100])),
                      legend=dict(orientation="h", y=-0.15))
    return fig


df = load_data()
Z = standardized_profiles(df)

# ---------------------------------------------------------------
# 2. SAVE HELPER (used by the finder and similar tabs)
# ---------------------------------------------------------------
def save_controls(data, key_prefix, source, score_col):
    """Multiselect and "Save selected" button shown under a results table."""
    user = st.session_state.get("user")
    if user is None:
        st.info("Log in from the sidebar to save universities.")
        return

    picked = st.multiselect("Select universities to save", data["institution"].tolist(),
                            key=f"save_pick_{key_prefix}")
    if not st.button("Save selected", key=f"save_btn_{key_prefix}"):
        return
    if not picked:
        st.warning("Pick at least one university first.")
        return

    saved, duplicates, problems = 0, 0, []
    for name in picked:
        row = data[data["institution"] == name].iloc[0]
        status, message = db.save_university(user["id"], name, row["country"],
                                             row["rank_label"], source, row[score_col])
        if status == "saved":
            saved += 1
        elif status == "duplicate":
            duplicates += 1
        else:
            problems.append(message)

    parts = []
    if saved:
        parts.append(f"{saved} saved")
    if duplicates:
        parts.append(f"{duplicates} already saved")
    if parts:
        st.success(", ".join(parts) + ".")
    for problem in problems:
        st.error(problem)


# ---------------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------------
# Accounts. st.session_state["user"] is None or {"id": ..., "username": ...}.
if "user" not in st.session_state:
    st.session_state["user"] = None

db_ready, db_error = db.init_db()
if not db_ready:
    st.sidebar.warning(db_error)

flash = st.session_state.pop("account_flash", None)
if flash:
    st.sidebar.success(flash)

if st.session_state["user"] is None:
    with st.sidebar.expander("Log in or register"):
        login_tab, register_tab = st.tabs(["Log in", "Register"])

        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Log in")
            if login_submitted:
                account, error = db.authenticate(login_username, login_password)
                if account:
                    st.session_state["user"] = account
                    st.rerun()
                else:
                    st.error(error)

        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Username", key="register_username")
                new_password = st.text_input("Password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm password", type="password",
                                                key="register_confirm")
                register_submitted = st.form_submit_button("Create account")
            if register_submitted:
                if new_password != confirm_password:
                    st.error("The two passwords do not match.")
                else:
                    account, error = db.register_user(new_username, new_password)
                    if account:
                        # log in right away, and keep the message for the next run
                        st.session_state["user"] = account
                        st.session_state["account_flash"] = (
                            f"Account created. You are signed in as {account['username']}.")
                        st.rerun()
                    else:
                        st.error(error)
else:
    st.sidebar.success(f"Signed in as **{st.session_state['user']['username']}**")
    if st.sidebar.button("Log out"):
        st.session_state["user"] = None
        st.rerun()

if "w_ar_score" not in st.session_state:
    for k, v in PRESETS["Balanced"].items():
        st.session_state[f"w_{k}"] = v


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[f"w_{k}"] = v


st.sidebar.title("🎓 Your preferences")
st.sidebar.selectbox("Start from a student profile", list(PRESETS), key="preset", on_change=apply_preset)

st.sidebar.subheader("How important is each thing? (0-10)")
WEIGHTS = {}
for col, name in METRICS.items():
    WEIGHTS[col] = st.sidebar.slider(name, 0, 10, key=f"w_{col}")

st.sidebar.subheader("Filters")
countries = st.sidebar.multiselect("Country (empty = all)", sorted(df["country"].unique()))
sizes = st.sidebar.multiselect("University size", ["Small", "Medium", "Large", "Extra large"])
focus = st.sidebar.multiselect("Subject focus", ["Full comprehensive", "Comprehensive", "Focused", "Specialist"])
research = st.sidebar.multiselect("Research intensity", ["Very high", "High", "Medium", "Low"])
max_rank = st.sidebar.slider("Only universities ranked in the top", 50, 1500, 500, step=50)
top_n = st.sidebar.slider("How many results", 5, 30, 10)

# ---------------------------------------------------------------
# 4. HEADER
# ---------------------------------------------------------------
st.title("🎓 University Finder")
st.markdown("**Question:** Given what a student cares about and where they want to study, "
            "which universities are the best match, and which ones are similar to a university they already like?")

tab1, tab_uni, tab_req, tab2, tab_saved, tab3, tab4 = st.tabs([
    "Find my university", "University Finder", "Requirement lookup",
    "Find similar universities", "My saved universities", "Explore the data", "About the data",
])

# ---------------------------------------------------------------
# TAB 1: FINDER
# ---------------------------------------------------------------
with tab1:
    if sum(WEIGHTS.values()) == 0:
        st.warning("All weights are 0. Move at least one slider in the sidebar.")
    else:
        results = find_universities(df, WEIGHTS, countries, sizes, focus, research, max_rank)
        if results.empty:
            st.warning("No universities match these filters. Try a larger rank limit or fewer filters.")
        else:
            top = results.head(top_n)
            c1, c2, c3 = st.columns(3)
            c1.metric("Universities that match your filters", f"{len(results):,}")
            c2.metric("Best match", top.iloc[0]["institution"][:32])
            c3.metric("Countries in your top results", top["country"].nunique())

            left, right = st.columns([3, 2])
            fig = px.bar(top.sort_values("match_score"), x="match_score", y="institution",
                         orientation="h", color="match_score", color_continuous_scale="Greens",
                         hover_data=["country", "rank_label"],
                         title="Best matches for your preferences",
                         labels={"match_score": "Match score (0-100)", "institution": ""})
            fig.update_layout(coloraxis_showscale=False)
            left.plotly_chart(fig, width="stretch")
            left.caption("The match score is the average of the metrics you care about, weighted by your sliders.")

            if len(top) >= 3:
                right.plotly_chart(radar(top.head(3), "Your top 3: score on each metric"), width="stretch")
                right.caption("A larger shape means a higher score. Look for the metrics you care about most.")

            show = top[["institution", "country", "rank_label", "size_label",
                        "focus_label", "match_score"]].rename(columns={
                "institution": "University", "country": "Country", "rank_label": "QS rank",
                "size_label": "Size", "focus_label": "Focus", "match_score": "Match score"})
            st.dataframe(show, width="stretch", hide_index=True)

            save_controls(top, "finder", "Finder", "match_score")

            st.subheader("Best countries for your preferences")
            by_country = (results.head(50).groupby("country")
                          .agg(universities=("institution", "count"), avg_match=("match_score", "mean"))
                          .sort_values("universities", ascending=False).head(10).reset_index())
            fig2 = px.bar(by_country, x="country", y="universities", color="avg_match",
                          color_continuous_scale="Blues", title="Countries among your 50 best matches",
                          labels={"universities": "Universities in top 50", "avg_match": "Average match"})
            st.plotly_chart(fig2, width="stretch")
            st.caption("Shows which countries have the most universities that fit your preferences.")

# ---------------------------------------------------------------
# TAB 2: UNIVERSITY FINDER (ADMISSIONS)
# ---------------------------------------------------------------
with tab_uni:
    st.markdown("Check your **SAT and IELTS scores** against admission requirements and keep only "
                "universities inside your **annual tuition budget**. The first tab matches you on "
                "academic reputation; this tab answers where you can realistically apply.")

    with st.expander("Data source"):
        use_live = st.checkbox("Use live data from the US College Scorecard API (US only)",
                               key="uf_use_live")
        api_key = st.text_input(
            "College Scorecard API key",
            value=os.environ.get("COLLEGE_SCORECARD_API_KEY", "DEMO_KEY"),
            type="password",
            disabled=not use_live,
            help="Request a free key at https://collegescorecard.ed.gov/data/api/. "
                 "DEMO_KEY allows a few requests per hour.",
        )
        st.caption("The College Scorecard publishes SAT scores (25th percentile where available, "
                   "otherwise the average of admitted students), admission rates and out-of-state "
                   "tuition for US institutions, but no IELTS requirements, so live rows are "
                   "matched on SAT only and universities that report no SAT score are skipped. "
                   "The built-in sample covers several countries with illustrative requirements.")

    if use_live:
        uni_df = fetch_scorecard_universities(api_key)
        if uni_df.empty:
            st.warning("Could not load live data (check the API key and your connection). "
                       "Showing the built-in sample dataset instead.")
            uni_df = sample_universities()
        else:
            st.caption(f"Live source: {len(uni_df):,} SAT-reporting US universities from the "
                       "College Scorecard.")
    else:
        uni_df = sample_universities()
        st.caption(f"Built-in sample: {len(uni_df)} universities in "
                   f"{uni_df['country'].nunique()} countries.")

    s1, s2, s3 = st.columns(3)
    sat_score = s1.slider("SAT overall score", 400, 1600, 1300, step=10, key="uf_sat")
    ielts_score = s2.slider("IELTS overall score", 0.0, 9.0, 6.5, step=0.5, key="uf_ielts")
    max_budget = s3.slider("Maximum annual tuition (USD)", 0, 80000, 40000, step=1000,
                           format="$%d", key="uf_budget")

    f1, f2 = st.columns([3, 2])
    target_countries = f1.multiselect("Target country (empty = all)",
                                      sorted(uni_df["country"].dropna().unique()),
                                      key="uf_countries")
    strictness = f2.selectbox(
        "Match strictness", [STRICTNESS_SAFETY, STRICTNESS_REACH], key="uf_strictness",
        help=f"Safety / Target: your score meets or beats the requirement. "
             f"Reach: the requirement is up to {SAT_REACH_TOLERANCE} SAT points or "
             f"{IELTS_REACH_TOLERANCE} IELTS bands above your score.",
    )

    matches = find_matches(uni_df, sat_score, ielts_score, max_budget,
                           target_countries, strictness)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total matches found", f"{len(matches):,}")
    avg_tuition = matches["annual_tuition_usd"].mean()
    m2.metric("Average tuition of matches",
              f"${avg_tuition:,.0f}" if pd.notna(avg_tuition) else "—")
    best_acceptance = matches["acceptance_rate"].max()
    m3.metric("Highest acceptance rate",
              f"{best_acceptance * 100:.0f}%" if pd.notna(best_acceptance) else "—")

    if matches.empty:
        st.info("**No universities match these criteria.** Your scores or budget are tighter "
                "than every requirement in the current dataset.")
        pool = uni_df if not target_countries else uni_df[uni_df["country"].isin(target_countries)]
        cheapest = pool["annual_tuition_usd"].min()
        lowest_sat = pool["min_sat_overall"].min()
        lowest_ielts = pool["min_ielts_overall"].min()
        tips = []
        if pd.notna(cheapest) and cheapest > max_budget:
            tips.append(f"Raise the budget to at least **${cheapest:,.0f}**, the cheapest option "
                        "in this pool.")
        if pd.notna(lowest_sat) and lowest_sat > sat_score:
            tips.append(f"The lowest SAT requirement here is **{lowest_sat:.0f}** "
                        f"(your score: {sat_score}).")
        if pd.notna(lowest_ielts) and lowest_ielts > ielts_score:
            tips.append(f"The lowest IELTS requirement here is **{lowest_ielts:.1f}** "
                        f"(your score: {ielts_score:.1f}).")
        if strictness == STRICTNESS_SAFETY:
            tips.append(f"Switch Match strictness to **Reach** to include universities up to "
                        f"{SAT_REACH_TOLERANCE} SAT points / {IELTS_REACH_TOLERANCE} IELTS bands "
                        "above your scores.")
        else:
            tips.append("Switch Match strictness to **Safety / Target** to see universities you "
                        "already qualify for.")
        if target_countries:
            tips.append("Remove the **Target country** filter to widen the pool.")
        st.markdown("**Try this:**")
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        matches["SAT delta"] = matches["sat_margin"].map(
            lambda v: f"{v:+.0f}" if pd.notna(v) else "—")
        matches["IELTS delta"] = matches["ielts_margin"].map(
            lambda v: f"{v:+.1f}" if pd.notna(v) else "—")

        st.dataframe(
            matches[["name", "country", "min_sat_overall", "SAT delta", "min_ielts_overall",
                     "IELTS delta", "annual_tuition_usd", "acceptance_rate_pct",
                     "website_url"]],
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("University", width="medium"),
                "country": "Country",
                "min_sat_overall": st.column_config.NumberColumn(
                    "SAT requirement", format="%d",
                    help="Required SAT overall score (25th percentile, or the average of admitted "
                         "students where the percentile is unavailable, for live data)."),
                "SAT delta": st.column_config.TextColumn(
                    "SAT delta", width="small",
                    help="Your SAT score minus the requirement (positive = you clear it)."),
                "min_ielts_overall": st.column_config.NumberColumn(
                    "IELTS requirement", format="%.1f"),
                "IELTS delta": st.column_config.TextColumn(
                    "IELTS delta", width="small",
                    help="Your IELTS score minus the requirement (positive = you clear it)."),
                "annual_tuition_usd": st.column_config.NumberColumn(
                    "Annual tuition", format="$%d", help="USD per year."),
                "acceptance_rate_pct": st.column_config.NumberColumn(
                    "Acceptance rate", format="%.0f%%"),
                "website_url": st.column_config.LinkColumn("Website", display_text="Visit site"),
            },
        )
        st.caption("Sorted by annual tuition, cheapest first. Click a column header to re-sort. "
                   "A dash means the requirement is not published.")

# ---------------------------------------------------------------
# TAB 3: REQUIREMENT LOOKUP (GROQ)
# ---------------------------------------------------------------
with tab_req:
    st.markdown("Look up a university's **SAT and IELTS requirements, annual tuition and "
                "acceptance rate**. Groq extracts the numbers from the model's knowledge; "
                "without an API key the app searches its bundled offline database instead.")

    has_api_key = bool(get_api_key())
    with st.form("requirement_lookup"):
        name_col, submit_col = st.columns([4, 1], vertical_alignment="bottom")
        query = name_col.text_input(
            "University name",
            placeholder="e.g. University of Toronto",
            key="req_query_input",
        )
        submitted = submit_col.form_submit_button("Extract requirements", type="primary")

    if submitted and query.strip():
        st.session_state["req_active_query"] = query.strip()
    active_query = st.session_state.get("req_active_query", "")

    if has_api_key:
        st.caption(f"Using Groq `{GROQ_MODEL}` in JSON mode; answers are cached for 24 hours.")
    else:
        st.caption("No `GROQ_API_KEY` found, so the bundled offline database is used. Add the "
                   "key to `.streamlit/secrets.toml` or your environment to extract any "
                   "university.")

    if not active_query:
        st.info("Search for a university above, or try **MIT**, **University of Oxford**, "
                "**University of Toronto** or **ETH Zurich**.")
    else:
        record, source, error_type = lookup_requirements(active_query)

        if error_type:
            st.warning(f"Groq could not answer ({error_type}), so the bundled data is shown "
                       "instead.")

        if record is None:
            st.warning(f"No bundled data for **{active_query}**.")
            if has_api_key:
                st.caption("Check the spelling or try the full official name.")
            else:
                st.caption("Add a Groq API key to extract requirements for universities "
                           "outside the offline database.")
        else:
            source_label = (f"Groq · {GROQ_MODEL}" if source == GROQ_SOURCE
                            else "Bundled offline database")
            with st.container(border=True):
                st.subheader(record.get("university") or active_query)
                st.caption(" · ".join(
                    part for part in [record.get("country"), f"Source: {source_label}"]
                    if part))

                sat_min = record.get("sat_min")
                sat_rec = record.get("sat_recommended")
                ielts_min = record.get("ielts_min_overall")
                tuition = record.get("annual_tuition_usd")
                acceptance = record.get("acceptance_rate")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("SAT minimum", f"{sat_min:.0f}" if sat_min is not None else "—")
                m2.metric("SAT recommended",
                          f"{sat_rec:.0f}" if sat_rec is not None else "—")
                m3.metric("IELTS minimum",
                          f"{ielts_min:.1f}" if ielts_min is not None else "—")
                m4.metric("Annual tuition",
                          f"${tuition:,.0f}" if tuition is not None else "—")
                m5.metric("Acceptance rate",
                          f"{acceptance * 100:.0f}%" if acceptance is not None else "—")

                subscores = [
                    ("Listening", record.get("ielts_min_listening")),
                    ("Reading", record.get("ielts_min_reading")),
                    ("Writing", record.get("ielts_min_writing")),
                    ("Speaking", record.get("ielts_min_speaking")),
                ]
                published = [f"{label} {value:.1f}" for label, value in subscores
                             if value is not None]
                if published:
                    st.markdown("**IELTS subscores:** " + " · ".join(published))

                missing = [label for label, key in [
                    ("SAT minimum", "sat_min"),
                    ("SAT recommended", "sat_recommended"),
                    ("IELTS minimum", "ielts_min_overall"),
                    ("IELTS subscores", "ielts_min_listening"),
                    ("annual tuition", "annual_tuition_usd"),
                    ("acceptance rate", "acceptance_rate"),
                ] if record.get(key) is None]
                if missing:
                    st.caption("Not published in this answer: " + ", ".join(missing) + ".")

            st.caption("These figures are AI-extracted or bundled estimates. Confirm them on "
                       "the university's admissions page before applying.")

# ---------------------------------------------------------------
# TAB 4: SIMILAR
# ---------------------------------------------------------------
with tab2:
    st.markdown("Pick a university you like. The app compares the scores of all 9 metrics "
                "and finds the universities with the most similar profile (cosine similarity).")
    names = df.sort_values("rank_num")["institution"].tolist()
    default = names.index("Seoul National University") if "Seoul National University" in names else 0
    liked = st.selectbox("University you like", names, index=default)
    a, b = st.columns(2)
    sim_countries = a.multiselect("Only show universities in these countries (empty = anywhere)",
                                  sorted(df["country"].unique()), key="sim_countries")
    sim_n = b.slider("How many similar universities", 3, 15, 8)

    idx = df.index[df["institution"] == liked][0]
    v = Z.loc[idx].values
    sims = Z.values @ v / (np.linalg.norm(Z.values, axis=1) * np.linalg.norm(v) + 1e-9)
    out = df.copy()
    out["similarity"] = sims.round(3)
    out = out.drop(index=idx)
    if sim_countries:
        out = out[out["country"].isin(sim_countries)]
    out = out.sort_values("similarity", ascending=False).head(sim_n)

    row = df.loc[idx]
    st.info(f"**{row['institution']}**  |  {row['country']}  |  QS rank {row['rank_label']}")

    if out.empty:
        st.warning("No universities found with these filters.")
    else:
        l, r = st.columns(2)
        figs = px.bar(out.sort_values("similarity"), x="similarity", y="institution", orientation="h",
                      hover_data=["country", "rank_label"], title="Most similar universities",
                      labels={"similarity": "Similarity (1 = identical profile)", "institution": ""},
                      color="similarity", color_continuous_scale="Purples")
        figs.update_layout(coloraxis_showscale=False)
        l.plotly_chart(figs, width="stretch")
        l.caption("Similarity compares the shape of the scores, not the level, so a lower-ranked "
                  "university can still be a close match.")
        compare = pd.concat([df.loc[[idx]], out.head(2)])
        r.plotly_chart(radar(compare, "Your university vs its 2 closest matches"), width="stretch")
        r.caption("Shapes that overlap closely show universities with a similar profile.")
        st.dataframe(out[["institution", "country", "rank_label", "size_label", "similarity"]].rename(
            columns={"institution": "University", "country": "Country", "rank_label": "QS rank",
                     "size_label": "Size", "similarity": "Similarity"}),
            width="stretch", hide_index=True)

        save_controls(out, "similar", f"Similar to {liked}", "similarity")

# ---------------------------------------------------------------
# TAB 5: MY SAVED UNIVERSITIES
# ---------------------------------------------------------------
with tab_saved:
    st.markdown("Universities you saved from the finder and the similarity search. "
                "Every account has its own list.")

    saved_user = st.session_state.get("user")
    saved_flash = st.session_state.pop("saved_flash", None)
    if saved_flash:
        st.success(saved_flash)

    if saved_user is None:
        st.info("Log in or register in the sidebar to keep your own list of universities.")
    else:
        saved_rows, saved_error = db.list_saved(saved_user["id"])
        if saved_error:
            st.error(saved_error)
        elif not saved_rows:
            st.info("Nothing saved yet. Open **Find my university** or **Find similar "
                    "universities**, select a few names and press **Save selected**.")
        else:
            saved = pd.DataFrame(saved_rows)
            # the ranking table has the fresh metric scores, the saved row has the rest
            details = df.set_index("institution")
            known = saved["institution"].isin(details.index)
            saved["qs_rank"] = saved["rank_label"].fillna("")
            saved.loc[known, "qs_rank"] = (saved.loc[known, "institution"]
                                           .map(details["rank_label"]))

            table = pd.DataFrame({
                "University": saved["institution"],
                "Country": saved["country"].fillna(""),
                "QS rank": saved["qs_rank"],
                "Source": saved["source"].fillna(""),
                "Score": saved["score"],
                "Saved on": saved["saved_at"],
                "Note": saved["note"].fillna(""),
            })
            st.dataframe(table, width="stretch", hide_index=True, column_config={
                "Score": st.column_config.NumberColumn(
                    format="%.1f", help="Match score, or similarity for similar universities."),
            })
            st.caption(f"{len(table)} saved universities, newest first.")

            st.subheader("Edit a note or remove a university")
            chosen = st.selectbox("Saved university", saved["institution"].tolist(),
                                  key="saved_pick")
            current_note = saved.loc[saved["institution"] == chosen, "note"].iloc[0]
            new_note = st.text_area("Note", value=current_note or "", key=f"note_{chosen}",
                                    placeholder="e.g. Ask about scholarships, deadline in January")
            edit_col, remove_col = st.columns(2)
            if edit_col.button("Save note", key=f"note_save_{chosen}"):
                ok, message = db.update_note(saved_user["id"], chosen, new_note)
                if ok:
                    st.session_state["saved_flash"] = message
                    st.rerun()
                else:
                    st.error(message)
            if remove_col.button("Remove university", key=f"remove_{chosen}"):
                ok, message = db.remove_saved(saved_user["id"], chosen)
                if ok:
                    st.session_state["saved_flash"] = message
                    st.rerun()
                else:
                    st.error(message)

            st.subheader("Compare saved universities")
            comparable = saved.loc[known, "institution"].tolist()
            picked_saved = st.multiselect("Pick up to 5 saved universities to compare",
                                          comparable, max_selections=5, key="saved_radar")
            if len(picked_saved) >= 2:
                st.plotly_chart(radar(details.loc[picked_saved].reset_index(),
                                      "Your saved universities: score on each metric"),
                                width="stretch")
                st.caption("A larger shape means higher scores. Compare the metrics you care about.")
            elif picked_saved:
                st.caption("Pick at least two universities to compare their profiles.")

            st.download_button("Download my list as CSV",
                               table.to_csv(index=False).encode("utf-8"),
                               file_name="my_saved_universities.csv", mime="text/csv")

# ---------------------------------------------------------------
# TAB 6: EXPLORE
# ---------------------------------------------------------------
with tab3:
    st.subheader("Countries with the most universities in the top N")
    n_top = st.slider("Top N universities", 50, 1000, 200, step=50, key="n_top")
    counts = df[df["rank_num"] <= n_top]["country"].value_counts().head(15).reset_index()
    counts.columns = ["country", "universities"]
    fig = px.bar(counts.sort_values("universities"), x="universities", y="country", orientation="h",
                 title=f"Top 15 countries in the QS top {n_top}", labels={"country": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption("A few countries (the United States, the United Kingdom, China) hold a large part of the top places.")

    l, r = st.columns(2)
    sub = df.dropna(subset=["ar_score", "er_score", "size_label"])
    fig = px.scatter(sub, x="ar_score", y="er_score", color="size_label", hover_name="institution",
                     hover_data=["country", "rank_label"], opacity=0.65,
                     title="Academic vs employer reputation",
                     labels={"ar_score": "Academic reputation", "er_score": "Employer reputation",
                             "size_label": "Size"})
    l.plotly_chart(fig, width="stretch")
    l.caption("Each dot is a university. Hover to see its name. The two reputations move together.")

    ranked = df.dropna(subset=["overall_score", "size_label"])
    fig = px.box(ranked, x="size_label", y="overall_score",
                 category_orders={"size_label": ["Small", "Medium", "Large", "Extra large"]},
                 title="Overall score by university size (ranked universities only)",
                 labels={"size_label": "Size", "overall_score": "Overall score"})
    r.plotly_chart(fig, width="stretch")
    r.caption("Size alone does not decide the score: small and extra large universities both reach the top.")

    corr = df[METRIC_COLS + ["overall_score"]].corr()
    names_c = [METRICS.get(c, "Overall score") for c in corr.columns]
    fig = px.imshow(corr.values, x=names_c, y=names_c, text_auto=".1f", zmin=-1, zmax=1,
                    color_continuous_scale="RdYlGn", title="Which scores move together?", aspect="auto")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green means two scores rise together. Red means one is high when the other is low.")

    st.subheader("Does a better rank mean smaller classes or more international students?")
    metric_choice = st.radio("Metric", ["Faculty/student ratio", "International students"], horizontal=True)
    mcol = "fsr_score" if metric_choice.startswith("Faculty") else "isr_score"
    sub = df[df["rank_num"] <= 500].dropna(subset=[mcol])
    fig = px.scatter(sub, x="rank_num", y=mcol, hover_name="institution", opacity=0.6,
                     labels={"rank_num": "QS rank (1 = best)", mcol: metric_choice + " score"},
                     title=f"QS rank vs {metric_choice.lower()} score")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Correlation between rank number and this score: {sub['rank_num'].corr(sub[mcol]):.2f}. "
               "A negative number means better-ranked universities score higher.")

# ---------------------------------------------------------------
# TAB 7: ABOUT
# ---------------------------------------------------------------
with tab4:
    st.markdown("""
### About the data
- **Source:** QS World University Rankings 2024, official results table (Excel), Quacquarelli Symonds.
- **Link:** https://www.topuniversities.com/world-university-rankings
- **Size:** 1,498 universities in 104 countries, with 9 indicator scores and an overall score.

### The 9 scores (0-100)
Academic reputation, employer reputation, faculty/student ratio, citations per faculty,
international faculty, international students, international research network,
employment outcomes, and sustainability.

### Cleaning steps
1. Skipped the 4 header rows in the Excel file and gave clear column names.
2. Checked for duplicate university names (none found) and removed extra spaces.
3. Turned text ranks (`=76`, `1001-1200`, `1401+`) into a number, and kept the original text for display.
4. Turned `-` into missing values so scores are real numbers.
5. Replaced codes (XL, FC, VH) with readable labels.
6. Kept only the useful columns.

### How the match score works
Match score = sum(weight x score) / sum(weights of the metrics the university has).
A university needs scores for at least 60% of your total weight to appear, so it is not ranked on very little data.

### Limits
- Only about 600 universities have an overall score, so the finder uses the indicator scores.
- QS measures reputation and research more than teaching quality.
- Mongolian universities are not in this table.

### Accounts and privacy
- Accounts and saved universities are for this class project only: they are not a real service.
- Passwords are stored as salted PBKDF2 hashes, never as plain text, but please do not reuse a real password.
- Saved lists live in a small SQLite file (`data/users.db`) that is not committed to GitHub.
- On the free hosting the app can restart or lose its disk, so the saved data may be reset at any time.
""")
    st.dataframe(df.head(50), width="stretch")
