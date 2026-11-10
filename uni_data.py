"""Admissions data for the University Finder tab.

Two sources are supported:

1. A built-in sample of universities in several countries. It is always available and
   uses approximate, illustrative requirements.
2. Optional live data from the US College Scorecard API, which covers US institutions
   only and does not publish IELTS requirements.
"""

import pandas as pd
import requests
import streamlit as st

DATA_COLUMNS = [
    "name",
    "country",
    "min_sat_overall",
    "min_sat_math",
    "min_sat_reading",
    "min_ielts_overall",
    "annual_tuition_usd",
    "acceptance_rate",
    "website_url",
]

# Illustrative sample data: roughly the 25th percentile SAT of admitted students,
# published IELTS minima and annual tuition for international students.
SAMPLE_UNIVERSITIES = [
    # United States
    dict(name="Massachusetts Institute of Technology", country="United States",
         min_sat_overall=1510, min_sat_math=780, min_sat_reading=740,
         min_ielts_overall=7.5, annual_tuition_usd=61990, acceptance_rate=0.04,
         website_url="https://www.mit.edu"),
    dict(name="Harvard University", country="United States",
         min_sat_overall=1520, min_sat_math=790, min_sat_reading=750,
         min_ielts_overall=7.5, annual_tuition_usd=59076, acceptance_rate=0.03,
         website_url="https://www.harvard.edu"),
    dict(name="Stanford University", country="United States",
         min_sat_overall=1520, min_sat_math=790, min_sat_reading=740,
         min_ielts_overall=7.0, annual_tuition_usd=65127, acceptance_rate=0.04,
         website_url="https://www.stanford.edu"),
    dict(name="California Institute of Technology", country="United States",
         min_sat_overall=1540, min_sat_math=790, min_sat_reading=750,
         min_ielts_overall=7.0, annual_tuition_usd=63255, acceptance_rate=0.03,
         website_url="https://www.caltech.edu"),
    dict(name="Columbia University", country="United States",
         min_sat_overall=1500, min_sat_math=770, min_sat_reading=740,
         min_ielts_overall=7.0, annual_tuition_usd=68400, acceptance_rate=0.04,
         website_url="https://www.columbia.edu"),
    dict(name="New York University", country="United States",
         min_sat_overall=1450, min_sat_math=740, min_sat_reading=710,
         min_ielts_overall=7.0, annual_tuition_usd=60438, acceptance_rate=0.12,
         website_url="https://www.nyu.edu"),
    dict(name="University of California, Berkeley", country="United States",
         min_sat_overall=1450, min_sat_math=750, min_sat_reading=710,
         min_ielts_overall=6.5, annual_tuition_usd=49350, acceptance_rate=0.11,
         website_url="https://www.berkeley.edu"),
    dict(name="University of Michigan", country="United States",
         min_sat_overall=1430, min_sat_math=740, min_sat_reading=700,
         min_ielts_overall=6.5, annual_tuition_usd=60500, acceptance_rate=0.18,
         website_url="https://umich.edu"),
    dict(name="University of Texas at Austin", country="United States",
         min_sat_overall=1350, min_sat_math=700, min_sat_reading=670,
         min_ielts_overall=6.5, annual_tuition_usd=42700, acceptance_rate=0.29,
         website_url="https://www.utexas.edu"),
    dict(name="University of Washington", country="United States",
         min_sat_overall=1400, min_sat_math=720, min_sat_reading=690,
         min_ielts_overall=6.5, annual_tuition_usd=43200, acceptance_rate=0.48,
         website_url="https://www.washington.edu"),
    dict(name="Purdue University", country="United States",
         min_sat_overall=1330, min_sat_math=690, min_sat_reading=660,
         min_ielts_overall=6.5, annual_tuition_usd=28794, acceptance_rate=0.53,
         website_url="https://www.purdue.edu"),
    dict(name="Arizona State University", country="United States",
         min_sat_overall=1240, min_sat_math=620, min_sat_reading=610,
         min_ielts_overall=6.0, annual_tuition_usd=33000, acceptance_rate=0.88,
         website_url="https://www.asu.edu"),
    # United Kingdom
    dict(name="University of Oxford", country="United Kingdom",
         min_sat_overall=1500, min_sat_math=780, min_sat_reading=740,
         min_ielts_overall=7.5, annual_tuition_usd=39000, acceptance_rate=0.17,
         website_url="https://www.ox.ac.uk"),
    dict(name="University of Cambridge", country="United Kingdom",
         min_sat_overall=1500, min_sat_math=780, min_sat_reading=740,
         min_ielts_overall=7.5, annual_tuition_usd=37000, acceptance_rate=0.18,
         website_url="https://www.cam.ac.uk"),
    dict(name="Imperial College London", country="United Kingdom",
         min_sat_overall=1480, min_sat_math=770, min_sat_reading=730,
         min_ielts_overall=7.0, annual_tuition_usd=43000, acceptance_rate=0.14,
         website_url="https://www.imperial.ac.uk"),
    dict(name="University College London", country="United Kingdom",
         min_sat_overall=1450, min_sat_math=750, min_sat_reading=720,
         min_ielts_overall=7.0, annual_tuition_usd=35000, acceptance_rate=0.13,
         website_url="https://www.ucl.ac.uk"),
    dict(name="University of Edinburgh", country="United Kingdom",
         min_sat_overall=1420, min_sat_math=730, min_sat_reading=700,
         min_ielts_overall=6.5, annual_tuition_usd=32000, acceptance_rate=0.41,
         website_url="https://www.ed.ac.uk"),
    # Canada
    dict(name="University of Toronto", country="Canada",
         min_sat_overall=1450, min_sat_math=750, min_sat_reading=710,
         min_ielts_overall=6.5, annual_tuition_usd=45000, acceptance_rate=0.43,
         website_url="https://www.utoronto.ca"),
    dict(name="University of British Columbia", country="Canada",
         min_sat_overall=1400, min_sat_math=720, min_sat_reading=690,
         min_ielts_overall=6.5, annual_tuition_usd=41000, acceptance_rate=0.52,
         website_url="https://www.ubc.ca"),
    dict(name="McGill University", country="Canada",
         min_sat_overall=1400, min_sat_math=720, min_sat_reading=690,
         min_ielts_overall=6.5, annual_tuition_usd=27000, acceptance_rate=0.46,
         website_url="https://www.mcgill.ca"),
    # Australia
    dict(name="University of Melbourne", country="Australia",
         min_sat_overall=1400, min_sat_math=720, min_sat_reading=690,
         min_ielts_overall=6.5, annual_tuition_usd=34000, acceptance_rate=0.70,
         website_url="https://www.unimelb.edu.au"),
    dict(name="University of Sydney", country="Australia",
         min_sat_overall=1380, min_sat_math=710, min_sat_reading=680,
         min_ielts_overall=6.5, annual_tuition_usd=36000, acceptance_rate=0.72,
         website_url="https://www.sydney.edu.au"),
    # Asia and Europe
    dict(name="National University of Singapore", country="Singapore",
         min_sat_overall=1450, min_sat_math=760, min_sat_reading=710,
         min_ielts_overall=6.5, annual_tuition_usd=29000, acceptance_rate=0.10,
         website_url="https://www.nus.edu.sg"),
    dict(name="ETH Zurich", country="Switzerland",
         min_sat_overall=1440, min_sat_math=770, min_sat_reading=690,
         min_ielts_overall=7.0, annual_tuition_usd=1600, acceptance_rate=0.27,
         website_url="https://ethz.ch"),
    dict(name="Delft University of Technology", country="Netherlands",
         min_sat_overall=1350, min_sat_math=720, min_sat_reading=650,
         min_ielts_overall=6.5, annual_tuition_usd=18000, acceptance_rate=0.45,
         website_url="https://www.tudelft.nl"),
    dict(name="University of Amsterdam", country="Netherlands",
         min_sat_overall=1330, min_sat_math=700, min_sat_reading=650,
         min_ielts_overall=6.5, annual_tuition_usd=16000, acceptance_rate=0.53,
         website_url="https://www.uva.nl"),
    dict(name="Trinity College Dublin", country="Ireland",
         min_sat_overall=1380, min_sat_math=710, min_sat_reading=680,
         min_ielts_overall=6.5, annual_tuition_usd=26000, acceptance_rate=0.34,
         website_url="https://www.tcd.ie"),
]

STRICTNESS_SAFETY = "Safety / Target"
STRICTNESS_REACH = "Reach"
SAT_REACH_TOLERANCE = 50
IELTS_REACH_TOLERANCE = 0.5
_EPS = 1e-9

# Official API: https://collegescorecard.ed.gov/data/api/
SCORECARD_URL = "https://api.data.gov/ed/collegescorecard/v1/schools.json"
SCORECARD_FIELDS = [
    "school.name",
    "school.school_url",
    "latest.student.size",
    "latest.admissions.sat_scores.percentile25.overall",
    "latest.admissions.sat_scores.percentile25.math",
    "latest.admissions.sat_scores.percentile25.critical_reading",
    "latest.admissions.sat_scores.average.overall",
    "latest.admissions.admission_rate.overall",
    "latest.cost.tuition.out_of_state",
    "latest.cost.tuition.in_state",
]


def sample_universities():
    """Return the built-in fallback dataset as a fresh DataFrame."""
    df = pd.DataFrame(SAMPLE_UNIVERSITIES, columns=DATA_COLUMNS)
    df["acceptance_rate"] = df["acceptance_rate"].astype(float)
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching US College Scorecard data...")
def fetch_scorecard_universities(api_key="DEMO_KEY", max_pages=2, per_page=100):
    """Fetch US universities from the College Scorecard API.

    Returns an empty DataFrame on any failure so the caller can fall back to the
    built-in sample. IELTS is not part of the Scorecard data and is left as NaN.
    Rows without any SAT score are dropped because they cannot be matched, and
    results are ordered by enrolment so the largest universities come first.
    """
    frames = []
    for page in range(max_pages):
        params = {
            "api_key": api_key,
            "fields": ",".join(SCORECARD_FIELDS),
            "per_page": per_page,
            "page": page,
            "sort": "latest.student.size:desc",
        }
        try:
            response = requests.get(SCORECARD_URL, params=params, timeout=20)
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError):
            break

        records = []
        for item in results:
            name = item.get("school.name")
            if not name:
                continue
            # The latest release often has no 25th percentile, so fall back to the
            # average score of admitted students.
            sat_overall = item.get("latest.admissions.sat_scores.percentile25.overall")
            if sat_overall is None:
                sat_overall = item.get("latest.admissions.sat_scores.average.overall")
            tuition = item.get("latest.cost.tuition.out_of_state")
            if tuition is None:
                tuition = item.get("latest.cost.tuition.in_state")
            url = item.get("school.school_url") or ""
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url
            records.append({
                "name": name,
                "country": "United States",
                "min_sat_overall": sat_overall,
                "min_sat_math": item.get("latest.admissions.sat_scores.percentile25.math"),
                "min_sat_reading": item.get(
                    "latest.admissions.sat_scores.percentile25.critical_reading"),
                "min_ielts_overall": None,
                "annual_tuition_usd": tuition,
                "acceptance_rate": item.get("latest.admissions.admission_rate.overall"),
                "website_url": url,
            })
        if not records:
            break
        frames.append(pd.DataFrame(records, columns=DATA_COLUMNS))
        if len(results) < per_page:
            break

    if not frames:
        return pd.DataFrame(columns=DATA_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    for col in ["min_sat_overall", "min_sat_math", "min_sat_reading", "min_ielts_overall",
                "annual_tuition_usd", "acceptance_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    return df[df["min_sat_overall"].notna()].reset_index(drop=True)


def find_matches(data, sat_score, ielts_score, max_tuition, countries=None,
                 strictness=STRICTNESS_SAFETY):
    """Filter universities by budget, country and score requirements.

    "Safety / Target" keeps universities whose requirement is at or below the
    student's score. "Reach" keeps universities whose requirement is above the
    student's score by at most 50 SAT points / 0.5 IELTS bands on every dimension.

    Missing requirements (e.g. no IELTS data from the College Scorecard) never
    exclude a university; they are shown as unknown instead.
    """
    df = data.copy()

    if not df.empty:
        df = df[df["annual_tuition_usd"].isna() |
                (df["annual_tuition_usd"] <= max_tuition)]
        if countries:
            df = df[df["country"].isin(countries)]

    df["sat_margin"] = sat_score - df["min_sat_overall"]
    df["ielts_margin"] = (ielts_score - df["min_ielts_overall"]).round(1)
    df["acceptance_rate_pct"] = df["acceptance_rate"] * 100

    if df.empty:
        return df.reset_index(drop=True)

    if strictness == STRICTNESS_REACH:
        sat_gap = df["min_sat_overall"] - sat_score
        ielts_gap = df["min_ielts_overall"] - ielts_score
        within_sat = sat_gap.isna() | (sat_gap <= SAT_REACH_TOLERANCE + _EPS)
        within_ielts = ielts_gap.isna() | (ielts_gap <= IELTS_REACH_TOLERANCE + _EPS)
        above_sat = sat_gap.fillna(0) > _EPS
        above_ielts = ielts_gap.fillna(0) > _EPS
        keep = within_sat & within_ielts & (above_sat | above_ielts)
    else:
        meets_sat = df["min_sat_overall"].isna() | (df["min_sat_overall"] <= sat_score + _EPS)
        meets_ielts = df["min_ielts_overall"].isna() | (
            df["min_ielts_overall"] <= ielts_score + _EPS)
        keep = meets_sat & meets_ielts

    df = df[keep]
    df = df.sort_values(["annual_tuition_usd", "name"], na_position="last")
    return df.reset_index(drop=True)
