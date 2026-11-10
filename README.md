# 🎓 University Finder

A Streamlit app that helps a student move from "I don't know where to apply" to a short list of
universities that fit their interests, their exam scores and their budget.

**The question it answers:** *Which universities actually fit me, and why?*

**Data:** the QS World University Rankings **2024** official results table (`qs_2024.xlsx`). The app
reports the exact edition, university count and country count it loaded on the *About the data* tab.

**Live app:** <PASTE STREAMLIT CLOUD URL HERE>

## Screenshots

| Find my university | Why this match? |
| --- | --- |
| ![Match results](<PASTE SCREENSHOT PATH: docs/finder.png>) | ![Match breakdown](<PASTE SCREENSHOT PATH: docs/why-match.png>) |

| Admissions fit | Saved universities |
| --- | --- |
| ![Admissions](<PASTE SCREENSHOT PATH: docs/admissions.png>) | ![Saved](<PASTE SCREENSHOT PATH: docs/saved.png>) |

## Features

- **Find my university** — weight the nine QS indicators with sliders (or pick a ready-made student
  profile), filter by country, size, subject focus, research intensity and rank band, and get a
  ranked list of matches.
  - Bar chart of the best matches, radar chart comparing your top three, and a country chart.
  - **Why this match?** picks any of your results and shows a stacked bar of how many match-score
    points each metric adds. The bars add up to the match score, and metrics a university has no
    score for are left out instead of being counted as zero.
  - A **Rank change** column (`▲ 12`, `▼ 5`, `–`) compares with the previous QS edition when that
    file is in the project.
- **University Finder (admissions)** — check SAT, IELTS and tuition against admission requirements,
  with a strictness setting and a **Chance** column (Safe / Target / Reach / Unknown) that follows
  one stated rule and is a rough guide, not a prediction, plus the university's QS rank when the
  name is in the loaded table. Uses the bundled sample, or live US data from the College Scorecard
  API.
- **Requirement lookup** — type any university and get a requirement record for it (AI-generated
  through Groq, with a bundled offline database as the fallback).
- **Find similar universities** — pick a university you like and see the closest profiles by
  similarity, with a radar comparison.
- **My saved universities** — accounts (register / log in), save a university from any result list,
  add a note, and compare saved universities side by side.
- **Explore the data** — country counts in the top N and the relationship between rank and the
  individual indicators.
- **About the data** — the edition, row count and country count that are actually loaded, the
  cleaning steps, how the match score works, and the limitations.
- Light and dark mode, and a layout that also works on a phone screen.

## Data sources

| Data | Source | Status |
| --- | --- | --- |
| QS indicator scores, ranks, country, size, focus, research intensity | QS World University Rankings **2024**, official results table (Excel) — <https://www.topuniversities.com/world-university-rankings> | `qs_2024.xlsx` in the project folder; this app is pinned to the 2024 edition |
| Rank change | The previous edition's `qs_*.xlsx` file | Only shown when two editions are present |
| Sample SAT / IELTS / tuition / acceptance rates | Bundled in `uni_data.py` | **source needed** — the figures are illustrative and their origin is not documented in this repository |
| Offline requirement records | Bundled in `fallback_requirements.json` | **source needed** |
| Live US SAT scores, admission rates, out-of-state tuition | US College Scorecard API — <https://collegescorecard.ed.gov/data/api/> | Live, optional; US institutions only |
| Requirement lookups for any university | Groq API, model `llama-3.3-70b-versatile` | Live, optional; results are AI-generated |

The figures shown for tuition and test scores are estimates. Always check the university's official
website before you rely on them.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

The QS workbook must be called `qs_YYYY.xlsx` and sit in the project folder (or in `data/`). This app
uses the **2024 edition** (`qs_2024.xlsx`) on purpose. A second workbook in the folder is only used
to fill the rank-change column; with a single edition that column is hidden.

## Secrets

Never commit real keys. The app reads them from Streamlit secrets or from environment variables:

| Name | Used for |
| --- | --- |
| `GROQ_API_KEY` | Requirement lookup through Groq |
| `COLLEGE_SCORECARD_API_KEY` | Live US College Scorecard data (defaults to the public `DEMO_KEY`) |

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your own values. The
real file is listed in `.gitignore`. Without a key the app still works: it falls back to the bundled
data.

## Limitations

- The table covers the universities QS ranks; institutions outside it (including Mongolian
  universities) cannot be found.
- Only a few hundred universities have an overall score, so the finder scores matches from the
  indicator scores instead.
- QS measures reputation and research more than teaching quality.
- Admissions figures are approximate, and the sample data is illustrative rather than official.
- AI answers can be wrong; verify them on the official website.
- Newer QS editions exist; this app uses the 2024 edition.
- Accounts and saved universities are a class project, not a real service. Saved lists live in a
  local SQLite file (`data/users.db`) that is not committed.
- **On free Streamlit Community Cloud the disk is temporary:** the app can restart, sleep after
  inactivity, and lose `data/users.db`, so saved universities and accounts may disappear at any
  time.
