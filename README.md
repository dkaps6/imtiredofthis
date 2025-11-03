# 🧠 Sharp Edge: Prop Intelligence System v3.1  
**Elite NFL Predictive Pipeline – Fully Automated**

_Last updated: November 2025_

---

## 🧾 Changelog (v3.1 – 2025-11-03)

| Area | Update Summary |
|------|----------------|
| **Opponent Mapping** | Added `build_opponent_map_from_props.py` — joins `props_raw` + `odds_game` via `event_id` to derive `team_abbr` + `opponent_abbr` for every player (now >90 % coverage). |
| **Name Canonicalization** | Introduced `scripts/utils/name_clean.py` for consistent player name cleanup (removes middle initials, suffixes, punctuation; adds static overrides). |
| **Metrics Coverage Audit** | `make_metrics.py` now writes `data/metrics_missing_core.csv` and prints coverage counts for any missing opponent/team/position values. |
| **Workflow Order Fixes** | Ensures props + odds fetch happens **before** metrics; pricing now uses `--props data/metrics_ready.csv` instead of the old invalid `--date`. |
| **Weather Integration** | Weather collection (`build_weather_week.py`) now executes **before** metrics and enriches environmental splits. |
| **Pricing CLI Cleanup** | `pricing.py` now takes only `--season`, `--props`, and `--write`; removed legacy `--date` argument. |
| **Error-Handling Overhaul** | Builders now emit warnings and write audit CSVs instead of silently skipping data. |
| **Expanded Debug Outputs** | Additional CSVs like `opponent_unmapped_debug.csv` and explicit row counts added for QA. |

---

## 🚀 Overview

This repository powers an **end-to-end NFL player-prop modeling engine**, integrating **free public football data** with **real-time sportsbook odds** and a fully parameterized **pricing model**.  
It automates ingestion, enrichment, prediction, and export across a reproducible CI/CD workflow.

build → enrich → metrics → price → export

markdown
Copy code

Outputs include player-level projections, fair-odds lines, value percentages, and Kelly tiers.

---

## 🧩 Architecture

### Core Pipeline
1. **Data ingestion**
   - Sources: `nflverse`, `nflreadr`, `nfldata`, `ESPN`, `The Odds API`
   - Fetches player props, team stats, game lines, and situational data
2. **Feature engineering**
   - Builds rolling form metrics: team EPA, SR, pressure/coverage rates, pace
   - Creates `metrics_ready.csv` — unified dataset for pricing
3. **Pricing**
   - Runs μ/σ model per player & market (volume × efficiency ± variance)
   - Produces `props_priced.csv` with edge %, fair odds, Kelly, and confidence tier
4. **Audit + logging**
   - Every stage writes debug counts + CSVs under `/data` and `/logs`

---

## 📁 Directory Layout
data/ → intermediate tables (team, player, metrics)
outputs/ → final fair-value odds and projections
logs/ → run summaries and debug reports
scripts/ → build and enrichment scripts
.github/ → Actions workflow for full automation

yaml
Copy code

---

## ⚙️ Key Scripts

| Script | Purpose |
|---------|----------|
| `fetch_props_oddsapi.py` | Pulls props & game lines from The Odds API |
| `build_opponent_map_from_props.py` | Derives player ↔ team ↔ opponent via `event_id` join |
| `make_team_form.py` | Builds team-level efficiency, pace, and situational splits |
| `make_player_form.py` | Generates player usage, target/rush share, and routes data |
| `make_metrics.py` | Merges all layers into `metrics_ready.csv` |
| `pricing.py` | Calculates fair-odds μ/σ model outputs |
| `calibration.py` | Optional post-model CRPS/Brier calibration |
| `correlations.py` | Experimental SGP correlation exploration |
| `build_weather_week.py` | Fetches weekly weather and stadium data |

---

## 🧠 Data Flow

fetch_props_oddsapi → props_raw.csv
odds_game.csv → game lines
↓
build_opponent_map_from_props
↳ opponent_map_from_props.csv
↓
make_team_form / make_player_form
↓
make_metrics
↳ metrics_ready.csv
↓
pricing
↳ props_priced.csv

yaml
Copy code

---

## 🔁 GitHub Actions Workflow

### `.github/workflows/full-slate.yml`

The full CI/CD build automates:
1. Environment setup  
2. Data ingestion  
3. Opponent mapping  
4. Metrics build + coverage audit  
5. Pricing model execution  

Artifacts uploaded:
outputs/props_priced.csv
outputs/game_lines.csv
logs/actions_summary.log

markdown
Copy code

**Required secret:**
ODDS_API_KEY = your key from https://the-odds-api.com/

yaml
Copy code

---

## 🧮 Model Framework

| Symbol | Definition |
|---------|-------------|
| **μ** | Expected player outcome = volume × efficiency × context |
| **σ** | Player outcome volatility (based on historical variance × matchup) |
| **Edge** | (book_odds − fair_odds) / book_odds |
| **Kelly** | edge × (p − q) / odds |
| **Tier** | Percentile-ranked confidence grouping |

---

## 🧰 Developer Notes

- **Canonical keys:** `player_clean_key`, `team_abbr`, `opponent_abbr`, `season`, `week`
- **Core audit files:**  
  - `metrics_missing_core.csv` → missing team/opponent/position  
  - `opponent_unmapped_debug.csv` → unmatched event_ids
- **Logs:**  
  - `logs/actions_summary.log` → build summary  
  - `logs/daily/*.json` → detailed traces

---

## 🧪 Local Usage

```bash
pip install -r requirements.txt
export ODDS_API_KEY=YOUR_KEY_HERE

python run_model.py --season 2025 --write outputs
Artifacts produced:

bash
Copy code
data/metrics_ready.csv
outputs/props_priced.csv
logs/actions_summary.log
🧭 Coverage Audits
After each run:

[opponent_map] rows=... missing_opponent=...

[make_metrics] missing core coverage rows: ...

Missing players are written to data/metrics_missing_core.csv for investigation.

📈 Roadmap
✅ Canonical name + event-ID joins (done)

🔄 Expanded role classification (slot vs wide, committee shares)

🌦️ Weather → μ/σ weighting

📊 Monte Carlo & Bayesian calibration layer

⚡ API & dashboard deployment wrapper

👥 Maintainer
Maintained by: @dkaps6
Contributors welcome — open an issue or PR with fix: or enhancement: prefixes.

🪪 License
MIT License

You may use, modify, and distribute for research and personal use.
Commercial use of sportsbook odds data must comply with the respective API provider’s terms.
