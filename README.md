🧠 Sharp Edge: Prop Intelligence System v3.1

Elite NFL Predictive Pipeline – Fully Automated

Last updated: November 2025

🚀 Overview

This repository powers an end-to-end NFL props modeling engine, designed to integrate multiple free data sources, live sportsbook odds, and a robust statistical modeling layer to output fair-value pricing, edges, and confidence tiers for every player prop.

It runs locally or on GitHub Actions through a single workflow:
build → merge → price → export.

🧩 Architecture
Core Pipeline

Data ingestion
Fetches and caches raw data from multiple redundant providers:

nflverse → primary

ESPN / NFLGSIS / API-Sports / MSF → failover

The Odds API → game lines + player props

Feature engineering

Builds team_form.csv and player_form.csv with rolling 4-game splits, EPA, success rate, pressure, coverage, etc.

Produces metrics_ready.csv with combined player + team + opponent + weather context.

Pricing & prediction

Runs your ELITE μ/σ model (plays × team share × efficiency ± volatility)

Applies post-mortem modifiers: pressure, coverage funnels, injury redistribution, pace smoothing

Outputs: props_priced.csv with fair odds, edge %, Kelly, and tier

Audit & logging
Each run logs status + metrics in logs/actions_summary.log and detailed JSON under logs/daily/.

📦 Working Directories
data/        ← intermediate tables (team/player/metrics)
outputs/     ← priced props, game lines, exports
logs/        ← build + pricing run summaries
scripts/     ← build + enrichment code
.github/     ← GitHub Actions workflow

🛠 Recent Additions (v3.1)
Area	Update
Opponent Mapping	Introduced build_opponent_map_from_props.py → joins props_raw + odds_game via event_id to resolve team/opponent for every player; >90 % coverage now.
Name Canonicalization	Added scripts/utils/name_clean.py → standardizes player names (drops middle initials, suffixes, and punctuation) across all inputs for stable joins.
Metrics Coverage Audit	make_metrics.py now runs core-coverage checks; writes data/metrics_missing_core.csv listing any player missing team/opponent/position.
Workflow Ordering Fixes	Ensures props & odds fetch run before metrics build; pricing now uses --props data/metrics_ready.csv instead of invalid --date.
Weather Integration	Weather data now imported before metrics, enriching environmental splits for each slate.
Pricing CLI Fix	pricing.py now cleanly accepts --season, --props, and --write; removed deprecated --date arg.
Improved Error Handling	All builders log row counts and missing-data warnings without aborting unless critical.
Expanded Debug Outputs	Each builder now emits secondary CSVs (e.g., opponent_unmapped_debug.csv) for targeted QA.
⚙️ Entry Points
Script	Purpose
run_model.py	End-to-end orchestrator; runs entire slate locally or on CI
engine.py	Core sequence (fetch → build → metrics → pricing → export)
fetch_props_oddsapi.py	Pulls props and game lines from The Odds API
make_team_form.py	Builds team-level efficiency, EPA/SR, and situational metrics
make_player_form.py	Compiles player-level usage, target/rush share, and route rates
build_opponent_map_from_props.py	Derives player ↔ team ↔ opponent mapping via event_id
make_metrics.py	Merges everything into metrics_ready.csv
pricing.py	Calculates μ/σ, fair odds, edges, Kelly fractions, tiers
calibration.py	CRPS/Brier calibration + μ shrinkage
correlations.py	Experimental pairwise SGP correlations
🧪 Quick Start (Local)
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export ODDS_API_KEY=YOUR_KEY_HERE

# Run full build + pricing
python run_model.py --season 2025 --write outputs


Artifacts produced:

outputs/game_lines.csv
outputs/props_priced.csv
data/metrics_ready.csv
logs/actions_summary.log

🔁 GitHub Actions Integration

The workflow .github/workflows/full-slate.yml automates:

Dependency setup (Python 3.11/3.12)

Data fetch + feature builds

Opponent mapping

Metrics join & coverage audit

Pricing and export
→ Uploads outputs/ as artifact nfl-outputs

Requires one secret:

ODDS_API_KEY = your API key from https://the-odds-api.com/

🧮 Data Flow Summary
fetch_props_oddsapi  →  props_raw.csv
odds_game.csv        →  game lines
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

🧰 Developer Notes

Canonical Keys: player_clean_key, team_abbr, opponent_abbr, season, week

Critical Files:

data/player_form.csv

data/team_form.csv

data/opponent_map_from_props.csv

data/metrics_ready.csv

Audit Files:

data/metrics_missing_core.csv

data/opponent_unmapped_debug.csv

Logs:

logs/actions_summary.log (compact)

logs/daily/run_<timestamp>.json (full trace)

📊 Model Framework

μ = volume × efficiency × contextual rules

σ = baseline variance × volatility factor

Edge = (book odds – fair odds) / book odds

Kelly = edge × (p – q) / odds

Tiering = auto bucket by confidence percentile

🧱 Roadmap (Next Up)

✅ Full event-ID opponent mapping (complete)

🔄 Expanded player role inference (slot vs wide, committee splits)

🌦️ Integrate weather into μ/σ context weighting

📈 Simulation harness for 10 k × Monte Carlo price validation

⚡ Fast API microservice wrapper for dashboard deployment

🪪 License

MIT — free for research and personal use.
Use The Odds API per its ToS & rate limits.
