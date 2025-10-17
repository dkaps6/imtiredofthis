# imtiredofthis Sharp Edge: Prop Intelligence System v3.0

Generated: 2025-10-09T19:28:16.314277Z

## What this is
A plug-and-play pipeline that:
1) Fetches NFL data with failovers (nflverse → ESPN → NFLGSIS → API-Sports → MSF)
2) Builds team & player feature tables
3) Fetches sportsbook props
4) Runs predictive models (MC/Bayes/Markov/ML with ABM adjustments)
5) Outputs edges + fair odds

## Entry points
- `run_model.py` — run everything end-to-end (local or CI)
- `fetch_all.py` — compatibility wrapper that only runs the provider chain
- `make_all.py` — compatibility wrapper that runs metrics + props + predictors

## Working dirs
- `data/`    — ingested + engineered tables
- `outputs/` — props + predictions
- `logs/`    — per-run audit trail

[README.md](https://github.com/user-attachments/files/22803928/README.md)

# NFL Props Model — Turn‑Key (Odds API + Free Features)

This repo is a **turn‑key pipeline** to pull NFL game lines & player props (including alternates) from **The Odds API**, build **external features** from free sources (nfl_data_py), price markets with your **ELITE model spec** (μ/σ + post‑mortem rules), and export tidy CSVs. It runs **locally** or on **GitHub Actions**.

> You provide one secret: `ODDS_API_KEY` (from https://the-odds-api.com/).  
> Optional: later add other sources (e.g., authenticated dashboards) by extending `scripts/*.py`.

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
export ODDS_API_KEY=YOUR_KEY_HERE
python run_model.py --date today --season 2025 --write outputs
```

Artifacts:
- `outputs/game_lines.csv` — H2H / spreads / totals (normalized)
- `outputs/props_priced.csv` — Player props (with alternates), model μ/σ, blended probabilities, fair odds, **edge%**, **kelly** and **tier**.
[Uploading README.md…]()

### What to do next

1. **Prime the data folders.** Drop any external scouting or share tables into `data/`. The builders now auto-detect both the `*_form.csv` files *and* the raw `espn_*.csv`, `msf_*.csv`, `apisports_*.csv`, `gsis_*.csv`, and `pfr_*` exports that already ship in this repo.
2. **Build team context:** `python scripts/make_team_form.py --season 2025`
   *The builder now **fails fast** if nflverse can’t serve 2025. Pass `--allow-fallback` only when you intentionally want an earlier proxy season.*
3. **Build player usage:** `python scripts/make_player_form.py --season 2025`
   *Same contract: no older data unless you explicitly add `--allow-fallback`.*
4. **Run the full engine (optional while debugging):** `python -m engine --season 2025 --debug`
   *Set `ALLOW_NFL_FALLBACK=1` if you truly need to let the pipeline reuse prior seasons. Otherwise it will halt when live pulls fail so you never price props with stale data.*

After each builder runs you should see `data/team_form.csv`, `data/team_form_weekly.csv`, and `data/player_form.csv` populated. They’ll report the `source_season` column so you can verify which year powered the current projections.

---

## Inspecting run summaries

Every invocation of `python -m engine` now appends a compact JSON line to `logs/actions_summary.log` and writes a detailed copy to `logs/daily/run_<RUN_ID>.json`. Each record captures:

- which steps succeeded/failed (fetch, team/player builders, metrics join, pricing, predictors, export)
- the row/column counts for critical CSVs (team_form, player_form, metrics_ready, props_priced)
- the `source_season` used for team/player fallbacks (only present when you enable them)
- run timing metadata (`run_id`, `started_at`, `duration_s`, etc.)

Use it on GitHub Actions to confirm a slate ran cleanly, or locally via:

```bash
tail -n 1 logs/actions_summary.log | jq
```

This surfaces the most recent run without downloading the full artifact bundle.

---

## What’s inside (modules)

- `scripts/odds_api.py` → pulls **game lines** and **player props** (event endpoint) from The Odds API.
- `scripts/features_external.py` → free features via **nfl_data_py**: schedules, IDs, weekly stats, injuries, depth, plus **rolling L4** team EPA/SR and player form.
- `scripts/id_map.py` → robust **player name → GSIS ID** resolver with a small cache file (`inputs/player_id_cache.csv`).  
- `scripts/model_core.py` → μ/σ scaffolding and the **post‑mortem rules** hooks (pressure, funnels, volatility widening, etc.).  
- `scripts/pricing.py` → **de‑vig**, probability/odds converters, **65/35 market blend**, **edge%**, **kelly**, **tiering**.  
- `scripts/correlations.py` → simple pairwise **SGP** correlations (placeholders you can tune).  
- `scripts/calibration.py` → **Brier/CRPS** scaffolding + μ shrinkage for next slate.  
- `engine.py` → orchestration: fetch → features → pricing → write CSVs.  
- `run_model.py` → small CLI wrapper.

> The μ calculation is intentionally modular: start with volume × efficiency; layer your rules (pressure, sacks, coverage, injuries, box counts, pace smoothing).

---

## GitHub Actions (Scheduled)

1. Add a repo Secret: **Settings ⇒ Secrets and variables ⇒ Actions ⇒ New repository secret**  
   - Name: `ODDS_API_KEY`  
   - Value: your API key.

2. Push repo. The workflow at `.github/workflows/full-slate.yml` can run manually or on schedule.  
   It uploads the **outputs/** folder as an artifact (`nfl-outputs`).

---

## Player name mapping (very important)

Sportsbook names can differ from official IDs. We ship a resolver that:
- Normalizes names, attempts a direct join to nfl_data_py IDs,
- Falls back to fuzzy-ish normalization,
- Persists the mapping in `inputs/player_id_cache.csv` so you only fix once.

If a player doesn’t map automatically, the row appears with `gsis_id` empty; you can **add a row** to the cache file with columns: `player_name_raw,gsis_id`. Re‑run and the mapping will fill.

---

## Extending μ and σ to your full spec

Open `scripts/model_core.py` and the “TODO” sections in `engine.py`:
- μ = **plays × team share × player share × efficiency**, then apply rules:  
  **pressure/sacks**, **coverage funnel (run/pass)**, **injury redistribution**, **box counts**, **pace smoothing**.  
- σ = market default × (1 ± volatility). Use widening when pressure mismatch or QB inconsistency flags appear.

---

## Legal & ToS

Use The Odds API per its terms and rate limits. For any other data source, ensure you have permission. This template is for **personal analytics** and research.

---

## License

MIT — do what you want, no warranty.
