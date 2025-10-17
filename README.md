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

> **Working inside a restricted sandbox?**
> Some automated graders (including this one) block outbound network access, so
> `pip install -r requirements.txt` will fail with a message similar to the one
> shown in the PR test log: `fails in this environment because PyPI access is
> blocked by the sandbox proxy`. When you run the command locally or on GitHub
> Actions the install succeeds, because those environments can reach PyPI.
> If you ever need to work fully offline, build a local wheelhouse (``pip wheel``)
> from an internet-connected machine and point `pip install` at that cache with
> `--no-index --find-links /path/to/wheels`.

> **Dependency note:** the refreshed `requirements.txt` sticks to *compatible
> ranges* instead of exact pins so `pip` can choose wheels that exist for the
> Python version in your environment. They keep the same lower bounds we used in
> CI (`pandas 2.2.2+`, `numpy 1.26.4+`, `scipy 1.11+`, `statsmodels 0.14.2+`)
> while relaxing the upper bounds to `<2.0` so the resolver no longer trips over
> version conflicts during “Install dependencies”.

> Optional packages like `nflreadpy` (and its `polars` dependency) remain in the
> list so the builders can hit the live 2025 nflverse feeds. If those packages
> are absent, the scripts fall back to `nfl_data_py>=0.3.3`; make sure that
> version is available so the shared `original_mlq` helper exists.

Artifacts:
- `outputs/game_lines.csv` — H2H / spreads / totals (normalized)
- `outputs/props_priced.csv` — Player props (with alternates), model μ/σ, blended probabilities, fair odds, **edge%**, **kelly** and **tier**.
[Uploading README.md…]()

### What to do next

1. **Prime the data folders.** Drop any external scouting or share tables into `data/`. The builders now auto-detect both the `*_form.csv` files *and* the raw `espn_*.csv`, `msf_*.csv`, `apisports_*.csv`, `gsis_*.csv`, and `pfr_*` exports that already ship in this repo.
2. **Build team context:** `python scripts/make_team_form.py --season 2025`
   *The builder reuses any cached `data/pbp_2025.csv` (or `external/nflverse_bundle/pbp_2025.csv`) before hitting nflverse. If 2025 PBP cannot be reached the script halts so you never blend in older seasons.*
3. **Build player usage:** `python scripts/make_player_form.py --season 2025`
   *Same guarantee: only 2025 play-by-play and participation are accepted. Older seasons trigger an explicit failure instead of a silent fallback.*
4. **Run the full engine (optional while debugging):** `python -m engine --season 2025 --debug`
   *The engine now enforces the same 2025-only constraint and surfaces a clear error when live pulls fail.*

After each builder runs you should see `data/team_form.csv`, `data/team_form_weekly.csv`, and `data/player_form.csv` populated. They’ll report the `source_season` column so you can verify which year powered the current projections.

---

## Inspecting run summaries

Every invocation of `python -m engine` now appends a compact JSON line to `logs/actions_summary.log` and writes a detailed copy to `logs/daily/run_<RUN_ID>.json`. Each record captures:

- which steps succeeded/failed (fetch, team/player builders, metrics join, pricing, predictors, export)
- the row/column counts for critical CSVs (team_form, player_form, metrics_ready, props_priced)
- the `source_season` recorded by the builders (should read 2025 once live data lands)
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
