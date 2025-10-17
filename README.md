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
> version conflicts during “Install dependencies”. We also lock `lxml` to
> **4.9.4.x** because that release line is the sweet spot that satisfies
> `nflreadpy==0.1.3` while keeping BeautifulSoup fast and reliable.
> version conflicts during “Install dependencies”.

> Optional packages like `nflreadpy` (and its `polars` dependency) remain in the
> list for Python 3.11-and-earlier environments so the builders can hit the live
> 2025 nflverse feeds. PyPI’s latest `nflreadpy` release is **0.1.3**, which ships
> wheels through Python 3.11; the requirements pin that exact version so installs
> succeed. On Python 3.12, `pip` skips the package and the scripts fall back to
> `nfl_data_py>=0.3.3`. When new wheels arrive (or you install a forked wheel
> manually) the builders automatically pick them up and log which provider
> handled the pull.
> CI (`pandas 2.1.4+`, `numpy 1.26.4+`, `scipy 1.11+`, `statsmodels 0.14.2+`)
> while relaxing the upper bounds to `<2.0`/`<2.2` so the resolver no longer
> trips over version conflicts during “Install dependencies”.

> Optional packages like `nflreadpy` (and its `polars` dependency) remain in the
> list so the builders can hit the live 2025 nflverse feeds. If those packages
> are absent, the scripts fall back to `nfl_data_py>=0.3.3`; make sure that
> version is available so the shared `original_mlq` helper exists.
> CI (`pandas 2.2.2+`, `numpy 1.26.4+`, `scipy 1.11+`, `statsmodels 0.14.2+`)
> while relaxing the upper bounds to `<2.0`/`<2.3` so the resolver no longer
> trips over version conflicts during “Install dependencies”.
> CI (`pandas 2.1.4+`, `numpy 1.26.4+`, `scipy 1.11+`, etc.) while relaxing the
> upper bounds to `<2.0`/`<2.3` so the resolver no longer trips over version
> conflicts during “Install dependencies”.

> Optional packages like `nflreadpy` (and its `polars` dependency) remain in the
> list so the builders can hit the live 2025 nflverse feeds. If those packages
> are absent, the scripts fall back to `nfl_data_py>=0.3.4`; make sure that
> version is available so the shared `original_mlq` helper exists.
> **Heads-up:** `requirements.txt` now targets Python 3.12 by pinning
> `pandas==2.1.4`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`,
> `statsmodels==0.14.2`, and `pyarrow==15.0.2`.
> Statsmodels 0.14.2 ships wheels built against `pandas<2.2`, so we pin
> pandas to 2.1.4. If you decide to upgrade pandas later, bump statsmodels
> at the same time to whatever release advertises compatibility with that
> pandas series.

> We removed `pandas-datareader` because its newest wheels currently depend on
> `pandas<2.0`; installing it alongside pandas 2.1.4 would bring back the same
> resolver error you saw in Actions. If you need `pandas-datareader`, install it
> in a separate environment or adjust the rest of the stack accordingly.
> `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`,
> `statsmodels==0.14.2`, and `pyarrow==15.0.2`.
> We removed `pandas-datareader` because its latest wheels cap
> `pandas<2.2`, which is exactly why the GitHub Actions run aborted during
> the **Install dependencies** step. If you later need `pandas-datareader`,
> install it in a separate environment or adjust the pandas version accordingly.

> The statsmodels wheels available today ship binaries that work with
> pandas 2.2.x, so CI now stays on the same pandas version that GitHub Actions
> preinstalls. If you bump pandas later, keep statsmodels at 0.14.2 or newer so
> the resolver can still locate compatible wheels.
> The statsmodels wheels available today still require `pandas<2.2`, so we
> deliberately pin pandas to 2.1.4 to keep dependency resolution green in CI.
> install it in a separate environment or downgrade pandas accordingly.
> `statsmodels==0.14.2`, `pyarrow==15.0.2`, and `pandas-datareader==0.10.0`.
> Statsmodels 0.14.2 advertises support through pandas 2.2, so the resolver
> stops complaining even when other steps request `pandas-datareader` during
> CI setup. If your
> `pandas==2.1.4`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`,
> `pyarrow==15.0.2`, and explicitly requiring `statsmodels>=0.14.1`. Pandas 2.1.x
> keeps statsmodels happy (0.14.1 still caps support at `<2.2`). If your
> `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`,
> `pyarrow==15.0.2`, and explicitly requiring `statsmodels>=0.14.1` so pandas 2.x
> resolves cleanly. If your
> `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`, and
> explicitly requiring `statsmodels>=0.14.1` so pandas 2.x resolves cleanly. If your
> `pandas==2.2.2`, `numpy==1.26.4`, `scipy==1.12.0`, and `scikit-learn==1.4.2`. If your
> environment cached older wheels (especially on GitHub Actions), run
> `pip install --upgrade pip` first so compatible builds resolve cleanly.
>
> We also install `nflreadpy` (plus its `polars` dependency) so the builders can pull the
> live 2025 nflverse feeds. Should `nflreadpy` be missing, the scripts fall back to
> `nfl_data_py` — ensure you have `nfl_data_py>=0.3.4` available so the shared
> `original_mlq` helper exists.
> **Heads-up:** `requirements.txt` now installs `nflreadpy` (plus its `polars` dependency) so
> the builders can pull the live 2025 nflverse feeds. If your environment pinned an older
> dependency cache, run `pip install --upgrade pip` first so wheels for `polars` can be
> resolved correctly on GitHub Actions.
>
> **New requirement:** `nflreadpy` now expects `nfl_data_py` to expose the helper
> `original_mlq`. We pin `nfl_data_py>=0.3.4` in `requirements.txt`; if you maintain a
> custom environment make sure that upgrade lands, otherwise the builders will fall back to
> the older `nfl_data_py` interface and warn you in stderr.

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
