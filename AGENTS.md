# AGENTS.md — How an Automation Agent Should Operate This Repo

**Repository name:** `imtiredofthis`

**Purpose:** End‑to‑end NFL player props pipeline — fetch data (with robust fallbacks), build team/player features, price props, and export tidy CSVs for betting and analysis.

This playbook is written for an *automation agent* (LLM/runner) to execute tasks deterministically, recover from common errors, and produce consistent outputs.

---

## 0) High‑Level Flow (contract for the agent)

1. **Fetch upstream data** (web + APIs) → write CSVs to `data/` and `outputs/`:
   - Sharp Football tendencies & pace (HTML parsing)
   - Depth charts (OurLads; ESPN fallback if available)
   - Injuries (if configured)
   - nflverse / GSIS / API‑Sports / MySportsFeeds (best‑effort)
   - Sportsbook props via Odds API (v4)
2. **Build features**:
   - `team_form` (seed with Sharp, then fall back/merge others)
   - `player_form` (position/role inference + usage/efficiency)
3. **Join pricing table** → `scripts/make_metrics.py`
4. **Run pricing models** (MC/Bayes/agent rules) → write priced props
5. **Validate** key columns and emit debug summaries
6. **Export final CSVs** under `outputs/`

**Do not fail** if a single provider is down. Prefer partial success with clear logs over a hard stop.

---

## 1) Required Runtime & Setup

- **Python**: 3.10+ recommended
- **Install**: `pip install -r requirements.txt` (if present). If not, install standard libs used here: `pandas`, `requests`, `numpy`, `pyyaml`, etc.
- **Environment variables / Secrets** (GitHub Actions uses these):
- `APISPORTS_KEY`
- `ESPN_COOKIE`
- `MSF_KEY`
- `MSF_PASSWORD`
- `NFLGSIS_PASSWORD`
- `NFLGSIS_USERNAME`
- `ODDS_API_KEY`

> The agent must surface a clear error if a required secret is missing *for the specific step that needs it*, but continue other steps when possible.

**Local .env example** (agent may create for user if missing):

```env
APISPORTS_KEY=...
ODDS_API_KEY=...
ESPN_COOKIE=...
MSF_KEY=...
MSF_PASSWORD=...
NFLGSIS_USERNAME=...
NFLGSIS_PASSWORD=...
```

**Directories the agent should ensure exist** (safe to `mkdir -p`):

- `data/`, `outputs/`, `logs/`, `external/`

---

## 2) Canonical Entry Point

Use the orchestrator if available: `/mnt/data/repo/imtiredofthis-main/engine/engine.py` (exists: **True**).

**CLI**:

```bash
python engine/engine.py --season 2025 --date YYYY-MM-DD --markets "player_reception_yds,player_rush_yds,player_pass_yds" --bookmakers "fanduel,draftkings,betmgm" --debug
```
- `--date` blank → latest slate
- `--bookmakers` and `--markets` are **comma‑separated**; both optional. If omitted the pipeline will use defaults from `scripts/config.py`.

- `--debug` prints top‑of‑file snapshots for sanity checks.

The engine will internally call provider pulls, feature builders, metrics join, pricing, and export. If the engine is not usable, fall back to **manual playbooks** below.

---

## 3) Manual Playbooks (when running steps individually)

> The agent should prefer this order. Each step is **idempotent** and should not crash if inputs are missing.

### 3.1 Providers (write raw CSVs)

**Sharp Football** (preferred seeding for team form: box/coverage/pace/tendencies):

```bash
python scripts/providers/sharpfootball_pull.py --season 2025
```
**Depth charts (OurLads first)**:

```bash
python scripts/providers/ourlads_depth.py --season 2025 --date YYYY-MM-DD
```
> Notes: OurLads HTML can include random numbers/spacing after names. This repo’s depth parser already strips artifacts and normalizes LAST, FIRST ordering. The agent must still log any rows with trailing text and keep the clean `player` and `position` keys.

**Injuries** (if available):

```bash
python scripts/providers/injuries.py --season 2025 --date YYYY-MM-DD
```

**nflverse / GSIS / API‑Sports / MSF** (best‑effort; run all, ignore failures):

```bash
python scripts/providers/gsis_pull.py --season 2025
python scripts/providers/espn_pull.py --season 2025
python scripts/providers/apisports_pull.py --season 2025
python scripts/providers/msf_pull.py --season 2025
```

**Sportsbook props (Odds API v4)**:

```bash
python scripts/fetch_props_oddsapi.py --markets player_reception_yds,player_rush_yds,player_pass_yds --bookmakers fanduel,draftkings,betmgm --region us
```
Key behavior the agent must know:

- Accepts many synonyms; canonical market keys include:
  - `player_reception_yds` (receiving yards)
  - `player_rush_yds` (rushing yards)
  - `player_pass_yds` (passing yards)
  - others in the script’s `MARKET_ALIASES`
- Book aliases mapped (e.g., `dk`, `draftkings`; `mgm`, `betmgm`; `rivers`, `betrivers`, `barstool`).

- Writes **`outputs/props_raw.csv`** and per‑book breakdowns; will devig if configured downstream.

### 3.2 Feature builders

**Team form** (Sharp FIRST, then merge fallbacks; normalizes %-like fields such as `"61%" → 0.61"`):

```bash
python scripts/make_team_form.py --season 2025
```
**Player form** (multi‑source position/role inference; usage/efficiency shares):

```bash
python scripts/make_player_form.py --season 2025
```

### 3.3 Pricing join & validation

**Join all metrics to props**:

```bash
python scripts/make_metrics.py
```
**Validate required columns** (non‑fatal; prints report and writes cleaned outputs):

```bash
python scripts/validate_metrics.py
```

> Afterwards, the agent should emit a compact summary of row/column counts and the first 5 rows of each key table when `--debug` was set.

---

## 4) Output Contracts (schemas the agent should protect)

The following CSVs are critical; agents must not change their column names without explicit instruction.

### 4.1 `data/team_form.csv` (created by `make_team_form.py`)
- `team`, `season`
- Tendencies (seeded from Sharp; 0–1 floats): `light_box_rate`, `heavy_box_rate`, `man_rate`, `zone_rate`, `pressure_rate`, `blitz_rate`, `pace_seconds`, `pass_rate_over_expected`, etc.

### 4.2 `data/player_form.csv` (created by `make_player_form.py`)
- `player`, `team`, `season`, `position`, `role`
- Usage & efficiency: `tgt_share`, `route_rate`, `rush_share`, `yprr`, `ypt`, `ypc`, `receptions_per_target`, `rz_share`, `rz_tgt_share`, `rz_rush_share`

### 4.3 `outputs/props_raw.csv` (created by `fetch_props_oddsapi.py`)
- `event_id`, `bookmaker`, `market`, `player`, `team`, `outcome`, `line`, `price`, `timestamp`

### 4.4 `outputs/props_priced_clean.csv` (downstream pricing step)
- `player`, `team`, `market`, `bookmaker`, `fair_line`, `fair_price`, `edge_pct`, `stake_units`, etc.

The agent must **fail softly** (skip rows) when upstream data is missing, but **never** silently rename or drop the above without logging.

---

## 5) Configuration knobs (for the agent)

Central config: `/mnt/data/repo/imtiredofthis-main/scripts/config.py` (exists: **True**)

- Default `BOOKS`, `MARKETS`, regions
- MC simulation parameters
- Paths used by writers (agent should not change paths; only read/update values if requested)
- A `summary_json()` helper prints the current run configuration

The agent should use config defaults when CLI flags are omitted.

---

## 6) Known Pitfalls & How the Agent Should Respond

- **OurLads HTML artifacts**: Names may include suffix numerals or spacing. Always strip to canonical `player` and preserve `position`. Log examples to `logs/ourlads_*.log`.

- **ESPN depth chart empty**: Not fatal; keep running other providers.

- **Sharp pages 403/structure shifts**: The Sharp puller renders HTML locally and selects tables by header signature. If zero rows, retry once; then continue.

- **Odds API rate‑limits**: Back off and retry with exponential wait; if still failing, proceed with whatever markets are cached or skip to feature building.

- **ID mapping**: Use `external/nflverse_bundle/outputs/id_map.csv` when present; otherwise rely on internal normalizers in feature builders.

- **Partial slates**: `--date` can be blank; builders should still compute season aggregates.

When any step fails, the agent must:
1) Log a **one‑line** error with the exact file & function.
2) Continue other independent steps.
3) At the end, produce a **Run Summary** listing successes/failures.

---

## 7) CI / GitHub Actions

Workflow: `/mnt/data/repo/imtiredofthis-main/.github/workflows/full-slate.yml` (exists: **True**)

- Dispatch inputs: `season`, `date`

- Env secrets: APISPORTS_KEY, ESPN_COOKIE, MSF_KEY, MSF_PASSWORD, NFLGSIS_PASSWORD, NFLGSIS_USERNAME, ODDS_API_KEY

- Typical job: setup → providers → features → pricing → artifacts upload

- Concurrency is enabled (`full-slate-${ github.ref }`)

The agent may trigger this workflow via repository_dispatch or prompt the user to click **Run workflow** when interactive.

---

## 8) Extending to a New Market or Book

- **Market synonyms** live inside `scripts/fetch_props_oddsapi.py` (`MARKET_ALIASES`). Add the new synonym → canonical key.

- **Book aliases** live in the same file (`BOOK_ALIASES`). Add lower‑case alias → vendor key.

- Ensure `make_metrics.py` recognizes the new `market` and has the right feature joins; add feature rows if needed.

- Update tests/validators (or at minimum, re‑run `validate_metrics.py`).

---

## 9) Minimal Smoke Test (agent can run locally)

```bash
# 1) Providers (best-effort)
python scripts/providers/sharpfootball_pull.py --season 2025
python scripts/providers/ourlads_depth.py --season 2025
python scripts/fetch_props_oddsapi.py --markets player_reception_yds --bookmakers draftkings,fanduel --region us

# 2) Features
python scripts/make_team_form.py --season 2025
python scripts/make_player_form.py --season 2025

# 3) Join + Validate
python scripts/make_metrics.py
python scripts/validate_metrics.py
```

Expected: `data/team_form.csv`, `data/player_form.csv`, `outputs/props_raw.csv`, and a non‑empty `metrics_ready.csv`. If empty, the agent must print the top 10 distinct `market` and `bookmaker` values found and suggest a likely alias fix.

---

## 10) Run Summary Template (what the agent should output at the end)

```
[RUN] 2025-10-19T00:00Z
- providers: sharp=OK ourlads=OK injuries=SKIP espn=EMPTY gsis=OK apisports=SKIP msf=SKIP
- props: 3 markets, 2 books → 412 rows
- features: team_form=32 rows, player_form=~600 rows
- metrics: 400 joined rows (12 columns added)
- outputs: outputs/props_priced_clean.csv (if pricing step enabled)
Artifacts: data/*.csv, outputs/*.csv, logs/*.log
```

---

### File existence map (for quick reference)

{
  "engine": true,
  "providers_adapter": true,
  "config": true,
  "make_team_form": true,
  "make_player_form": true,
  "make_metrics": true,
  "fetch_props": true,
  "validate_metrics": true,
  "workflow": true
}

