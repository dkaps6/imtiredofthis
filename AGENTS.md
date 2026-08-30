# AGENTS.md — 2026 Production Operating Contract

**Repository:** `dkaps6/imtiredofthis`

**Purpose:** end-to-end NFL player-prop research and production pipeline.

This file defines how automation should operate the repository after the 2026 production overhaul began. Historical migrations and backtests remain available for research lineage, but they are **not** alternate production entry points.

---

## 0) Canonical production authority

The only canonical production orchestration path is:

`.github/workflows/full-slate.yml`

The old `engine/engine.py` path is retired and intentionally fails closed. Do not recreate or invoke a second production pipeline from legacy scripts.

Current production order:

1. resolve `SEASON`, `PRIOR_SEASON`, slate date and authoritative NFL week;
2. build current Ourlads roster/roles;
3. build authoritative team/week schedule map;
4. optionally fetch current player props/game odds;
5. build team context with explicit active/prior provenance;
6. build promoted M89/M90 QB context;
7. build weather and injuries;
8. build coverage/WR-CB context;
9. optionally build current-season PBP enrichments;
10. build PlayerForm prior/current evidence;
11. build canonical context, Bayesian, ML, State and rule components;
12. materialize promoted ensemble calibration status;
13. assemble deterministic metrics;
14. run joint simulation/pricing;
15. for QB passing yards, apply the promoted M89/M90 football-only synthesis before sportsbook comparison;
16. run strict audits and upload artifacts.

Do not bypass this order when producing live 2026 projections.

---

## 1) Runtime season contract

Production defaults are:

- `SEASON=2026`
- `PRIOR_SEASON=2025`

The authoritative values come from `scripts/config.py` / `scripts/runtime_context.py` and workflow inputs. The NFL week must come from `data/team_week_map.csv` / schedule authority, never ISO week arithmetic or a hard-coded opening date.

A reference to 2025 is legitimate only when it is explicitly a **prior-season** source, historical validation lineage, or backtest season. A live/current-season field must never silently relabel 2025 data as 2026.

Every important provider/context artifact should eventually expose:

- active season;
- target week;
- source season;
- source/provider name;
- freshness/fetched-at state;
- fallback state (`current`, `prior`, `unavailable`, `stale`).

Use `python scripts/audit_2026_production_readiness.py` during the overhaul. Once its P0/P1 findings are closed it should become a required gate.

---

## 2) Research-to-production rule

Do **not** integrate every migration.

Only validated/promoted winners and the exact artifacts/parameters required to reproduce them belong in production. Failed, source-blocked, forensic and diagnostic migrations remain research evidence only.

Promoted production concepts currently include:

- joint Monte Carlo game environment and finite opportunity allocation;
- empirical-Bayesian player baselines;
- fixed pass/rush opportunity framework;
- gentle pressure adjustment;
- top-five rushing opportunity pool;
- WR hierarchy target-share sharpening while preserving team WR target mass;
- supervised ML v2 and State v2 component models;
- M89 corrected QB stat/context semantics;
- M89/M90 football-only QB passing-yards residual synthesis.

A model is not fully integrated merely because its Python module runs. Its **validated calibration/weights, feature semantics and provenance** must also survive a clean production checkout.

Broad QB mean-projection research is frozen after M90. Production fixes are allowed; a new generic QB feature hunt is not.

---

## 3) Sportsbook separation contract

Sportsbook player lines/odds are comparison/decision information unless a model is explicitly labeled market-assisted.

For the independent football projection:

- sportsbook variables must not construct QB synthesis features;
- sportsbook variables must not train MC/ML/State ensemble weights;
- postgame forensic/casebook variables must never enter pregame prediction;
- market-assisted outputs must remain separately labeled and must not silently replace the football model.

Audit other markets during the 2026 overhaul because legacy game-odds context may still influence some non-QB simulation paths.

---

## 4) Provider behavior

Provider failures should be handled according to the importance of the data, not with one blanket rule.

**Required identity/core inputs** (schedule, roster/slate identity, required production model artifacts) should fail closed if missing or internally inconsistent.

**Optional enrichments** may fail softly, but the failure must be visible and the artifact must be labeled unavailable rather than silently reusing stale cached data.

Never treat file existence alone as proof that an artifact is valid. Validate schema, rows, season/week and provenance.

Current key source families include:

- nflreadpy/nflverse — schedules, weekly player stats, PBP;
- Ourlads — current roster/depth roles;
- Sharp Football — team tendencies/context enrichment;
- ESPN/NFL.com/nflverse — injury context depending on builder;
- NWS + authoritative schedule/stadium map — weather;
- FantasyPoints/other WR-CB sources — optional coverage enrichment;
- The Odds API — live sportsbook props/game odds when explicitly enabled.

Do not reactivate legacy GSIS/API-Sports/MySportsFeeds paths merely because credentials or old files exist. They require an explicit 2026 source/semantic audit before use.

---

## 5) Canonical artifacts

Protect these schemas/meanings unless a migration explicitly versions them:

- `data/team_week_map.csv` — authoritative schedule/team/opponent/week map;
- `data/roles_ourlads.csv` — current roster/depth roles;
- `data/team_form.csv` — active-slate team context with provenance;
- `data/qb_promoted_team_context.csv` — M89/M90 corrected QB team context;
- `data/player_game_logs.csv` — historical weekly player evidence;
- `data/player_form.csv` / `data/player_form_consensus.csv` — current pregame player evidence;
- `data/model_ml_diagnostics.csv` / `data/model_state_diagnostics.csv` — current component diagnostics;
- `data/model_ensemble_weights.csv` — **promoted/frozen** production ensemble calibration by market;
- `data/model_ensemble_diagnostics.csv` — current production calibration status;
- `data/metrics_ready.csv` — canonical pricing input rows;
- `outputs/props_raw.csv` / `outputs/odds_game.csv` — sportsbook inputs when live odds enabled;
- `outputs/props_priced_clean.csv` — final priced projection output;
- `model/qb_pass_synthesis_v1.json` — transparent M89/M90 deployment fit.

Do not silently rename/drop required columns. Add explicit versioned migrations when semantics change.

---

## 6) Current Full Slate usage

The workflow is manually dispatchable. Inputs include:

- `season` (default 2026)
- `prior_season` (default 2025)
- `date`
- `fetch_live_odds`

Prefer a **no-live-odds dry run** first during production changes so provider/context/model plumbing can be checked without Odds API credits. Then run a controlled live-odds slate.

When live odds are disabled, PlayerForm should build its universe from Ourlads + authoritative schedule and must not consume stale props placeholders.

---

## 7) Production validation requirements

A production run should fail rather than claim success when:

- active season/week identity is unresolved;
- team/opponent identity is ambiguous;
- required roster/schedule artifacts are empty;
- promoted QB synthesis exists but the calibrated pass-yards ensemble base is unavailable;
- promoted QB context cannot be built for all required teams;
- an official-attempt conversion is missing/out of range;
- a priced QB pass-yards row bypasses the promoted synthesis;
- a source is labeled current while its source season is actually prior/stale.

Expected QB pricing audit columns include:

- `mc_proj`
- `ml_proj`
- `state_proj`
- `ensemble_proj`
- `qb_synthesis_proj`
- `qb_synthesis_correction`
- `qb_synthesis_version`
- `qb_attempt_conversion`
- `qb_pred_attempts`
- `qb_pred_ypa`

The final QB `model_proj` must equal the promoted synthesis mean within numerical tolerance.

---

## 8) Testing and audits

Before merging production changes:

```bash
python -m compileall scripts engine tests
python scripts/utils/audit_repo.py --strict
pytest -q
```

During the 2026 overhaul also run:

```bash
python scripts/audit_2026_production_readiness.py
```

Do not weaken a truth/integrity gate to make a run green. Fix the data/source/semantic mismatch or label the source unavailable.

---

## 9) Overhaul priorities

The authoritative roadmap is `docs/production/2026_MAIN_BRANCH_AUDIT.md`.

Current order:

1. P0 correctness blockers;
2. Team Context v3 / provenance cleanup;
3. provider 2026 readiness matrix;
4. roster/player Week 1 readiness;
5. research-to-production promotion ledger;
6. 2026 dry run + controlled live run;
7. resume RB, WR and TE model refinement on the corrected production foundation.

Do not spend production-overhaul time rewriting historical/backtest code simply because it contains 2025. Historical years are expected there. Focus on code that is executed by, imported into, or supplies artifacts to Full Slate.
