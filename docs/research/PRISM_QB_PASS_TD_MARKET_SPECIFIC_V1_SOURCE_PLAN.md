# PRISM QB Passing-TD Market-Specific V1 — Frozen Source / Anti-Retest Plan

Date: 2026-10-18

Status: **FROZEN BEFORE SOURCE EXECUTION / SOURCE + LINEAGE ONLY / NO TD MODEL SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_QB_PASS_TD_MARKET_SPECIFIC_V1_SOURCE_AUDIT`

## 1. Why the queue is advancing here

The signed-opportunity frontier has been preserved fail-closed:
- WR exact authority recovery remains blocked by immutable byte identity;
- TE signed-opportunity source authority is live, but the already-frozen bridge requires the shared combined `READY_ALL` recovery state and therefore has no independently authorized execution path;
- WR/TE named-player opportunity-uncertainty science remains confirmed, but exact historical specialist receiving-yard distribution authority required for full-stack integration is no longer recoverable without violating prior parity rules.

Those are source/execution blocks, not permission to weaken gates.

The previously recorded QB passing-touchdown gap is therefore promoted from backlog to the next **genuinely distinct market-specific source frontier**.

This does not reopen or retune any blocked WR/TE/RB mechanism.

## 2. Scientific object

PRISM currently has strong QB passing-yards science but no dedicated certified QB passing-TD projection layer.

The eventual football question is:

> For this named QB in this game, how many passing touchdowns should PRISM expect after separating pass opportunity, team scoring/pass-TD creation, QB ownership and QB-specific conversion?

This V1 source audit does **not** answer that predictive question.

It asks only:

> Does the repository's sportsbook-independent historical football source preserve enough stable named-QB passing-TD and scoring-area passing topology to freeze a later player-first mechanism without outcome leakage?

## 3. Existing code is not certified QB pass-TD authority

The audit must preserve these distinctions.

### Odds acquisition is downstream only

`scripts/fetch_props_oddsapi.py` includes `player_pass_tds`.

That proves the market can be acquired from books.
It does **not** provide football projection authority and may not enter this source audit.

### Current simulation does not project QB passing TDs

`scripts/simulation_v2.py`:
- has a QB `pass_yards` output;
- has an `anytime_td` output;
- has no `pass_tds` market alias or QB passing-TD simulation output.

The generic `anytime_td` lane is rushing/receiving TD logic and must not be reused as passing-TD authority.

### Current ATD science explicitly excludes passing TDs

`scripts/enrich_player_scoring_v2.py` and
`scripts/validate_anytime_td_model_status_v1.py` preserve ATD as rushing + receiving only.

Canonical ATD remains:
`DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED`.

Do not merge ATD and QB passing-TD science.

### Prior QB TD-rate context is not a TD model

`scripts/backtest/fit_qb_joint_attempts_ypa_walkforward.py` computes a strictly-prior `qb_recent_td_rate`, but uses it only as one candidate contextual feature for a **passing-YPA** residual model.

It did not certify a passing-TD projection.

## 4. Anti-retest boundaries

Do not reopen or disguise:
- M89/M90 passing-yards mean synthesis;
- closed QB Stage-2 allocation;
- closed QB Stage-1 player-specific utility signal;
- prior QB game-script / efficiency / chaos families as a TD model;
- generic ATD probability;
- player high-value opportunity persistence V1;
- player high-value conversion identity V1;
- team direct high-value creation V1;
- team scoring-territory drive-creation V1;
- sportsbook-implied team totals or passing-TD prices as football predictors.

The high-value opportunity families above were RB/receiver opportunity-quality questions and their exact frozen transmissions remain closed.

A later QB pass-TD mechanism must be a new **QB/team passing-TD decomposition**, not a renamed yardage or ATD formula.

## 5. Frozen historical source

Primary source:
nflverse / nflfastR play-by-play through the repository's current `nflreadpy.load_pbp` path.

Historical source-audit seasons:
- 2022 REG;
- 2023 REG;
- 2024 REG;
- 2025 REG.

No 2026 outcomes.

Required PBP semantics, allowing only deterministic alias resolution:
- season;
- week;
- season type;
- offensive team / posteam;
- passer stable player ID;
- passer name where present;
- pass attempt;
- passing touchdown indicator;
- yardline_100.

Optional integrity/context fields may be reported:
- complete_pass;
- interception;
- sack / qb_dropback;
- game_id.

No sportsbook fields.

## 6. Source object to persist

For every named-QB team-game with at least one official pass attempt, persist:

- season;
- week;
- team;
- game_id where present;
- stable passer ID;
- passer display name where present;
- pass attempts;
- passing TDs;
- pass-TD rate;
- pass attempts inside opponent 20;
- pass attempts inside opponent 10;
- pass attempts inside opponent 5;
- pass TDs from inside opponent 20;
- pass TDs from inside opponent 10;
- pass TDs from inside opponent 5.

Also persist team-game totals:
- team pass attempts;
- team pass TDs;
- number of passers with attempts;
- top passer attempt share.

This is descriptive source topology only.

## 7. Strictly-prior history feasibility

Without scoring any future-TD relationship, report whether later target QB-games can form strictly-prior same-QB histories.

For 2024 and 2025 target rows:
- same named QB;
- strictly earlier game weeks/seasons only;
- target row excluded;
- report counts with >=3, >=5 and >=8 prior QB-games.

Do not calculate:
- predictive TD correlation;
- MAE;
- Brier/log loss;
- calibration;
- candidate expected TDs;
- player ranking.

## 8. Frozen source gates

Per season 2022-2025:

1. REG PBP nonempty;
2. >=30 offensive teams;
3. >=40 distinct stable passer IDs with an official pass attempt;
4. passer stable-ID coverage among pass attempts >=99.0%;
5. `pass_touchdown` coverage among pass attempts >=99.9%;
6. `yardline_100` coverage among pass attempts >=99.0%;
7. every regular-season week present in the source contains >0 pass attempts;
8. every regular-season week contains >0 passing TDs;
9. every regular-season week contains >0 pass attempts inside opponent 20;
10. duplicate `season/week/team/passer_id` source rows after aggregation = 0;
11. impossible passing-TD count rows (`pass_tds > pass_attempts`) = 0;
12. scoring-area count monotonicity violations
    (`inside5 <= inside10 <= inside20 <= attempts`) = 0;
13. scoring-area TD monotonicity violations
    (`inside5_td <= inside10_td <= inside20_td <= pass_tds`) = 0.

Cross-season history feasibility:
- 2024 target rows with >=3 strictly-prior same-QB games >=300 and >=35 QBs;
- 2025 target rows with >=3 strictly-prior same-QB games >=300 and >=35 QBs.

All gates are source/readiness gates.
They may not be lowered after execution.

If a field's nflreadpy semantics differ from the assumed name, deterministic alias resolution is allowed only when the replacement field has the same documented event meaning. No statistical proxy is allowed.

## 9. If source-ready

A later mechanism must be frozen separately before any TD outcome relationship is scored.

The first predictive architecture should decompose, rather than jump straight to a universal QB TD-rate scalar:

`QB PASS OPPORTUNITY -> TEAM PASS-TD CREATION / SCORING ACCESS -> NAMED QB OWNERSHIP -> QB-SPECIFIC PASS-TD CONVERSION -> PASS-TD COUNT DISTRIBUTION`

Required player-first rules:
- starter/current QB authority must be pregame;
- named-QB history must be strictly prior and shrunk toward a broader prior;
- team creation and player conversion are separate questions;
- red-zone/scoring-area pass opportunity is separate from conversion;
- passing yards and passing TDs remain separate markets;
- rushing TDs remain separate from passing TDs.

Any later candidate must include named-player/equal-player protections, not only pooled QB error.

## 10. If source-blocked

Fail closed.

Do not:
- use sportsbook pass-TD lines as the parent;
- infer pass TD from passing yards with an arbitrary league multiplier;
- use ATD as a proxy;
- lower identity/coverage gates;
- use 2026 results to repair source semantics.

Document the exact missing source and move on.

## 11. Integrity at freeze

- QB passing-TD predictive candidates scored: **0**;
- TD probability/count models fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6/Week-7 outcomes used: **0**;
- sportsbook predictors: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_QB_PASS_TD_MARKET_SPECIFIC_V1_SOURCE_AUDIT_FROZEN`.
