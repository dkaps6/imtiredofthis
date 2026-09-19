# Football Context Role/Room Qualification Checkpoint — 2026-09-19

**Status:** HISTORICAL_QUALIFICATION_EXECUTION_PASS / REDUNDANCY_AUDIT_NEXT  
**Branch at execution:** `research-football-context-execution-v1`  
**Execution SHA:** `6df2b62d5e473793f6dc1ccb1b285420857b1565`  
**Actions run:** `35430064642`  
**Job:** `105862922004`  
**Artifact:** `10580511425` (`football-context-role-room-qualification-v1`)  
**Artifact digest:** `sha256:54c14d4635c0df82e0c4d4097ab1d049e26442a85e00d9be25506a2f19af4b3d`

## Boundary

This checkpoint is engineering/source/QA qualification evidence only. It does not claim predictive lift and does not authorize production integration. No sportsbook data was used. No paid odds pull was made. Issue #535 and failed-closed families were untouched.

## Historical base

The canonical repository history was deterministically rehydrated for 2019–2025 rather than reacquired as a new research source.

- player-games: **37,104**
- unique historical players in usage materialization: **1,468**
- schedule rows: **3,742**
- team-week rows: **3,742**
- player-game SHA256: `33e9ba1e98e7d9057642cca6707925f4fd3ee3b3e7cd3a246e46ced15c7020e8`
- schedule SHA256: `60db4d57a7132b4f00d7f51996dab19b4d171e8e90393f3f95c8fa8b19b14f04`
- team-week SHA256: `ab61d6aeff466aa53b6347829a8bf4796896a76797d78b59e6fd1fdb17f5639a`

The focused gate passed **22 tests** before historical execution.

## Join / integrity result

The role-room join published 37,104 player-game rows with:

- room match rate: **1.0000**
- known-both-context rate: **0.9138**
- stable-ID coverage in candidate profiles: **1.0000**
- duplicate published keys: **0**
- canonical-base fanout: **0**
- candidate-profile unknown rate: **0.086217**

This clears the current identity/fanout integrity requirements for the profiled family.

## Strict-prior stability evidence

Adjacent-period Spearman repeatability across the 2019–2025 player-game history:

| Feature | Spearman | Pairs | Pregame coverage |
|---|---:|---:|---:|
| prior5 rush-share mean | 0.9778 | 31,692 | 0.9604 |
| prior5 target-share mean | 0.9773 | 31,692 | 0.9604 |
| prior3 rush-share mean | 0.9615 | 31,692 | 0.9604 |
| prior3 target-share mean | 0.9500 | 31,692 | 0.9604 |
| prior rush share | 0.8367 | 31,692 | 0.9604 |
| room top-2 target share | 0.8374 | 30,605 | 0.9344 |
| room top-2 rush share | 0.8197 | 30,605 | 0.9344 |
| room top-1 rush share | 0.8183 | 30,605 | 0.9344 |
| room top-1 target share | 0.7444 | 30,605 | 0.9344 |
| prior target share | 0.6996 | 31,692 | 0.9604 |
| returning rush-opportunity overlap | 0.0479 | 13,705 | 0.5150 |
| returning target-opportunity overlap | 0.0302 | 24,772 | 0.7640 |

### Interpretation

The strict-prior rolling usage features are highly persistent and broadly available. Position-room concentration is also materially persistent and broadly available. The returning-opportunity-overlap variables behave differently: they are event/change descriptors rather than persistent player tendencies, and their adjacent-period persistence is intentionally low. Their broad coverage is also thinner, especially rushing overlap.

The low overlap persistence should **not** be interpreted as a scientific failure of the personnel-change concept. It means the generic tendency-stability statistic is a poor semantic fit for a turnover/event feature. Those variables should remain descriptive/event-regime candidates until an event-specific qualification path is frozen; they must not be forced through a persistence gate designed for tendencies.

## Transition diagnostics

Known-context rates for the core fantasy positions are consistently high after the first-game/cold-start states. Examples in 2025:

- QB: 0.9320
- RB: 0.9256
- TE: 0.9308
- WR: 0.9287

2025 transition incidence among known rows:

- RB any transition: **0.6441**; joint player+room transition: **0.0470**
- WR any transition: **0.4472**; joint player+room transition: **0.0399**
- TE any transition: **0.3622**; joint player+room transition: **0.0439**
- QB any transition: **0.4668**; joint player+room transition: **0.0438**

The transition framework therefore identifies a non-trivial but not ubiquitous cohort of games where player usage and/or room structure has changed. The narrower joint player+room cohort is approximately 4–5% for the major positions in 2025 and is a plausible stale-history diagnostic cohort for future preregistered testing.

## Redundancy audit status

A code-level audit of current production confirms the canonical Bayesian baseline already consumes leakage-safe prior-season and current-season aggregate `tgt_share` / `rush_share` evidence (`scripts/modeling/bayesian_v2.py`). Therefore raw usage history is not automatically new information merely because it is materialized under the context program.

However, the new context family contains structures that are not the same object as the current season-aggregate Bayesian inputs:

- rolling 3-game and 5-game strict-prior shares;
- room top-1/top-2 concentration;
- player/room transition diagnostics;
- team-change/current-team-tenure state;
- returning-opportunity overlap.

This establishes structural non-identity, but **does not yet complete the frozen redundancy requirement**. Before any candidate is marked `READY_FOR_FROZEN_EXPERIMENT`, the next pass must quantify how reconstructible the candidate is from the actual canonical pregame production opportunity inputs.

## Current qualification interpretation

No candidate is promoted to predictive science in this checkpoint because the redundancy audit is not complete.

Mechanically:

- the rolling 3/5 usage and room-concentration candidates clear identity, sample, coverage and stability floors;
- returning target overlap misses the broad 0.80 pregame-coverage floor (0.7640);
- returning rush overlap misses it materially (0.5150);
- all profiled candidates have zero duplicate/fanout defects and 1.0 stable-ID coverage.

The next gate is incremental-information / redundancy evidence, not more source acquisition.

## Next executable work

1. Materialize the canonical pregame production opportunity baselines at the same historical player-game grain (especially Bayesian prior/current target and rush share state).
2. Build an outcome-free redundancy audit comparing candidate context fields with those canonical inputs using deterministic association/reconstructibility diagnostics.
3. Add event-specific qualification treatment for one-time transition/turnover indicators so low temporal persistence is not confused with invalidity.
4. Only after that audit, run the frozen qualification inventory and allow `READY_FOR_FROZEN_EXPERIMENT` dispositions.
5. If a candidate clears all gates, freeze a separate predictive experiment before inspecting target outcome lift.

## Disposition

`ROLE_ROOM_HISTORICAL_QUALIFICATION_EVIDENCE_PASS_REDUNDANCY_AUDIT_REQUIRED`
