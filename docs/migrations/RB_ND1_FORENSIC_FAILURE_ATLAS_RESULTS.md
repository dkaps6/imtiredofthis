# RB-ND1 — Forensic Failure Atlas Results

## Disposition

**ADVANCE_RB_ND2_PREGAME_ROLE_STATE_PLAYER_SHARE_RECONSTRUCTION**

RB-ND1 does not promote a football model. It reverse-engineers the 2025 RB rushing-yard errors and identifies the next justified new-data research family.

## Authoritative run

- workflow: `RB-ND1 Forensic Failure Atlas`
- run: **`33503240202`**
- job: **`99841197836`**
- tested SHA: **`1b689322b48e7de52530bca5a9e2d7039a7a1a9b`**
- artifact: `rb-nd1-forensic-failure-atlas`
- artifact ID: **`9798563163`**
- artifact SHA256: **`d92ccd3fdc36edf523ed6da3af3c76e227e2b5a5ae8c9a49680b6661a53c7d7c`**
- execution: success
- sportsbook inputs to football model: `0`
- model fitting/search: `0`
- production change: `0`

## Coverage

- frozen M94C RB/FB player-games: `1,393`
- market-covered player-games: `899`
- M96E authoritative W6-18 rows: `961`
- M96E + market W6-18 common rows: `633`
- PBP player-rush matches: `1,244`
- PBP team-game matches: `1,393`
- same-week historical injury-report matches: `52`
- nflverse PBP rows: `46,452`
- rush rows: `14,200`
- player-rush groups: `2,193`
- team-games: `544`

## Primary decomposition

### Carries = team rush attempts × player team-rush share

Exact two-factor Shapley absolute contribution totals:

- team-rush-volume contribution: `2,554.2161` — **`39.87%`**
- player-share/backfield-allocation contribution: `3,851.5330` — **`60.13%`**

This is the most important new opportunity finding: **the larger source of carry movement/error is player share/backfield allocation, not team rush volume.** M94C's team-level volume work remains useful, but the player allocation layer is the larger unresolved opportunity problem.

### Rushing yards = carries × YPC

Exact two-factor Shapley absolute contribution totals:

- opportunity/carry contribution: `20,427.2343` — **`51.51%`**
- efficiency contribution: `19,231.4876` — **`48.49%`**

This independently confirms M96A: rushing-yard error is almost perfectly joint between opportunity and efficiency. The new detail is that, within opportunity, **player allocation is the larger carry subproblem**.

## Error atlas

Largest compound mechanism classes by share of all M94C absolute rushing-yard error:

1. `OPPORTUNITY__PLAYER_SHARE`: n `512`, MAE `18.10`, **31.63% of all absolute error**
2. `EFFICIENCY__TEAM_VOLUME`: n `185`, MAE `28.59`, **18.06%**
3. `EFFICIENCY__PLAYER_SHARE`: n `256`, MAE `20.07`, **17.54%**
4. `OPPORTUNITY__TEAM_VOLUME`: n `135`, MAE `22.51`, **10.37%**
5. `EFFICIENCY__MIXED`: **6.98%**
6. `MIXED__PLAYER_SHARE`: **6.54%**
7. `MIXED__TEAM_VOLUME`: **4.87%**
8. `OPPORTUNITY__MIXED`: **3.07%**
9. `MIXED__MIXED`: **0.95%**

The three classes whose carry-side primary mechanism is `PLAYER_SHARE` account for roughly **55.5% of all absolute rushing-yard error**. This makes current player role/backfield allocation the highest-leverage new opportunity family.

## Special forensic flags

These flags overlap and are not additive attribution shares:

- explosive-run shock: n `157`, MAE `47.31`; those games contain `25.35%` of total absolute error
- substantial non-RB/QB rushing competition: n `329`, MAE `22.04`; `24.75%`
- game-script miss: n `325`, MAE `21.28`; `23.60%`
- role-collapse: n `59`, MAE `32.59`; `6.56%`
- same-week injury-report flag: n `52`, MAE `18.99`; `3.37%`
- new-role initialization flag: n `8`, MAE `41.81`; `1.14%`

Explosive shocks are especially important because they demonstrate why a pure mean-efficiency correction cannot eliminate the right-tail error. They belong in a distribution/tail module, not a blind universal YPC boost.

## M96E versus M94C versus market on the exact common W6-18 listed-RB universe

n = `633`

- M94C MAE: **`25.595447`**
- M96E MAE: **`25.527995`**
- archived DK/FD consensus MAE: **`24.139810`**

Therefore later M96 work **did** improve M94C on an apples-to-apples listed-RB sample, but only by `0.06745` yards/game. The remaining market gap is still about `1.3882` yards/game on this slice. The correct conclusion is not that M96 was useless; it is that its retained efficiency router is too small to solve the dominant role/allocation problem uncovered here.

## Where the market advantage sits

Market advantage versus M94C by forensic mechanism:

- `EFFICIENCY__TEAM_VOLUME`: **`+2.815` MAE yards**
- `OPPORTUNITY__PLAYER_SHARE`: **`+2.446`**
- `EFFICIENCY__PLAYER_SHARE`: **`+2.181`**
- `OPPORTUNITY__TEAM_VOLUME`: `+1.282`
- `MIXED__PLAYER_SHARE`: `+1.232`
- `MIXED__TEAM_VOLUME`: `+1.214`
- `EFFICIENCY__MIXED`: `+0.926`

The market does not dominate every class. M94C beats it in `OPPORTUNITY__MIXED` and especially `MIXED__MIXED`. The new research path should therefore repair specific missing football state rather than replacing the independent projection with market information.

## M96E capability by forensic class

M96E improves M94C most in:

- `EFFICIENCY__PLAYER_SHARE`: `+0.264` MAE yards
- `EFFICIENCY__TEAM_VOLUME`: `+0.221`
- `OPPORTUNITY__PLAYER_SHARE`: `+0.204`
- `MIXED__PLAYER_SHARE`: `+0.116`

It worsens `OPPORTUNITY__MIXED` by `0.179` and `MIXED__TEAM_VOLUME` by `0.290`. Preserve M96E as evidence/capability; do not pretend it solved the point model.

## Large-miss casebook examples

Examples show several distinct mechanisms:

- Jonathan Taylor W10: actual `32` carries / `244` yards vs M94C `14.38` / `79.76`; large team-volume/opportunity + explosive component.
- Rico Dowdle W5: actual `23/206` vs `11.63/46.42`; share/allocation + explosive component.
- Jahmyr Gibbs W12: actual `15/219` vs `15.47/66.08`; workload essentially correct, extreme efficiency/explosive failure.
- James Cook W8: actual `19/216` vs `18.22/70.85`; workload close, extreme efficiency/explosive failure.
- Ray Davis W18: actual `21/151` vs `2.21/17.98`; enormous role/share miss.
- Jerome Ford W1: actual `6/8` vs `15.99/111.80`; canonical false-high role/share failure; archived market line `46.5`.
- James Cook W18: actual `2/15` vs `23.37/104.90`; role-collapse/share failure.
- Jacory Croskey-Merritt W1: actual `10/82` vs `0.84/3.23`; canonical new-role initialization failure.

The model needs different modules for these failure mechanisms rather than one universal correction.

## Critical data-integrity / information finding — M94C had no populated historical depth role

Inspection of the authoritative `m94c_2025_rb_trace.csv` shows:

- `role`: **0 non-null rows out of 1,393**
- `rules_role`: **0 populated rows out of 1,393**

This is consistent with the historical-input contract: nflverse weekly rosters define the player universe, but historical depth data are merged only when the depth source is explicitly week-tagged. Date-based/full-season depth releases are intentionally skipped because a safe pregame cutoff cannot be proven.

Therefore the 2025 RB backtest effectively asked historical usage/context to infer current player role **without a true historical pregame depth-role state**. This is a materially different missing-information problem from the M95/M96 coefficient experiments and strongly aligns with:

- Vegas's advantage on false-high/low-workload games;
- Week-1 rookie/new-team initialization misses;
- role-collapse casebook errors;
- the 60.13% player-share contribution in carry decomposition.

This does not justify using current Ourlads data retrospectively. The fix must be leakage-safe.

## Next migration — RB-ND2 Pregame Role-State / Player-Share Reconstruction

ND2 should keep M94C's team rush-volume layer frozen initially and attack player share/backfield allocation using only pregame-available football data.

Audit/build these families before fitting:

1. lagged prior-game / rolling participation or snap involvement from nflverse participation data;
2. lagged carries, touches, team RB share, red-zone/short-yardage usage and backfield concentration;
3. current-week weekly roster membership/status;
4. same-week injury/practice/game-status data with exact week/time semantics; no forward filling from later information;
5. competing-RB availability/vacancy and backfield member count;
6. team-change/new-team and no-NFL-history/rookie indicators;
7. Week-1 prior-role construction from prior-season team continuity plus rookie/new-team priors;
8. timestamp-safe historical depth-chart source only if provenance proves it was available before that game;
9. QB/non-RB rush-allocation tendency as a separate competition input.

Primary modeling target should be **player share of team rush attempts / backfield allocation**, not raw rushing yards. M94C team rush attempts stay fixed for the first test. Any resulting carry improvement can then be translated through the existing yardage architecture and evaluated against both actual football outcomes and the downstream market benchmark.

ND2 must use precommitted temporal validation and must preserve high-workload 20+/25+ behavior while reducing false-high 0-10 carry cases. Sportsbook information remains downstream only.