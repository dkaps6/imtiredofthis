# All-Player / All-Position Replay V1 — Frozen Plan

**STATUS: FROZEN BEFORE REPLAY EXECUTION / RESEARCH ONLY / NO PRODUCTION CHANGE**

## Objective

Run one leakage-safe 2026 Weeks 1-4 replay across the complete pregame football roster universe for:
- QB
- RB/FB
- WR
- TE

The purpose is to measure the current player-centric stack as a whole before any further feature search.

This is **not** a sportsbook-board-only replay. Sportsbook rows are downstream reconciliation evidence only.

## Position state entering replay

### QB
Use current protected QB science unchanged.
Do not reopen generic QB mean.

Protected authorities include:
- M89/M90 passing-yard synthesis;
- QB C2 mean-neutral distribution selector;
- current production starter/role semantics.

### RB
Frozen phase-end disposition:
`RB_PLAYER_LEVEL_SCOPE_BUTTONED_UP_BASELINE_PLUS_PROSPECTIVE_ALLOCATION`

For 2026 Weeks 1-4:
- production-authorized RB stack only;
- do not back-apply the Week-5 carry/snap allocation shadow;
- do not reopen M96;
- do not add receiving-mean or width treatments.

The Week-5 room-allocation shadow remains prospective only.

### WR
Baseline:
- M38
- WR-R15

Player-state additions:
- Target Share Trajectory V1 is **not legally available** in 2026 Weeks 1-4 because the frozen contract requires four prior same-season team games.
- Player Target Depth Distribution Shadow V1 **is eligible** when its exact strictly-prior receiver-history gate is satisfied.

### TE
Baseline:
- TE-R5P

Player-state additions:
- Target Share Trajectory V1 is not legally available in 2026 Weeks 1-4.
- Player Target Depth Distribution Shadow V1 is eligible under its exact strictly-prior receiver-history gate.

## Player Target Depth Distribution Shadow authority

Contract:
`docs/research/PLAYER_TARGET_DEPTH_DISTRIBUTION_SHADOW_V1_CONTRACT.md`

Week-5 lock:
`docs/research/PLAYER_TARGET_DEPTH_DISTRIBUTION_SHADOW_V1_WEEK5_LOCK.md`

Run:
`37677366697` SUCCESS

Artifact:
`11507970080`

Lock row digest:
`sha256:1a5b6a914517310cdfb0fe5c92fcba7a77a28b6a988664c0f7dcc49cd2b8eb2a`

Frozen transform:
`depth_distribution_scale = sqrt(prior8_target_depth_sd / position_anchor)`

Exact anchors:
- WR `10.02786868561449`
- TE `6.735519692444827`

The transform:
- fits zero coefficients;
- preserves exact receiving-yard mean;
- preserves target entitlement;
- preserves team target/pass volume;
- uses no sportsbook input.

## Replay universe

Primary replay population:

Every pregame football player row in the canonical skill-position universe for the requested week, subject to the same football availability/identity rules used by the historical reconstruction.

This means the replay is not conditioned on whether a sportsbook offered a prop.

Required position families:
- QB
- RB/FB
- WR
- TE

Required markets for player-game scoring where defined:
- QB: pass_yards
- RB/FB: rush_yards, rec_yards, receptions, rush_rec_yards
- WR: rec_yards, receptions
- TE: rec_yards, receptions

Other production markets may be retained as diagnostics but are not required to answer the player-individualization question.

## Pregame / leakage boundary

For target 2026 week W:
- no target-week game result may enter any feature;
- completed prior 2026 weeks may enter only through already-authorized strictly-prior state;
- 2025 may seed returning-player history where the frozen feature contract allows it;
- no future 2026 game may enter;
- target-week actuals are joined only after projection/distribution state is frozen.

The replay must record explicit max feature season/week lineage for every newly evaluated player-state feature.

## Exact trajectory rule

Do not weaken Target Share Trajectory V1.

Weeks 1-4 exact status:
- W1: ineligible
- W2: ineligible
- W3: ineligible
- W4: ineligible

Reason:
fewer than four prior same-season team games.

Trajectory is therefore recorded as:
`FROZEN_FEATURE_NOT_YET_ELIGIBLE`

Historical Week-5+ remains the correct integration-test population for that transform.

## Target-depth replay rule

Target-depth dispersion may use prior 2025 history for returning WR/TE players.

Exact feature:
- individual receiver;
- regular-season target events;
- latest up to 8 completed receiver target-games;
- at least 4 prior receiver target-games;
- at least 10 finite air-yard targets;
- population SD of finite air_yards;
- strictly before target game.

Unavailable players receive scale `1.0`.

No alternate minimums/windows/fallback formula may be searched after outcomes are visible.

## Distribution reconstruction

For WR/TE receiving yards, the replay must operate on the empirical football draw array, not a Normal approximation.

Required sequence:
1. reconstruct canonical baseline empirical football draws under the target-week pregame cutoff;
2. align the baseline draws to the exact final football receiving-yard mean under existing production semantics;
3. apply the frozen target-depth distribution transform;
4. clip at zero;
5. re-align to the same exact mean;
6. score baseline and shadow against the same realized outcome.

The replay may not infer empirical draws from `model_sd` alone.

## Point-mean scoring

Primary player-individualization scoreboard:
- MAE
- median absolute error
- signed bias
- RMSE
- model-closer-than-live-selected-line on overlapping preserved live-board rows only

Report by:
- week
- position
- market
- player
- player-week
- opportunity tier where already defined by production state

Do not invent a new selection threshold from these results.

## Distribution scoring

For WR/TE receiving yards only, paired baseline vs target-depth shadow:

Primary:
- empirical CRPS

Required:
- 50% central interval coverage + width
- 80% central interval coverage + width
- 90% central interval coverage + width
- PIT/rank histogram summary
- absolute realized error by frozen depth-scale quantile
- exact baseline-vs-shadow mean gap

The point-mean MAE must be invariant for the target-depth shadow and is an integrity check.

## Preserved live-board reconciliation

Weeks 1-4 frozen production-board results are **not** the replay universe.

They are used only to check reconstruction overlap and preserve factual live history.

Authorities:
- W1/W2 durable board/graded authority in PR #626 / committed market-track-record files;
- W1/W2 full-board run `35740684289`, artifact `10699408992`;
- W1-W3 postmortem run `36622608145`, artifact `11059171429`;
- W4 expanded postmortem run `37485600879`, artifact `11423535484`;
- W4 recovered live board artifact `11197776900`.

Closed factual betting record remains:
- W1 204-205
- W2 192-198
- W3 233-208
- W4 220-207
- cumulative 849-818, 50.93%, -44.10u

This record is not refit and is not overwritten by the all-player replay.

## Required output partitions

1. `all_players_point_scoreboard.csv`
   - one row per player/week/market with frozen pregame mean + actual.

2. `wr_te_rec_yards_distribution_scoreboard.csv`
   - baseline vs target-depth shadow paired empirical metrics.

3. `feature_eligibility_audit.csv`
   - exact trajectory eligibility;
   - target-depth support;
   - lineage cutoff.

4. `live_board_overlap_audit.csv`
   - overlap with preserved live-board rows only;
   - no sportsbook fields used upstream.

5. `replay_summary.json`
   - counts, parity, leakage guards, position/market summaries.

## Interpretation gate

This replay is diagnostic.

It may answer:
- where player-level mean projection is working/failing;
- whether the frozen WR/TE player-specific uncertainty transform improves distribution quality;
- whether current position-level sharing remains too strong.

It may **not** automatically:
- promote a new coefficient;
- create a betting threshold;
- reopen closed science;
- turn Week-1-4 outcomes into a new RB router;
- weaken trajectory eligibility;
- refit the target-depth scale.

Any new science after this replay requires a new hypothesis and frozen pre-outcome contract.

No paid OddsAPI pull is authorized or required.
