# Player Active-Role Participation V1 — Mechanism Freeze

Date: 2026-10-16

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY**

Version:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE`

Parent source authority:
- disposition `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`;
- run `38895205962`;
- job `116953914726`;
- exact source head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`;
- artifact `12044155086`;
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`.

This file freezes the predictive mechanism **before any participation score is
computed**.

## Scientific question

> Does this named player's own strictly-prior participation history forecast
> whether he records any offensive snap in the target game better than a
> strictly-prior position-level participation baseline?

This is Stage 1 only:

`ACTIVE ROLE / ZERO-STATE PARTICIPATION PROBABILITY`.

Stage 2 remains separate:

`CONDITIONAL OPPORTUNITY ALLOCATION GIVEN PARTICIPATION`.

No Stage-2 opportunity amount, carries, targets, yards, TD outcomes, market
line, sportsbook price or target-game stat may enter this mechanism.

## Historical seasons and independence

Score:
- 2024 — development/mechanism evidence;
- 2025 — **independent confirmation season**.

The probability mapping, support gates, score metrics, bootstrap design and
qualification gates in this document are frozen before either season is scored.

No 2026 outcomes are permitted.

## Pregame universe and binary target

Use the exact parent source contract.

For each scheduled team-week:

1. weekly roster authority;
2. status `ACT` where status is available;
3. stable player identity;
4. map to QB / RB / WR / TE family;
5. remove the exact official game-day inactive identities from the frozen M78
   snapshot;
6. require a valid team-week PFR offensive-snap source;
7. retain every remaining player, including a player who ultimately records
   zero offensive snaps.

Target:
- `y = 1` if the player records any offensive snap;
- `y = 0` if he records zero offensive snaps in a valid target team-week.

A missing player snap row is zero only when the target team-week snap source is
valid.

Target-game snaps are an outcome only. They may not decide pregame membership
or construct any predictor.

## Position separation

Adjudicate independently:
- QB;
- RB = RB/FB/HB;
- WR;
- TE.

No pooling of positions for a pass decision.

## Strictly-prior chronology

All predictive information for target season `s`, target week `w` must have
`prior_week < w`.

Do not use kickoff ordering inside the same NFL week. Same-week games are not
prior evidence.

Player history additionally requires:
- same stable player identity;
- same current target team;
- same position family;
- prior row from the exact eligible pregame universe.

A trade/team change therefore resets same-team player history. If the player
does not yet have three eligible games with his current target team, the
player-specific candidate is unsupported.

## Frozen baseline probability

For each target row, define the position baseline from **all eligible prior
player-games in the same season and position family with week < target week**.

Let:
- `K_pos` = prior positive-participation player-games;
- `N_pos` = prior eligible player-games.

Use Jeffreys smoothing:

`p_baseline = (K_pos + 0.5) / (N_pos + 1.0)`.

Baseline support requires:
- `N_pos >= 100`.

The baseline is recomputed strictly walk-forward for every target week.

## Frozen named-player state

For the target player/team, take the most recent **up to four** eligible
same-team player-games with week < target week.

Let:
- `n_player` = number of available prior eligible games, maximum 4;
- `k_player` = number of those games with any offensive snap.

Candidate support requires:
- `n_player >= 3`.

Do not search alternate windows.
Do not substitute all-team career history.
Do not use target-game information.
Do not use snap percentage magnitude; only prior participation occurrence
(`0/1`) is used in V1.

## Frozen candidate probability mapping

Use one fixed weak position-anchored prior with effective sample size 2.

`p_candidate = (k_player + 2 * p_baseline) / (n_player + 2)`.

This is not fit or tuned.

At the minimum three player games, player history carries 3/5 of the probability
mass; with four games it carries 4/6. The position prior prevents unsupported
0/1 certainty while leaving the named player's recent same-team participation
history as the majority signal.

No alternative prior strength will be tested inside V1.

## Evaluation row identity

Compare baseline and candidate on the **same exact supported target rows**:
- valid pregame universe;
- valid target snap source;
- stable player identity;
- baseline `N_pos >= 100`;
- player same-team `n_player >= 3`.

Unsupported rows are not silently dropped from the production universe; they
are simply not eligible to test the V1 player-specific mechanism.

If V1 later qualifies for deployment, unsupported players must retain an
explicit baseline/fallback path. This freeze does not authorize production.

## Frozen scoring metrics

Primary:
- Brier score.

Define:
- `delta_brier = Brier_baseline - Brier_candidate`;
- positive is better for the named-player candidate.

Guardrail:
- Bernoulli log loss.

Define:
- `delta_logloss = LogLoss_baseline - LogLoss_candidate`;
- positive is better.

Because both frozen probability mappings are interior to (0,1), no probability
clipping or post-hoc calibration is permitted.

Also report, but do not optimize:
- observed participation rate;
- mean predicted baseline probability;
- mean predicted candidate probability;
- calibration-in-the-large error;
- target rows;
- distinct players;
- positive outcomes;
- zero outcomes.

No thresholded accuracy, F1, or classification cutoff is a qualification metric.

## Named-player heterogeneity requirement

The aggregate position result is not sufficient.

For each position/season, among players with at least five scored target rows,
compute each player's own:

`player_delta_brier = player_Brier_baseline - player_Brier_candidate`.

Report:
- number of evaluable named players;
- equal-player mean `player_delta_brier`;
- median `player_delta_brier`;
- share of named players with positive `player_delta_brier`.

The **2025 equal-player mean improvement** is a qualification gate.

This prevents a small number of high-row players from manufacturing a
position-average win.

## Dependence-aware inference

Use a fixed percentile cluster bootstrap.

Replicates:
- 5,000.

Seed:
- `20261016`.

Confidence interval:
- 95%, percentile [2.5%, 97.5%].

For each position and relevant season/pool compute Brier-delta intervals under:

1. **player cluster bootstrap**
   - resample stable player IDs with replacement;
   - retain all scored rows for each sampled player;
   - repeated sampled clusters receive repeated weight.

2. **game cluster bootstrap**
   - resample unique target game IDs with replacement;
   - retain every scored player row belonging to each sampled game;
   - repeated sampled games receive repeated weight.

For the pooled 2024+2025 diagnostic, cluster keys must be season-qualified.

Also compute a 2025 **equal-player bootstrap**:
- one per-player Brier delta per evaluable named player;
- resample players equally with replacement;
- 5,000 replicates, same seed;
- 95% percentile CI.

No iid row bootstrap may substitute for these inference gates.

## Frozen support gates

All must pass separately for both 2024 and 2025 before a position can qualify.

QB:
- scored target rows >= 350;
- distinct scored players >= 45;
- positive outcomes >= 100;
- zero outcomes >= 100;
- evaluable named players with >=5 scored rows >= 30.

RB:
- scored target rows >= 800;
- distinct scored players >= 80;
- positive outcomes >= 500;
- zero outcomes >= 75;
- evaluable named players with >=5 scored rows >= 55.

WR:
- scored target rows >= 1,200;
- distinct scored players >= 140;
- positive outcomes >= 900;
- zero outcomes >= 50;
- evaluable named players with >=5 scored rows >= 90.

TE:
- scored target rows >= 700;
- distinct scored players >= 75;
- positive outcomes >= 550;
- zero outcomes >= 25;
- evaluable named players with >=5 scored rows >= 50.

If a support gate fails, that position is **mechanism source-blocked at scoring**.
Do not lower the gate.

## Frozen qualification gates

Adjudicate each position separately.

A position qualifies only if **all** of the following are true:

### 2024 development evidence
1. support gates pass;
2. `delta_brier > 0`;
3. `delta_logloss >= 0`.

### 2025 independent confirmation
4. support gates pass;
5. `delta_brier > 0`;
6. `delta_logloss >= 0`;
7. 2025 player-cluster 95% CI lower bound for `delta_brier` is strictly > 0;
8. 2025 game-cluster 95% CI lower bound for `delta_brier` is strictly > 0;
9. 2025 equal-player mean `player_delta_brier > 0`;
10. 2025 equal-player bootstrap 95% CI lower bound is strictly > 0.

### pooled corroboration
11. pooled 2024+2025 `delta_brier > 0`;
12. pooled player-cluster 95% CI lower bound is strictly > 0;
13. pooled game-cluster 95% CI lower bound is strictly > 0.

The 2025 independent-confirmation gates are mandatory. A pooled win cannot
rescue a 2025 failure.

## Frozen dispositions

Per position:

Qualified:
- `QB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `RB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `WR_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `TE_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`

Source/support blocked while scoring:
- matching `*_ACTIVE_ROLE_PARTICIPATION_MECHANISM_SOURCE_BLOCKED`

Scientifically closed:
- matching `*_ACTIVE_ROLE_PARTICIPATION_MECHANISM_CLOSED`

Integrity failure:
- matching `*_ACTIVE_ROLE_PARTICIPATION_MECHANISM_INTEGRITY_FAILURE`.

Overall disposition is descriptive only; every position keeps its own decision.

## If V1 closes

Do not rescue V1 with:
- a different history window;
- two-game or five/eight-game windows;
- snap-percentage thresholds;
- depth-chart subsets;
- starter-only subsets;
- alternate prior strengths;
- isotonic/logistic calibration;
- target-game weekly-stat presence;
- position pooling;
- TD outcomes;
- sportsbook/market variables.

Record the result and move to a genuinely different player-state information
family.

## If a position qualifies

Qualification authorizes only a separate future deployment/bridge freeze for
Stage 1.

It does **not** automatically:
- mutate production projections;
- remove a player from the universe;
- zero opportunity;
- alter WR-R15 or TE-R5P;
- alter the Week-6 WR/TE trust locks;
- authorize Stage 2;
- certify ATD.

Any deployment mapping from `p_active_role` into opportunity generation must
be frozen and tested separately.

## Protected continuity

Preserve:
- Player Projection First Principle;
- WR opportunity-difficulty trust qualification and frozen Week-6 WR lock;
- TE trust qualification and frozen Week-6 TE lock;
- all validated production science;
- all terminal anti-retest lanes in Issue #535;
- sportsbook separation;
- Week-6 OddsAPI guardrail.

## Integrity boundary

This frozen mechanism uses:
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0.

At freeze time:
- participation predictors scored: 0;
- candidate models fit: 0;
- projection values changed: 0;
- production changed: false.

Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
