# TE-R4 Strict-Prior Participation Source Audit — Frozen Plan

## Purpose
TE-R3 showed that team TE target-pool context can materially improve target forecasts but still fail the full individual receiving-yard objective, in part because the correction remained almost universally positive. The next genuinely new information family is **individual participation/role**.

Before any new predictive model, TE-R4 audits whether nflverse snap-count data can provide sufficiently broad, leakage-safe strict-prior TE offensive participation history.

This is a source/availability audit only. No outcome model is fit and production is unchanged.

## Lineage
- Parent TE-R3 result commit: `16747b8a0194dceaabd280d131e22d3281530d94`.
- TE-R3 canonical run: `34126813280`; disposition `TE_TARGET_POOL_CONTEXT_MODEL_FAIL`.
- Exact target universe: TE rows from Joint Pass/Receiving Conservation V1 run `34081764151`, artifact `10004223287`.
- Snap source: `nflreadpy.load_snap_counts(seasons=[2020,...,2025])`.
- Sportsbook inputs: 0.

## Target cohort
Regular-season TE player-games from 2021-2025 in `joint_v1_paired_player_casebook.csv`. 2020 snap data are source history only so 2021 target rows can have prior observations.

## Frozen source fields
- `offense_pct` (or `offense_percentage` alias)
- `offense_snaps`

Identity key: normalized player name plus canonicalized team for same-game source matching. Player-history availability is also reported across team changes using normalized player identity alone.

## Strict-prior definitions
For each target TE player-game at ordinal `season*100 + week`:
- `prior1_anyteam`: most recent snap observation for the player with ordinal strictly less than target ordinal;
- `prior3_anyteam`: at least three strictly earlier player snap observations;
- `prior1_same_team`: most recent strictly earlier snap observation for the same player and target team;
- `prior3_same_team`: at least three strictly earlier same-team observations.

The target game's own snaps and all future snaps are forbidden.

## Frozen outputs
Pooled and by target season:
- target player-games;
- prior1/prior3 availability any-team;
- prior1/prior3 availability same-team;
- field-level non-null availability for `offense_pct` and `offense_snaps` under each history definition;
- median number of prior snap observations;
- median number of same-team prior observations.

Source integrity:
- snap rows and seasons present;
- duplicate rate on `(season, week, team, player_key)`;
- exact count of same/future observations used (must be zero);
- same-game exact-key source match rate reported diagnostically only; it is not required for prior-history eligibility.

## Frozen gates
`STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE` requires all:
1. source contains all six seasons 2020-2025;
2. duplicate rate <=1%;
3. pooled prior1 any-team availability >=75%;
4. pooled prior3 any-team availability >=60%;
5. every target season prior1 any-team availability >=65%;
6. both snap fields have >=70% pooled prior1 non-null availability;
7. pooled prior1 same-team availability >=55%;
8. zero same/future observations used;
9. sportsbook inputs = 0.

If integrity is intact but one or more coverage gates fail -> `STRICT_PRIOR_TE_PARTICIPATION_PARTIAL_ONLY`.
If source/schema/integrity fails -> `STRICT_PRIOR_TE_PARTICIPATION_INELIGIBLE`.

## Interpretation
A source pass does not imply predictive value. It only authorizes a separately frozen TE-R5 player-role experiment. No target-game snap share may ever be used to predict that same game.

## Authorized TE-R5 if eligible
A player-centric target/yard experiment may combine the already-researched team TE pool with strict-prior individual snap share, room competition, player target history, and opponent matchup context. It must evaluate full individual projection metrics and cannot rescue TE-R3 by post-hoc Q4 filtering or generic recentering.
