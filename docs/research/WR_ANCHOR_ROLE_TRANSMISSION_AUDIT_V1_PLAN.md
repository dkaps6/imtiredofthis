# WR Anchor / Role-Transmission Audit V1 — Diagnostic Plan

Status: **FROZEN DIAGNOSTIC-ONLY PLAN**

Parent closures:
- `RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`
- `OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_FAILED_CLOSED`

This audit moves to a different architecture layer. It does not modify team WR-room mass and does not score a new projection candidate.

At freeze:
- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs: **0**
- production mutations: **0**

## Question

Does leakage-safe current receiver participation/role information already exist before kickoff but fail to propagate through the WR hierarchy because the M38 WR1 entitlement is frozen while WR-R15 only redistributes WR2+ mass?

This is a transmission audit, not another WR1 current-state candidate.

The already-closed WR1-only fixed-room candidate is not reopened.

## Authority

Use the frozen WR-R15 OOS confirmation authority only:

- run `34238301577`
- artifact `10061328722`
- disposition `WR_R15_WR1_ANCHORED_PARTICIPATION_OOS_PASS`
- confirmation seasons: 2023 and 2024
- train 2022 -> test 2023
- train 2023 -> test 2024

Inputs:
- `wr_r15_confirmation_predictions.csv`
- the exact nflverse participation loader and strict-prior feature builder already used by WR-R15

No refit and no new target projection may be constructed.

## Frozen identities per team-game

For every WR team-game in 2023-2024:

1. **M38 anchor**
   - the frozen baseline WR rank 1 identity.

2. **Strict-prior participation leader**
   - WR with the highest `prior1_same_team_offense_pct` from games strictly before the target game;
   - require at least one finite same-team prior observation;
   - deterministic player-key tie break.

3. **Final WR-R15 entitlement leader**
   - WR with the highest already-scored `entitlement_tgt_share` in the authorized WR-R15 candidate output.

4. **Actual top-target set**
   - target-game actual targets are labels only;
   - all players tied at the maximum actual targets belong to the top-target set.

## Diagnostic cohorts

Primary cohort:
`strict-prior participation leader != M38 anchor`.

Control:
`strict-prior participation leader == M38 anchor`.

Report separately by 2023, 2024 and pooled.

## Frozen measurements

No new candidate is scored.

Measure:

- rate of anchor/participation-leader mismatch;
- whether WR-R15 final entitlement leader follows the participation leader;
- M38-anchor hit rate against actual top-target set;
- strict-prior participation-leader hit rate against actual top-target set;
- final WR-R15 leader hit rate against actual top-target set;
- actual-target difference: participation leader minus M38 anchor;
- baseline vs authorized WR-R15 total per-team WR target absolute-error sum;
- mismatch vs match error concentration;
- baseline and final entitlement assigned to anchor and participation leader;
- how much final entitlement rank changes even though the M38 anchor itself is immutable;
- source coverage / duplicate / future-row audit.

## Frozen structural criteria

A separate new hierarchy-state hypothesis is warranted only if **all** are true:

1. pooled anchor/participation mismatch cohort contains at least 150 team-games;
2. on mismatch games, participation-leader actual-top hit rate exceeds M38-anchor hit rate by at least 5 percentage points pooled;
3. that participation-leader advantage is nonnegative in both 2023 and 2024;
4. on mismatch games, mean actual targets(participation leader - anchor) is positive pooled and nonnegative in both seasons;
5. WR-R15 final leader follows the participation leader in fewer than 50% of mismatch games;
6. mismatch-game final WR target absolute-error sum is at least 5% worse than match-game final WR target absolute-error sum after normalizing per WR row;
7. strict-prior participation future violations = 0;
8. sportsbook inputs = 0;
9. candidate variants scored = 0.

If these do not all hold, close the immutable-anchor transmission hypothesis. Do not rescue it with alternate snap windows, target-share leaders, depth ranks, thresholds, WR1 exceptions, or 2026 outcome fitting.

## Interpretation safeguards

Passing this diagnostic would **not** authorize changing the M38 anchor.

It would only justify freezing a separate candidate before any scoring.

Failure means the anchor is not the missing transmission bottleneck, and research must move to another new-information architecture.

## Required artifacts

- `team_game_anchor_transmission.csv`
- `player_detail_anchor_transmission.csv`
- `cohort_summary.csv`
- `season_summary.csv`
- `source_audit.json`
- `summary.json`
- `RESULT.md`
