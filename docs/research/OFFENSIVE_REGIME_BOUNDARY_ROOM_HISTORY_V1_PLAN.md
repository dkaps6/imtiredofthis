# Offensive Regime Boundary Room History V1 — Frozen Plan

Status: **FROZEN BEFORE SCORING**

Parent diagnostic:
`WR_ROOM_REGIME_INSTABILITY_AUDIT_V1`

Parent failed candidate remains:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

This is a separate architecture hypothesis. It is not a rescue by excluding 2024, changing a rolling window, adding recency/shrinkage, or routing WR around the failed result.

At freeze time:

- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs: **0**
- 2026 outcomes used: **0**
- production mutations: **0**

## Scientific motivation

The diagnostic found that the old team's prior-season room history becomes materially less trustworthy when the offense crosses an identity boundary.

The cleanest structural marker with historical coverage is primary-QB continuity:

- the prior-season primary QB being absent from the target-week pregame roster had a same-direction relationship with Receiver Room Targets-Per-Play harm in 2022, 2023, 2024 and 2025;
- verified play-caller change independently replicated in 2024 and 2025, but the repo does not currently have equally complete pre-2023 cross-season play-caller coverage.

Therefore V1 intentionally freezes the **minimal historically testable mechanism**: a hard primary-QB regime boundary.

Play-caller information is corroborating evidence only in V1. It is not used by the candidate because doing so would prevent an untouched historical replication with the current frozen source coverage.

## Hypothesis

> Prior-season team receiver-room targets-per-play belongs to the same predictive regime only when the prior-season primary QB is still present on the target-week pregame roster.

When that hard identity continuity is broken, history before the boundary should not automatically remain in the team room-rate denominator.

This is a **history segmentation hypothesis**, not a weight-tuning hypothesis.

## Frozen pregame regime boundary

For team `t`, target season `s`, target week `w`:

1. Compute the prior-season primary QB from season `s-1` completed regular-season player logs:
   - position = QB;
   - group by team/player;
   - sum official player passing attempts already present in the canonical normalized historical logs;
   - highest pass-attempt total is the frozen prior primary QB;
   - deterministic player-key tie break.

2. Read the target-week leakage-safe pregame roster universe already built from nflverse weekly rosters.

3. Define:

`qb_regime_break(t,s,w) = 1`

iff the prior-season primary QB is **not** on team `t`'s target-week pregame QB roster.

Otherwise:

`qb_regime_break(t,s,w) = 0`.

No depth result, target-week participation, target-week pass attempts or target-week outcome may define this flag.

No threshold is fitted.

## Frozen room-rate construction

Rooms remain the existing three:

- WR
- TE
- RB_FB

The boundary rule applies to **all three rooms**. There is no WR-only routing.

### Stable regime

If `qb_regime_break = 0`:

use the exact parent targets-per-play history semantics:

`R_g = sum(room targets from prior season + strict-prior current-season team games) / sum(offensive plays over those same games)`

### Broken regime

If `qb_regime_break = 1` and at least one current-season game exists strictly before the target week:

`R_g_boundary = sum(room targets from strict-prior current-season team games only) / sum(offensive plays from those same games only)`

All pre-boundary prior-season team observations are excluded.

### Week 1 / zero post-boundary-history fallback

If `qb_regime_break = 1` and no current-season team game exists strictly before the target week:

use one frozen, leakage-safe league prior:

`R_g_league_prior = sum(room targets across all teams in prior season) / sum(offensive plays across all teams in prior season)`

This is not blended with the team prior.

It is a deterministic zero-history fallback and has no fitted weight.

### Candidate room target

For every room:

`candidate_room_targets = projected_offensive_plays * R_g_regime_safe`

where `R_g_regime_safe` is the stable-regime rate, broken-regime current-only rate, or frozen league-prior fallback defined above.

The projected offensive-play source is unchanged from the parent architecture.

## What is explicitly NOT allowed

Do not:

- choose a different history length;
- tune a minimum current-games threshold;
- blend current and prior history;
- shrink to league mean with a fitted weight;
- add exponential recency;
- add a WR-only exception;
- add a 2024 exception;
- use play-caller as a second candidate variant;
- add WR roster turnover;
- add bias offsets;
- add fixed57 blending;
- change WR-R15 or TE-R5P specialist ordering;
- use sportsbook information;
- use 2026 outcomes;
- search multiple boundary definitions after seeing scores.

There is exactly one frozen candidate.

## Clean-validation problem and solution

The structural hypothesis was discovered by examining 2022-2025 outcomes.

Therefore **2022-2025 are not an untouched confirmation set for this candidate**.

They may be used later only as already-seen mechanism context, not as evidence that the newly frozen candidate is independently validated.

### Stage A — untouched historical falsification

First score the frozen candidate on **2020 and 2021 only**.

These seasons were not used in the regime-instability audit that generated the candidate.

Regular-season weeks:

- 2020: Weeks 1-17
- 2021: Weeks 1-18

This is a reverse-time independent falsification screen, not a prospective confirmation.

If Stage A fails the frozen structural gates, close V1 immediately.

### Stage B — second untouched historical replication

Only if Stage A passes, run the **unchanged** frozen candidate on **2018 and 2019 only**.

Regular-season weeks:

- 2018: Weeks 1-17
- 2019: Weeks 1-17

No formula, fallback, gate, boundary definition or source may change between Stage A and Stage B.

### Production restriction

Even if both historical stages pass, this architecture is **not automatically eligible for production integration** because it was discovered retrospectively from 2022-2025.

A successful V1 becomes eligible for a frozen prospective 2026 shadow evaluation only.

No 2026 outcome may be used to alter the candidate.

## Frozen Stage-A scorecard

Compare three arms descriptively:

1. current production room baseline;
2. original parent Receiver Room Targets-Per-Play rate;
3. frozen regime-boundary candidate.

The parent is a comparator, not another candidate variant.

Primary qualification is candidate vs production baseline. The structural mechanism is also checked candidate vs parent on the boundary cohort.

### Required gates

All must pass:

1. pooled macro MAE improves vs production baseline;
2. 2020 macro MAE improves vs production baseline;
3. 2021 macro MAE improves vs production baseline;
4. pooled WR MAE improves vs production baseline;
5. 2020 WR MAE improves vs production baseline;
6. 2021 WR MAE improves vs production baseline;
7. pooled TE MAE nonworse vs production baseline;
8. pooled RB_FB MAE nonworse vs production baseline;
9. pooled macro p90 nonworse vs production baseline;
10. pooled macro absolute bias nonworse vs production baseline;
11. summed-room MAE improves vs production baseline;
12. summed-room p90 nonworse vs production baseline;
13. candidate closer rate > 50%;
14. on `qb_regime_break=1` WR rows, candidate MAE is lower than the unchanged parent targets-per-play candidate;
15. on `qb_regime_break=1` pooled all-room rows, candidate MAE is lower than the unchanged parent;
16. stable-regime rows are numerically identical to the unchanged parent rate construction;
17. target-game outcomes used upstream = 0;
18. sportsbook inputs = 0;
19. parameters fit = 0;
20. candidate variants scored = 1;
21. room rates finite and in [0,1];
22. summed room rate <= 1;
23. QB boundary source coverage = 100% for scored team-games.

No gate may be relaxed after the run.

## Stage-B rule

If and only if all Stage-A gates pass:

- freeze the exact Stage-A code hash;
- run 2018-2019 unchanged;
- apply the same gates with year labels changed to 2018/2019;
- no repair is allowed except a mechanical source/plumbing repair that leaves formula, cohort, fallback and gates unchanged.

## Dispositions

Stage A:

- `OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_SUPPORTED`
- or `OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_FAILED_CLOSED`

Stage B if reached:

- `OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_REPLICATED`
- or `OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_B_FAILED_CLOSED`

A historical replication disposition still does not authorize production integration.

## Required Stage-A artifacts

- `room_detail_2020_2021.csv`
- `boundary_audit_2020_2021.csv`
- `season_summary.csv`
- `boundary_summary.csv`
- `source_coverage.csv`
- `summary.json`
- `RESULT.md`

The artifact must state the exact branch head and candidate count.
