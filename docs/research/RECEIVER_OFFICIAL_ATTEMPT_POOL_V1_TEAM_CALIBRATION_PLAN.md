# Receiver Official-Attempt Pool V1 — Historical Team Calibration Plan

Date: 2026-09-25

Status: **FROZEN BEFORE HISTORICAL OUTCOME SCORING**

Parent structural result:

`OPPORTUNITY_PARTITION_SEMANTICS_V1_CONFIRMED`

Authority:
- run `36141882752`
- artifact `10867051736`
- result commit `61c6b49b5652eb025d99ab4640658ff17f3d5a18`

This phase does not change player projections. It tests the opportunity anchor only.

## 1. Structural hypothesis

Current receiver simulation allocates targets from projected team dropbacks.

Promoted football semantics define:

`official pass attempts = dropbacks * pass_attempts_per_dropback`

The candidate opportunity anchor is therefore the already-promoted strict-prior
official-attempt conversion.

No new parameter is fit.

## 2. Historical scope

Evaluate separately:

- 2024 regular season
- 2025 regular season
- pooled 2024-2025

Use the canonical leakage-safe historical input builders.

For each target week/team, all forecast inputs must come strictly before that
game. Target-week outcomes may be joined only after both opportunity forecasts
are frozen.

## 3. Forecasts

For each team-game:

Baseline opportunity forecast:

`B = rules_plays_est * rules_pass_rate`

where `rules_pass_rate` is the existing projected dropback rate.

Candidate official-attempt forecast:

`C = B * pass_attempts_per_dropback`

using the strict-prior historical team context already consumed by the promoted
QB stack.

No threshold, clipping rule or multiplier beyond the existing production
conversion is introduced.

## 4. Actual outcomes

After forecasts are frozen, construct:

### Actual official pass attempts
Sum canonical weekly player `pass_att` across all team passers.

### Actual team receiver targets
Sum canonical weekly player `targets` across the team.

### Actual dropbacks
Use target-game historical team/PBP observation:

`actual_dropbacks = actual_offensive_plays * actual_dropback_rate`

This is diagnostic only and confirms the baseline's intended semantic.

## 5. Required provenance

Every scored row must report:

- season
- week
- team
- pregame projected plays
- pregame projected dropback rate
- baseline projected dropbacks
- strict-prior attempt conversion
- conversion source
- candidate projected official attempts
- actual official attempts
- actual receiver targets
- actual dropbacks

If a team-game lacks a legitimate strict-prior attempt conversion, it must remain
baseline/no-change and be reported separately. Do not impute from the target
game.

## 6. Frozen metrics

For baseline and candidate, separately against:

1. actual official pass attempts;
2. actual team receiver targets.

Report by season and pooled:

- n
- MAE
- RMSE
- bias
- absolute bias
- correlation
- median AE
- p75 AE
- p90 AE
- 5+ opportunity miss rate
- 8+ opportunity miss rate
- 10+ opportunity miss rate
- changed-row candidate closer rate

Also report baseline projected dropbacks against actual dropbacks.

## 7. Frozen scientific gates

`RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_SUPPORTED` requires all:

### Official attempts
1. candidate MAE strictly improves in 2024;
2. candidate MAE strictly improves in 2025;
3. candidate pooled MAE strictly improves;
4. candidate pooled p90 AE is nonworse;
5. candidate pooled absolute bias improves;
6. changed-row candidate closer rate > 50%.

### Team targets
7. candidate MAE strictly improves in 2024;
8. candidate MAE strictly improves in 2025;
9. candidate pooled MAE strictly improves;
10. candidate pooled p90 AE is nonworse;
11. candidate pooled absolute bias improves;
12. changed-row candidate closer rate > 50%.

### Provenance/integrity
13. target-game outcomes used upstream = 0;
14. sportsbook inputs = 0;
15. candidate variants scored = 1;
16. parameters fit = 0;
17. strict-prior conversion provenance is explicit for every changed row.

If any gate fails:

`RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_FAILED_CLOSED`

No rescue tuning.

## 8. Interpretation

Passing this team-level calibration does not authorize production.

It authorizes a separately frozen player-level full-stack test where receiver
target allocation uses official attempts instead of dropbacks.

That later candidate must initially leave:

- target shares unchanged;
- M38 unchanged;
- WR-R15 unchanged;
- TE-R5P unchanged;
- catch rates unchanged;
- YPT unchanged;
- QB distributions/means unchanged;
- rushing unchanged;
- residual target semantics unchanged;
- sportsbook independence unchanged.

## 9. No-rescue rule

Do not try after seeing results:

- alternate attempt-conversion windows;
- targetable-pass multipliers;
- position carveouts;
- team carveouts;
- QB carveouts;
- pressure-based conversion;
- scramble-specific target adjustments;
- sportsbook-conditioned logic;
- combined rushing repair;
- hierarchical reconciliation.

Any such idea must be a separate hypothesis with its own frozen validation plan.
