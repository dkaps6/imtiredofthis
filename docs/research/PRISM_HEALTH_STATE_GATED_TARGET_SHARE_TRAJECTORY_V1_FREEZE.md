# PRISM Health-State-Gated Target Share Trajectory V1 — Frozen Integration Plan

Date: 2026-10-18

Status: **FROZEN BEFORE CANDIDATE SCORING / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1`

## 1. Scientific question

The existing Target Share Trajectory V1 is qualified and historically supported.

The new observation-state bridge proved that its frozen RECENT2 history can
contain a team-game in which the named player did not have source-certified
positive offensive participation:

- WR 2024: 260 / 1,619 trajectory rows (**16.06%**);
- TE 2024: 100 / 844 (**11.85%**);
- TE 2025: 108 / 865 (**12.49%**).

This V1 asks one narrow player-state question:

> When the already-qualified trajectory signal's RECENT2 window contains at
> least one observation that is not source-certified as positive offensive
> participation, is it better to treat that trajectory delta as **unknown /
> neutral** rather than affirmative evidence of role movement?

This is not a new trajectory formula.

It is a source-state quality gate over the existing frozen mechanism.

## 2. Immutable comparison authority

### Existing trajectory integration

Run:
`37783801061`

Artifact:
`11552284671`

Name:
`player-trajectory-oos-fold-integration-v1`

Digest:
`sha256:6d4e9a2a87fc2b364a847e6328d201f42fa4bf63b10f87416001c4fec8de03d2`

Disposition:
`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED`

Use:
`trajectory_oos_fold_integration_rows.csv`

This file is the exact historical OOS-fold integration authority and already
contains:
- canonical parent targets;
- original frozen trajectory delta;
- original trajectory-integrated targets;
- exact WR anchor flag;
- baseline YPT;
- realized target/receiving-yard outcomes;
- room-share fields.

Do not rebuild WR-R15, TE-R5P, PBP or trajectory state.

### Observation-state bridge

Run:
`39154571787`

Artifact:
`12146118822`

Name:
`prism-target-share-trajectory-observation-state-bridge-v1-39154571787`

Digest:
`sha256:76d60dc56adcce1ac640a0311419af937072b634ff5a0199d46308626641bb84`

Disposition:
`PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_READY`

Use:
`trajectory_observation_state_bridge_rows.csv`.

## 3. Frozen populations

Primary qualification population:
- 2024 Weeks 5+;
- WR + TE;
- exact 2,463 rows from the existing OOS trajectory integration.

Primary directly affected rows:
- rows where
  `recent2_observation_contaminated == true`.

Source inventory before outcome scoring:
- WR: 260 affected rows / 135 affected players;
- TE: 100 affected rows / 48 affected players.

All primary rows remain in the allocation system.

No row is dropped.

Secondary independent TE guardrail:
- 2025 Weeks 5+;
- exact 865 TE rows;
- 108 directly affected rows / 60 affected players.

TE 2025 cannot rescue a failed 2024 primary result.

It can veto a 2024 pass if the same frozen action materially worsens the
independent TE season under the gates below.

## 4. Frozen candidate action

Baseline comparator:
the already-qualified original trajectory integration:
- `shadow_targets`;
- `shadow_rec_yards`;
- `shadow_room_share`.

Candidate starts from the exact original parent:
- `baseline_targets`;
- `baseline_ypt`;
- exact WR `is_wr1` anchor state.

For each player row:

`health_state_trajectory_delta = trajectory_delta`

when:
`recent2_observation_contaminated == false`.

Otherwise:

`health_state_trajectory_delta = 0.0`.

That is the only candidate change.

No other state class gets a coefficient.

`REPORTED_INJURY_PARTICIPATED` is not automatically neutralized because V1
has no source-valid "limited" threshold.

## 5. Exact conservation semantics

Reapply the already-frozen trajectory integration semantics.

### WR
For every event/team WR room:
- exact `is_wr1 == true` anchor remains byte-value unchanged;
- only non-anchor baseline targets are reweighted by
  `exp(health_state_trajectory_delta)`;
- exact non-anchor target pool preserved;
- exact total WR room target mass preserved.

### TE
For every event/team TE room:
- all baseline target rows reweighted by
  `exp(health_state_trajectory_delta)`;
- exact TE target pool preserved.

Tolerance:
`1e-12`.

## 6. Efficiency translation

No receiving efficiency change.

`candidate_rec_yards = candidate_targets * baseline_ypt`.

Rows with zero baseline target mean retain zero receiving-yard mean.

No YPT refit.
No efficiency signal.
No width change.

## 7. Primary endpoints

The **mechanism endpoint** is on directly affected 2024 rows:

`original_trajectory_abs_target_error - health_gated_abs_target_error`.

The health-state candidate must improve mean target AE on directly affected rows.

Named-player protections on directly affected rows:
- equal-player mean target-AE improvement > 0;
- median affected-player improvement >= 0;
- affected-player improved share > worsened share;
- player-cluster bootstrap
  `P(mean target-AE improvement > 0) >= 0.80`.

Bootstrap:
- cluster = `position_group|player_clean_key`;
- 5,000 reps;
- seed `20261018`;
- preserve all directly affected rows within sampled player clusters.

No alternate bootstrap after scoring.

## 8. System guardrails

The candidate also must preserve or improve the exact full allocation system.

2024 primary all-row gates:

1. pooled WR+TE target MAE <= original trajectory target MAE;
2. WR target MAE <= original trajectory target MAE;
3. TE target MAE <= original trajectory target MAE;
4. pooled receiving-yard MAE <= original trajectory receiving-yard MAE;
5. WR receiving-yard MAE <= original trajectory receiving-yard MAE;
6. TE receiving-yard MAE <= original trajectory receiving-yard MAE;
7. pooled within-room target-share MAE <= original trajectory room-share MAE;
8. rows whose recent2 state is participation-certified have target MAE
   non-worse as a collateral-harm guardrail;
9. all WR/TE conservation gaps <=1e-12.

The no-state-change rows may still move if a contaminated teammate changes the
room renormalization. That is why the clean-row collateral gate is explicit.

## 9. 2025 TE independent guardrail

The same frozen action is applied without modification to 2025 TE.

Require:
1. directly affected TE target MAE improves;
2. all-row TE target MAE non-worse;
3. all-row TE receiving-yard MAE non-worse;
4. all-row TE room-share MAE non-worse;
5. conservation <=1e-12.

If 2024 otherwise passes but 2025 TE fails any of these gates, disposition is
closed.

2025 TE cannot rescue failed 2024 gates.

## 10. Disposition

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_QUALIFIED`

requires every frozen mechanism, player-protection, system and 2025 guardrail
gate to pass.

Otherwise:

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_CLOSED`.

If an exact-source/identity/conservation failure prevents scoring:

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_INTEGRITY_FAILURE`.

## 11. Binding anti-rescue

After scoring, do not:
- change RECENT2 to another window;
- change the existing four-prior-team-game rule;
- neutralize only injury-reported no-snap but not unresolved rows;
- neutralize `REPORTED_INJURY_PARTICIPATED`;
- add a snap-percentage threshold;
- fit a partial delta multiplier;
- use return-state subsets;
- use WR-only or TE-only rescue;
- carve WR1/WR2+/TE1 subsets;
- change the bootstrap probability gate;
- use sportsbook or 2026 outcomes.

A failure closes this exact observation-state gating action.

## 12. Integrity at freeze

- candidate outcomes inspected before freeze: **0**;
- fitted coefficients: **0**;
- thresholds searched: **0**;
- new provider rebuilds: **0**;
- 2026 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_FROZEN`.
