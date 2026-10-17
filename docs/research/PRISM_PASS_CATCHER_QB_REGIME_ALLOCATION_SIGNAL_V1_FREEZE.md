# PRISM Pass-Catcher × QB-Regime Allocation Signal V1 — Predictive Freeze

Date: 2026-10-17

Status: **FROZEN BEFORE PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Parent source result:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_RESULT.md`

Parent source disposition:
`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_READY_ALL`

Version:
`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1`

## 1. Player-first question

For a named WR/TE whose target-week pregame QB regime can be identified
without leakage:

> Does the way this player was allocated targets in prior games with that same
> QB contain next-game target-allocation information beyond the player's
> ordinary strictly-prior current role state?

The scientific object remains the named player in his actual current offensive
situation.

Primary outcome:
**player target share in the target game**.

Receiving efficiency, receiving yards, receptions markets, TDs and sportsbook
prices are not scored in V1.

## 2. Why this is distinct

Protected work remains protected.

This V1 is not:
- raw recent target-share replacement;
- WR/TE trajectory renamed;
- WR-R15 or TE-R5P replacement;
- pair-specific YPT;
- a broad QB/receiver combination model;
- a team receiver-room target-rate model;
- a QB mean model;
- a receiving-yard model.

The new information is the identity-conditioned allocation contrast:

`same-QB strictly-prior mean target share - overall strictly-prior mean target share`.

That contrast answers whether this named player's role historically changes
under the specific QB regime expected before kickoff.

## 3. Frozen authorities

Source state:
- run `39067415356`;
- exact head `d189d2305b745d0c7f104e6f4fd8d0d305d1da16`;
- artifact `12113054267`;
- digest `sha256:61b701091bdfcdad32e86d0c6ba2e6a471b773656c95b4fa6179adbbb54214a2`.

Files:
- `passcatcher_qb_regime_target_state.csv`;
- `active_passcatcher_completed_game_history.csv`.

The target-game completed-game file may be joined **only after** the pregame
state row is fixed. Its target share is an evaluation outcome, never a predictor.

## 4. Frozen cohort

Families are adjudicated separately:
- WR;
- TE.

Seasons:
- **2024 = development / first directional screen**;
- **2025 = untouched confirmation**.

A target player-game is signal-eligible only when ALL hold:
1. target pregame stable identity = true;
2. QB-regime source valid = true;
3. chronology violation = false;
4. same-QB prior games >=2;
5. overall prior target-share mean finite;
6. overall prior latest-2 target-share mean finite;
7. same-QB prior target-share mean finite;
8. target-game player targets and team targets resolve after state freeze;
9. target-game team targets >0.

Do not lower these gates after scoring.

Rows not signal-eligible remain part of source coverage accounting but are not
used to rescue the signal.

## 5. Frozen ordinary-role baseline

For every active player in the WR or TE room, define the ordinary current-role
baseline share before looking at the target outcome:

1. use `overall_prior_target_share_latest2` when finite;
2. otherwise use `overall_prior_target_share_latest` when finite;
3. otherwise use `overall_prior_target_share_mean` when finite;
4. otherwise use 0.

This is a diagnostic current-role baseline, not a claim that it is the
production entitlement.

It deliberately gives recent player role a strong baseline so the QB-regime
signal cannot win merely by restating generic recent usage.

## 6. Frozen QB-regime signal

For a signal-eligible player:

`qb_regime_delta_i = same_qb_prior_target_share_mean_i - overall_prior_target_share_mean_i`.

For an ineligible active player:

`qb_regime_delta_i = 0`.

No alternate history window.
No >=3-only rescue.
No QB-change-only subgroup.
No player-position rank threshold.
No fitted shrink coefficient.

## 7. Frozen candidate construction

For each team-week-family room:

`raw_i = clip(baseline_i + qb_regime_delta_i, 0, 1)`.

Let:

`baseline_room_mass = sum(baseline_i)`.

Then preserve that exact finite family-room mass:

- if `sum(raw_i) > 0`,
  `candidate_i = raw_i * baseline_room_mass / sum(raw_i)`;
- otherwise candidate equals baseline.

This V1 therefore changes **allocation within the WR or TE room** while holding
the baseline room's total target-share mass fixed.

It cannot create extra WR/TE room opportunity.

This is intentionally a no-fit candidate:
- fitted coefficients = 0;
- candidate variants = 1;
- searched thresholds = 0.

## 8. Evaluation

Primary paired error per signal-eligible player-game:

`abs(predicted target share - actual target share)`.

Report for baseline and candidate:
- MAE;
- median absolute error;
- RMSE;
- p90 absolute error;
- closer rate;
- signed bias.

Also report conditional target-count error using the **realized target-game team
target total only as a postgame evaluation denominator**:

`predicted player targets = predicted share * actual team targets`.

This target-count view is diagnostic only. It does not permit target-game team
targets into a predictor.

## 9. Frozen gates

Adjudicate WR and TE separately.

To qualify, ALL must pass.

### Development — 2024

1. signal-eligible rows >=1,000 WR / >=600 TE;
2. candidate target-share MAE < baseline target-share MAE;
3. candidate p90 absolute share error <= baseline;
4. candidate closer rate >50%;
5. candidate target-count MAE <= baseline;
6. family-room mass conservation max absolute gap <=1e-12;
7. chronology violations = 0.

### Confirmation — 2025

8. signal-eligible rows >=1,000 WR / >=600 TE;
9. candidate target-share MAE < baseline target-share MAE;
10. candidate p90 absolute share error <= baseline;
11. candidate closer rate >50%;
12. candidate target-count MAE <= baseline;
13. 5,000-replicate paired **team-week cluster bootstrap** 95% CI for
    `candidate_abs_error - baseline_abs_error` has upper bound <0;
14. 5,000-replicate paired **player cluster bootstrap** 95% CI has upper bound <0;
15. median equal-player mean error delta <0;
16. family-room mass conservation max absolute gap <=1e-12;
17. chronology violations = 0.

### Integrity

18. target-game outcomes used in predictors = 0;
19. sportsbook inputs = 0;
20. OddsAPI calls = 0;
21. 2026 outcomes = 0;
22. candidate variants = 1;
23. fitted coefficients = 0;
24. threshold searches = 0;
25. production changes = false.

All gates are mandatory.

## 10. Interpretation if PASS

A PASS means only:

> same-QB-regime history contains incremental predictive information about the
> named pass catcher's target allocation beyond a strong ordinary recent-role
> baseline, while preserving the room's finite target-share mass.

It would authorize a **separately frozen authority-integration experiment**.

That later experiment must test the signal against the actual canonical
entitlement state:
- TE may use exact OOS TE-R5P / qualified Stage-2 authority where legal;
- WR must respect the known 2025 WR-R15 OOS-authority block and may not relabel
  final-fit training overlap as OOS.

No receiving-yard or market translation follows automatically from a PASS.

## 11. Interpretation if FAIL

Fail closed.

Do not rescue by:
- using >=3 same-QB games only;
- changing latest-2 to latest-3/4;
- fitting a shrink weight;
- splitting QB changes vs stable QB;
- selecting only WR1/TE1;
- using target count instead of share as the primary endpoint;
- pooling WR and TE;
- moving to receiving yards;
- using 2026 Week-4/5 outcomes;
- adding sportsbook context.

## 12. Integrity at freeze

At this freeze:
- predictive candidate scored: **0**
- 2026 outcomes used: **0**
- Week-6 outcomes used: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- coefficients fit: **0**
- thresholds searched: **0**
- production changes: **0**

Disposition:
`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_FROZEN`.
