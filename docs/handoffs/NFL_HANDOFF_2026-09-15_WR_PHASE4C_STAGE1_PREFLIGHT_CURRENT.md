# NFL HANDOFF — 2026-09-15 — WR PHASE 4C STAGE-1 SEALED PREFLIGHT CURRENT

**GitHub is canonical. Chat memory is secondary. Do not restart research.**

Repo: `dkaps6/imtiredofthis`

Current user priority: **WR receiving yards**. RB is parked. Do not reopen paid Full Slate work.

## Read order

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. **this file**
4. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PLAN.md`
5. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_SOURCE_AUDIT.md`
6. `docs/research/WR_PHASE4C_STAGE1_LAYER2_DOMAIN_AMENDMENT_V1.md`
7. latest GPT-5.6 / Claude comments in GitHub Issue #535, especially GPT-5.6 comment `5688220080` and any Claude response after it.
8. `NFL_MASTER_CONTINUITY_RECORD.md` only if older lineage is needed.

---

# IMMEDIATE CHECKPOINT / WHAT TO DO NEXT

Active branch:

`research-wr-phase4c-stage1-pass-volume-predictor-v1`

Current branch head at handoff creation:

`19187f967835827988fca2f2d4e77c46d65bf58d`

Evaluator domain-amendment commit:

`5b728ddfab6cfb167481b78b0f9ed849838f0326`

**Stage-1 blind 2024 outcomes are still SEALED. `--run-outcomes` has NOT been executed.**

Canonical sealed preflight:

- run `35024523296` — SUCCESS
- job `104568212642` — SUCCESS
- artifact `10418149149`
- artifact name `wr-phase4c-stage1-pass-volume-preflight-v1`
- digest `sha256:1752b8b434424f79091d9db9955c9ac91cb4216d314ef3e90af7b0b9dcb9cdea`
- exact upstream Phase4B artifact reverified before execution: `10404525877`
- upstream digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- focused Stage-1 synthetics: **5/5 PASS**
- blind-output contract guard: **PASS**

GPT-5.6 posted the exact implementation-review request to Claude in Issue #535 comment:

`5688220080`

At handoff creation, GitHub API showed **no later comment after `2026-09-15T21:17:35Z`**. First action in the next chat: re-check Issue #535 for Claude’s implementation verdict.

### If Claude returns `IMPLEMENTATION_REVIEW_PASS`

Run the **single frozen Stage-1 `--run-outcomes` exposure** against the exact same authority artifact and preserve the result regardless disposition. Do not redesign, retune, add features, alter thresholds, or run a rescue.

### If Claude raises a blocker

Independently verify it. Fix only the blocker/mechanics if valid. Re-run sealed preflight before any outcome exposure. Do not expose blind outcomes until Claude’s implementation review is cleared.

---

# STAGE-1 SCIENCE — FROZEN, DO NOT CHANGE

Plan freeze commit:

`3c96d1d1c7aa3d541646e2396d0d73cb636e594a`

Plan file:

`docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PLAN.md`

The user and Claude jointly cleared this design. Claude’s Stage-1 plan review was `REVIEW_PASS`, conditional only on proving zero Vegas lineage in the historical `plays_est` / `dropback_rate` source. GPT-5.6 traced and documented that lineage in Issue #535 comment `5687809210`; the condition is satisfied.

## Question

Can a strictly pregame **football-only forecast of realized team pass volume** add blind-2024 information beyond both the Phase-4B football baseline and the posted game total?

Decisive comparison: **C > B out of sample**.

Predicting script well by itself is insufficient.

## Realized script target

`realized_pass_volume = plays_est * dropback_rate`

This is deliberately narrower than a broad score/margin trajectory model because Gate 0 implicated scoring/pass-volume realization, while spread/blowout views did not advance.

Do NOT reopen PR #561’s failed “will Vegas be confirmed?” classifier.

## Script model

Football-only features, exactly six:

1. own prior-8 plays
2. own prior-8 dropback rate
3. opponent-defense prior-8 plays allowed
4. opponent-defense prior-8 dropback rate allowed
5. home flag
6. rest differential

Rolling window = 8, minimum prior games = 3, strictly prior, forward cross-season only.

Model:

`StandardScaler -> Ridge(alpha=20.0)`

No tuning.

Cross-fit:

- 2023 script prediction: fit on 2022 only
- 2024 script prediction: fit on 2022+2023 only
- 2021 is history backstop only

## Historical-source lineage — verified clean

`plays_est` and `dropback_rate` come directly from completed nflverse PBP through:

`build_historical_inputs.py -> build_all_historical_inputs() -> build_team_weekly_from_pbp() -> scripts/utils/pbp.py::get_pbp()`

Exact source semantics:

- `plays_est = len(g)` where `g` is realized offensive plays defined by `qb_dropback == 1 OR rush_attempt == 1`
- `dropback_rate = mean(qb_dropback)` over the same offensive-play set

No spread, total, moneyline, implied total, market calibration, player prop, or Vegas-derived field enters either quantity. Market columns are joined only later in PR #558’s diagnostic. Thus Stage-1 C’s predicted pass volume has **zero market/Vegas lineage**.

## Downstream A/B/C

Canonical Phase4B Layer2 population:

- 2023 = 510 team-games — fit season
- 2024 = 515 team-games — blind test season

Response:

`r = actual_team_targets - implied_team_target_pool`

A = intercept-only residual calibration.

B = A + `market_total` + frozen `<38` low-total indicator.

C = B + OOS `pred_realized_pass_volume`.

A0 raw authority is reported separately.

No spread, moneyline, `market_team_implied`, splines, interactions, alternate total thresholds, or rescue features in the decisive B/C test.

## Frozen Stage-1 PASS gate

All conditions must pass:

1. >=95% coverage of full 515-row 2024 Layer2 domain and >=95% of addressable primary-tail domain; zero temporal leakage.
2. Script predictor beats naive prior8-play × prior8-dropback on blind 2024 with paired NFL-game-cluster CI lower bound >0.
3. `MAE_B - MAE_C >= 0.10` targets/team-game and paired CI lower >0.
4. `MAE_A - MAE_C >= 0.10` and paired CI lower >0.
5. `RMSE_C <= RMSE_B`; absolute bias not worse by >0.25 targets.
6. Frozen WR-room translation improves B→C with paired CI lower >0.
7. C reduces team-target absolute error vs B on the primary tail and same direction on at least one secondary tail.

Bootstrap:

- 10,000 reps
- seed `20260915`
- cluster by actual NFL game so opposing team rows stay together.

Frozen 0.10 team-target materiality bar is ~2.212% of the Phase4B Layer2 ~4.520381 MAE reference. Do not move it.

Any failure =>

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

No alternate alpha/model/PROE/market threshold/spread/margin target/interactions/subgroups/reverse rotation rescue.

PASS => only authorizes a separately frozen Stage-2 integration/replay design. It does NOT authorize production change.

---

# BLIND LAYER2-DOMAIN AMENDMENT — IMPORTANT

First sealed preflight attempt:

- run `35023900963`
- job `104566140347`
- evaluator `--preflight-only` PASS
- 5/5 synthetics PASS
- exact Phase4B artifact/digest PASS
- final contract guard failed before upload/outcomes because the original tail coverage denominator was mathematically impossible.

No blind Stage-1 outcome metric was scored.

Problem: original frozen tail counts came from all Layer1 rows, but B/C team-pool predictions exist only on the anchor-observable Layer2 response domain.

Structural intersection audit, membership only:

- primary `UNDERPROJECT_30_PLUS_OPP_DOM`: raw Layer1 222 unique team-games -> **208 Layer2-addressable**, 14 outside
- `ACTUAL_100_PLUS_OPP_DOM`: raw 87 -> **85 addressable**, 2 outside
- full `UNDERPROJECT_30_PLUS`: raw 312 -> **294 addressable**, 18 outside

Amendment file:

`docs/research/WR_PHASE4C_STAGE1_LAYER2_DOMAIN_AMENDMENT_V1.md`

Frozen Stage-1 tail denominator is now:

`Layer1 frozen tail membership ∩ canonical Phase4B Layer2 2024 domain`

Raw parent counts remain disclosed and the 14/2/18 outside rows are classified `OUTSIDE_STAGE1_LAYER2_DOMAIN`; they were not silently dropped.

This changed no target, feature, model, materiality threshold, bootstrap rule, or tail definition. It only repaired the response-domain denominator before any blind outcome scoring.

Canonical amended preflight is the successful run `35024523296` listed above.

Preflight proves:

- `blind_2024_outcomes_scored=false`
- no `stage1_result.json`
- sealed CSV = **515/515** canonical 2024 Layer2 rows
- sealed CSV contains zero `actual_team_targets`, `actual_wr_room_targets`, or `realized_pass_volume` columns
- target/future rolling rows used = 0
- script model market inputs = `[]`
- market lineage in script target/priors = 0
- 2023 script fit: 2022 only, 542 train / 544 emitted historical predictions
- 2024 script fit: 2022+2023 only, 1086 train / 544 emitted historical predictions
- downstream A/B/C fit cohort = exact 510 2023 Layer2 team-games, same cohort all arms
- 2024 Layer2 coverage = 515/515 = 100%
- addressable primary-tail coverage = 208/208 = 100%

Two orchestration-only patch workflow failures also exist in Actions (YAML parse; runner push rejected for workflow permission). Neither scored Stage-1 outcomes or changed science. Do not confuse them with scientific failures.

---

# PHASE 4B AUTHORITY-EXACT OPPORTUNITY ATTRIBUTION — CLOSED / PRESERVE

Canonical Phase4B one-shot:

- run `34987211287` SUCCESS
- job `104442281504`
- artifact `10404525877`
- digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- result doc commit `e9d80025f6cc203503fc7c7b169218f4b54cece8`

Key findings:

- R15 is healthy/improved, not the problem.
- WR2+ target MAE baseline `1.9748` -> R15 `1.8100`; delta about `-0.1648` targets/game, replicated in 2023/2024 with CI below zero.
- frozen disposition `R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED`.
- M38/WR1 did not emerge as isolated structured failure.
- overall receiving-yard error roughly 50/50 opportunity vs efficiency: mean abs opportunity ~16.26 yd, efficiency ~16.43 yd.
- opportunity dominates only ~44.96% overall but becomes more important in tails: ~60.26% of actual 100+ games and ~55.92% of >=30-yard misses.
- team target-pool vs WR-room-share components are mixed; no single upstream smoking gun.

Preserve M38. Preserve R15. Do not build an R15 challenger.

---

# PHASE 4C GATE 0 — PASSED / CLOSED AS DESIGN GATE

Branch:

`research-wr-phase4c-tail-gamescript-gate-v1`

Canonical Gate-0:

- run `34993703697` SUCCESS
- job `104464523520`
- artifact `10407226343`
- digest `sha256:fb4ff45d66dac4d68d727ca3786c77f35b213a68b5658d9d721ac650ddd80cc6`
- result commit `ecc120fbd8dc08f4596d1e67d90d775d869c2ea0`
- disposition `ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN`

Load-bearing findings:

- low-total `<38` games depleted primary opportunity-tail failures
- continuous `market_total` positively associated with primary tail and Layer2 signed team-pool/direct target residual
- raw market total / `<38` are load-bearing
- `market_team_implied` is supporting only, not load-bearing
- spread/blowout magnitude did not advance

Claude independently audited Gate 0 and confirmed the disposition while noting the implied-team-total leg is weaker.

---

# IMPORTANT CLOSED LANES — DO NOT RETEST

- R17 target-depth distribution
- R18 receiver target CPOE
- R19 receiver catchability
- R20 early/no-extended read priority
- M72 explosive weapon × defense
- M75 separation/cushion/aDOT/YACOE/secondary quality
- M84 player-level WR-CB assignment without valid historical responsibility data
- R3 residual persistence/calibration (explicitly closed by user)
- R7 explosive/YAC/air-yard persistent-player traits
- R9-R11 NGS lane
- C1 shared QB/receiver target-mass adjustment
- C3 broad joint QB/receiver combination
- ND3 vacancy/dynamic entitlement
- shared-tail/C2→WR1 variants
- percentile/skew/threshold-tuning rescue
- blindly retuning M38/R15
- PR #561 pregame Vegas-confirmation classifier
- PR #558 broad market injection result remains a valid narrow null; Stage1 is distinct because it asks whether a football-only predicted pass-volume state adds beyond market specifically on the Phase4B team-pool residual.

Sportsbook is downstream diagnostic/context only; never select upstream football features from player-prop lines.

---

# PRODUCTION / OTHER LANES

Do not trigger paid Full Slate. Week-1 production repair is already closed/green and not the current task.

Production authorities remain unchanged:

- QB mean = M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution = `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR = M38 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE = `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing = `RB_P3_SYNTHESIS_V1` qualified W1 route
- RB receptions/opportunity = R26
- RB receiving-yard tail/distribution = R22 using frozen R19 assets

RB remains parked. Do not continue it unless user changes priority.

---

# COLLABORATION RULE

Continue GPT-5.6 + Claude collaboration through Issue #535. Do not merely agree with Claude: independently verify objections/results. Likewise send exact lineage, commit/run/artifact/digest when asking Claude to review.

The next chat should move autonomously from the exact checkpoint above without asking the user to re-explain anything.
