# NFL HANDOFF — 2026-09-15 — WR PHASE 4C STAGE-1 RESULT CURRENT

**GitHub is canonical. Chat memory is secondary. Do not restart research.**

Repo: `dkaps6/imtiredofthis`

Current user priority: **WR receiving yards**. RB is parked. Do not reopen paid Full Slate work.

## Read order

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. **this file**
4. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_RESULT.md`
5. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_PLAN.md`
6. `docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_SOURCE_AUDIT.md`
7. `docs/research/WR_PHASE4C_STAGE1_LAYER2_DOMAIN_AMENDMENT_V1.md`
8. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535, especially GPT-5.6 result-review request comment `5688386255` and any Claude response after it.
9. `NFL_MASTER_CONTINUITY_RECORD.md` only if older lineage is needed.

---

# IMMEDIATE CHECKPOINT / WHAT HAPPENED

Active branch:

`research-wr-phase4c-stage1-pass-volume-predictor-v1`

Stage-1 blind outcomes have now been exposed **exactly once** under the frozen reviewed design. Do **not** rerun `--run-outcomes`, do not modify the one-shot workflow to retrigger it, and do not rescue/tune the result.

Claude cleared implementation in Issue #535 comment `5688276336` with:

`IMPLEMENTATION_REVIEW_PASS`

GPT-5.6 then executed the single frozen outcome exposure.

Canonical one-shot outcome lineage:

- orchestration commit `8b49e35f194f8dfb6538be7148590370987f8534`
- run `35025966255` — SUCCESS
- job `104572956822` — SUCCESS
- artifact `10419147232`
- artifact name `wr-phase4c-stage1-pass-volume-outcomes-v1`
- digest `sha256:09722be047456271ab2d73132cdd9c7a46dd0e771558314aba49992e096e407d`
- exact upstream Phase4B artifact reverified: `10404525877`
- upstream digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- focused Stage-1 synthetics re-run immediately before exposure: **5/5 PASS**
- full 2024 Layer2 coverage: **515/515**
- amended primary-tail coverage: **208/208**

Canonical result doc:

`docs/research/WR_PHASE4C_STAGE1_PASS_VOLUME_PREDICTOR_V1_RESULT.md`

Frozen disposition:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

No production change. No M38/R15 change. No Stage-2 integration authorization.

GPT-5.6 posted the exact result lineage/metrics to Claude in Issue #535 comment `5688386255` and requested `STAGE1_RESULT_REVIEW_PASS` if the result implementation and gate application are correct. At this handoff commit, that independent result-audit response may still be pending. The frozen observed disposition does not depend on a second opinion; Claude review is an audit, not permission for a rescue.

---

# STAGE-1 RESULT — LOAD-BEARING FACTS

## Gate table

- `coverage_integrity` = **PASS**
- `script_blind_skill` = **PASS**
- `C_gt_B_material` = **FAIL**
- `C_gt_A_material` = **FAIL**
- `no_rmse_bias_tradeoff` = **PASS**
- `wr_room_translation_positive` = **FAIL**
- `tail_mechanism_alignment` = **FAIL**

All seven gates were prospectively required, so any failure forced the negative disposition.

## The script predictor itself worked

Blind 2024 realized-pass-volume MAE:

- naive prior8 plays x prior8 dropback product = `6.8094587704`
- frozen six-feature Ridge = `6.5037568726`
- improvement = `+0.3057018978`
- paired game-cluster 95% CI = `[0.1407001638, 0.4784092432]`

Thus Stage-1 did **not** fail because the football-only script model had zero skill.

## Decisive downstream B -> C increment was real but too small

Team-target MAE:

- B = `6.1220690966`
- C = `6.0874347241`
- B-minus-C = `+0.0346343725` targets/team-game
- paired 95% CI = `[0.0025006154, 0.0664271401]`

This is a statistically positive paired difference, but the frozen practical-materiality floor was `0.10`. Therefore `C_gt_B_material = FAIL`.

Do not reinterpret the positive CI as a Stage-1 qualification. Materiality and CI were both required prospectively.

## C vs A also failed

- A MAE = `6.1379956469`
- C MAE = `6.0874347241`
- A-minus-C = `+0.0505609228`
- paired 95% CI = `[-0.0067095171, 0.1078708382]`

Below the 0.10 floor and CI crosses zero.

## RMSE/bias guardrail passed

- B RMSE `7.7763650026` -> C `7.7186318220`
- B bias `+0.2192072163` -> C `+0.0973726466`

No hidden RMSE/bias tradeoff.

## WR-room translation failed

- B WR-room MAE = `4.6988714187`
- C WR-room MAE = `4.6916908893`
- B-minus-C = `+0.0071805294`
- paired 95% CI = `[-0.0102977010, 0.0245112189]`

Tiny positive mean, not robust.

## Tail mechanism alignment failed

Primary `UNDERPROJECT_30_PLUS_OPP_DOM`:

- 208/208 addressable rows
- B-minus-C = `-0.0319909391`
- CI `[-0.0784707864, 0.0153926705]`
- C worse direction

Secondary `ACTUAL_100_PLUS_OPP_DOM`:

- 85/85 addressable rows
- B-minus-C = `-0.0849257174`
- CI `[-0.1635847717, -0.0064260775]`
- C worse, CI entirely below zero

Secondary full `UNDERPROJECT_30_PLUS`:

- 294/294 addressable rows
- B-minus-C = `+0.0094126584`
- CI `[-0.0324308503, 0.0515509403]`
- tiny unstable positive

Primary tail failure means the mechanism-alignment gate is closed regardless of the tiny positive full-tail slice.

---

# SCIENTIFIC INTERPRETATION / STOPPING RULE

The Stage-1 experiment answered a narrower question than “is game script relevant?”

It found:

1. A football-only pass-volume model can forecast realized pass volume better than a naive rolling-history product.
2. Adding that predicted scalar to B yields a small, statistically positive full-cohort target-pool improvement.
3. The magnitude is only `0.0346` targets/team-game, far below the prospectively frozen `0.10` materiality floor.
4. That small signal does not robustly improve WR-room targets.
5. It moves the primary opportunity-dominant WR underprojection tail in the wrong direction and materially worsens the actual-100+ opportunity-dominant tail.

Therefore the prospectively frozen contract closes this Stage-1 lane:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

Do **not** try:

- alternate Ridge alpha
- alternate learner/model family
- PROE
- spread or implied-team-total additions
- margin/score trajectory targets
- alternate total thresholds
- nonlinear interactions/splines
- subgroup/tail refits
- reverse rotation
- post-hoc blending or coefficient clipping
- different materiality floor
- a second blind outcome exposure

Those are all post-outcome rescues and are forbidden by the frozen plan.

The negative Stage-1 disposition does **not** reopen M38 or R15. Phase4B remains authoritative that R15 is healthy/improved and M38 did not emerge as the isolated failure.

---

# SEALED PREFLIGHT LINEAGE — PRESERVE

Canonical sealed preflight:

- run `35024523296` — SUCCESS
- job `104568212642` — SUCCESS
- artifact `10418149149`
- digest `sha256:1752b8b434424f79091d9db9955c9ac91cb4216d314ef3e90af7b0b9dcb9cdea`
- exact upstream Phase4B artifact `10404525877`
- upstream digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- focused synthetics 5/5 PASS
- blind-output guard PASS
- preflight output 515/515 Layer2 rows and 208/208 amended primary-tail rows

Blind Layer2-domain amendment remains valid:

- primary raw 222 -> 208 addressable
- actual-100+ raw 87 -> 85 addressable
- full underproject raw 312 -> 294 addressable

The rows outside Layer2 were `OUTSIDE_STAGE1_LAYER2_DOMAIN`, not silently dropped.

---

# PHASE 4B / GATE 0 — DO NOT REOPEN

Phase4B canonical:

- run `34987211287`
- artifact `10404525877`
- digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- R15 WR2+ target MAE improved from about `1.9748` to `1.8100`
- disposition `R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED`

Phase4C Gate 0 canonical:

- run `34993703697`
- artifact `10407226343`
- digest `sha256:fb4ff45d66dac4d68d727ca3786c77f35b213a68b5658d9d721ac650ddd80cc6`
- disposition `ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN`

Gate 0's descriptive bridge remains valid; Stage-1 only says this frozen predicted-pass-volume increment is not actionable enough downstream.

---

# CLOSED LANES — KEEP CLOSED

- R17 target-depth distribution
- R18 receiver target CPOE
- R19 receiver catchability
- R20 early/no-extended read priority
- M72 explosive weapon x defense
- M75 separation/cushion/aDOT/YACOE/secondary quality
- M84 player-level WR-CB assignment without valid historical responsibility data
- R3 residual persistence/calibration
- R7 explosive/YAC/air-yard persistent-player traits
- R9-R11 NGS lane
- C1 shared QB/receiver target-mass adjustment
- C3 broad joint QB/receiver combination
- ND3 vacancy/dynamic entitlement
- shared-tail/C2->WR1 variants
- percentile/skew/threshold-tuning rescue
- blindly retuning M38/R15
- PR #561 Vegas-confirmation classifier
- Stage-1 pass-volume post-output rescues listed above

Sportsbook remains downstream diagnostic/context only; do not select upstream football features from player-prop lines.

---

# PRODUCTION / OTHER LANES

Do not trigger paid Full Slate. Production authority is unchanged.

RB remains parked. Do not resume it unless the user changes priority.

---

# NEXT ACTION

1. Re-check Issue #535 for Claude's response to GPT-5.6 result-review request comment `5688386255`.
2. If Claude returns `STAGE1_RESULT_REVIEW_PASS`, record the audit in the paper trail. Do not reopen Stage-1.
3. If Claude alleges a mechanics/gate-application blocker, independently verify it. Only a genuine implementation/gate-application defect can justify correcting the canonical result. A scientific rescue/tuning request is not a blocker and must be rejected.
4. With Stage-1 closed, the next WR receiving-yards research direction must be chosen from the surviving evidence and closed-lane constraints. Do not automatically jump to RB or revive a failed WR lane.
5. If selecting a new WR direction requires a genuinely consequential scientific choice among multiple untested hypotheses, stop for user input; otherwise continue autonomously from the documented evidence.

Continue GPT-5.6 + Claude adversarial collaboration through Issue #535 with exact commit/run/artifact/digest lineage.
