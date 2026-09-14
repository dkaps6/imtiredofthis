# WR Post-R17 Delivery-Quality Anti-Retest V1

**STATUS: AUDIT ONLY. NO NEW CANDIDATE AUTHORIZED. NO RESULT RUN.**

## Purpose

Record GPT-5.6's independent post-R17 anti-retest view before Claude's requested challenge in Issue #535. This document is deliberately not a frozen experiment plan.

Canonical production authority remains M38 + `WR_R15_PRODUCTION_MODEL_V1`. The active question remains WR receiving-yard translation after opportunity, not target entitlement.

## New closure from WR-R17

WR-R17 target-depth distribution failed valid 2023 Stage A and therefore never opened the 2024 holdout.

Canonical evidence:
- branch `research-wr-r17-target-depth-distribution-v1`
- frozen plan `7142c52fdee49e345ace9226df4dfeb40cc41a96`
- valid run `34898469962`
- head `838e15389da8dd834745f35e799d3ce25a64e8f2`
- artifact `10369531993`
- digest `sha256:d0c50a17b274677e6780207c77150d5c64e0c688536a6b8ce7f5ea7dacc0fe4b`
- disposition `NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`
- valid feature coverage 1,775 / 2,076 = 85.50%
- 2024 holdout not scored.

R17 closes, without rescue:
- prior-8 target-depth SD;
- prior-8 target-depth IQR;
- prior-8 deep15 target-share;
- alternate threshold/percentile/skew/deep-mass rescue from the same target-depth-distribution family;
- reopening R16's `wr_target_depth_sd8` separately.

## Remaining R16 feature-family audit

The old R16 proposal is not authorized as a package. It already had a name-first receiver-history defect and mixed multiple overlapping families. Its `team_*` delivery variables overlap M70/M71 and its target-depth SD is now independently closed by R17.

### 1. `wr_completed_air_yards_per_target8`

**Disposition: DO NOT ADVANCE as a new primary mechanism.**

Why:
- It is a realized efficiency component: completed air yards divided by targets.
- WR-R7 already tested strict-prior `PLAYER_AIR_PER_TARGET_PRIOR8` inside the YPR-dominant receiving-yard error subgroup. It failed, with Spearman `-0.091237`, YPR-component Q4-Q1 `-6.7190 yd`, and total receiving-yard residual gap `-4.9179 yd`.
- M70 explicitly decomposed QB efficiency into completed-air and YAC components and built `completed_air_per_att` / air-per-completion baselines.
- Using only completed air rather than all intended air changes the arithmetic but not enough of the football information family to justify another standalone persistent-efficiency test after R7/M70-M71.

This may remain a descriptive decomposition variable, but it should not receive its own predictive lane.

### 2. `wr_completed_air_recent3_minus8`

**Disposition: DO NOT ADVANCE absent materially new mechanism.**

A recent-minus-long window does not automatically create new football information. It is still the same realized completed-air family and risks becoming a volatility/state repackaging of information already challenged by R7 plus the M70-M71 efficiency/volatility program.

No window tuning (3v8, 2v6, 4v8, etc.) should be used to rescue it.

### 3. `team_cpoe_mean8`, `team_air_per_attempt_mean8`, `team_deep15_completion_rate8`, `team_completed_air_per_attempt8` and deltas

**Disposition: CLOSED / DROP FROM ANY NEXT WR CANDIDATE.**

Claude previously confirmed the direct construction overlap with M70's `mean_cpoe`, completed-air, and deep-attempt families. M82's integration ledger classifies M70-M71 `qb_efficiency_volatility_uncertainty` as `SIGNAL_SCREEN_FAILED`, reopenable only with a new decision/process observable rather than the same history at a different aggregation level.

Changing QB -> team aggregation or QB yards -> WR yards target variable is not sufficient novelty by itself.

### 4. `wr_target_cpoe_mean8`

**Disposition: POSSIBLY DISTINCT, BUT NOT YET AUTHORIZED.**

This is the only old-R16 variable that retains a credible novelty argument after R17:
- it is receiver-attributed rather than QB/team aggregate;
- CPOE conditions the observed completion result on expected completion difficulty, so it is not identical to raw intended air depth, target-depth distribution, YAC, explosive rate, or historical YPT;
- WR-R7 did not test receiver-attributed CPOE;
- R17 did not test completion quality at all.

But material anti-retest risk remains:
- M70/M71 already tested strict-prior CPOE/efficiency/volatility information at QB level and failed the broader residual-prediction frontier;
- receiver-target CPOE is still a realized prior outcome, not a new pregame sensor;
- receiver-specific aggregation may merely re-label the same QB/receiver efficiency history rather than add independent information;
- C3's broad joint QB/receiver family is closed, so a new CPOE formulation must not become another generic QB/receiver combination under a new name.

Therefore `wr_target_cpoe_mean8` cannot advance merely because its aggregation level is different.

### 5. `wr_cpoe_recent3_minus8`

**Disposition: POSSIBLY DISTINCT STATE VARIABLE, BUT EVEN WEAKER NOVELTY CLAIM THAN LEVEL CPOE.**

A change-state question is conceptually different from a static player level, but M70-M71 already challenged efficiency volatility/uncertainty and the project has repeatedly rejected window/recency rescue after failed static signals. A recent-minus-long receiver CPOE delta should advance only if the mechanism is prospectively framed as a real receiver-target delivery-state transition and Claude independently agrees it is materially different from M70-M71/R3/C3—not because an alternate window might correlate better.

## Preliminary frontier after GPT audit

There is **no authorized R18 yet**.

If Claude independently agrees a genuinely distinct lane remains, the narrowest plausible question is:

> Does strictly-prior **receiver-attributed completion-over-expectation state**, evaluated only on already-projected WR opportunity from M38/R15, contain next-game receiving-yard residual information not captured by raw air-depth/YAC/YPT history or QB/team efficiency history?

If that question survives collaboration, the candidate should be narrower than old R16:
- receiver-attributed CPOE only;
- perhaps one prospectively justified change-state term, not a model-zoo of windows;
- no `team_*` delivery features;
- no completed-air feature as a primary candidate;
- no target-depth distribution feature;
- GSIS-first identity using the proven R17 prior-roster bridge;
- exact WR-R15 authority cohort;
- 2023 development / untouched 2024 holdout;
- no sportsbook data;
- no generic YPT/YAC/explosive-history rescue;
- no post-result interaction/window search.

An even stricter interpretation is also acceptable: if Claude concludes receiver-attributed CPOE is only a repackaging of M70-M71/C3, then this lane should be closed before execution and the project should seek materially new information rather than run it.

## Collaboration gate

Issue #535 currently contains GPT's request for Claude to:
1. independently verify R17's exact run/artifact failure;
2. challenge receiver-attributed CPOE novelty versus R7/M70/M71/C3;
3. challenge completed-air-per-target as too close to realized efficiency persistence;
4. assess whether recent-minus-long CPOE is a real state mechanism or only window noise;
5. propose a better genuinely-open mechanism if one exists.

Until that response is reconciled, status remains:

`POST_R17_DELIVERY_QUALITY_AUDIT_COMPLETE_CANDIDATE_NOT_AUTHORIZED`

No production change, no RB work, no paid Full Slate run.
