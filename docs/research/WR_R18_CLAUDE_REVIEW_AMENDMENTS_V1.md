# WR-R18 Claude Review Amendments V1

**STATUS: ACCEPTED BEFORE ANY WR-R18 FOOTBALL RESULT. REPORTING / PROCESS ONLY. NO GATE, COHORT, FEATURE, SIGN, OR STOP-RULE CHANGE.**

Claude returned `REVIEW_PASS` on the frozen WR-R18 plan and `RESEARCH_EVIDENCE_CLASSIFICATION_V1` in Issue #535 comment `5671604249`.

The frozen WR-R18 science remains unchanged. The following reporting/process additions are required for the eventual Stage-A/result record.

## 1. CPOE-null selection audit

Alongside Stage A, report for otherwise-eligible receiver-targeted official pass attempts before the target game:
- count and rate with non-null nflverse `cpoe`;
- count and rate with null `cpoe`;
- air-yard availability on null-CPOE vs non-null-CPOE target events;
- mean/median `air_yards` for null-CPOE vs non-null-CPOE target events when available;
- a simple association/contrast sufficient to disclose whether CPOE missingness appears related to target depth.

This audit is descriptive only. It may not change the frozen 16-valid-CPOE-event support floor or select a different subset after results.

## 2. Mediation ambiguity must be explicit

If raw WR target CPOE passes but the receiver-specific residualized signal fails the frozen mediation/novelty gate, the result remains mechanically `WR_TARGET_CPOE_TEAM_ROLE_MEDIATED` and 2024 stays sealed.

However, the result document must state that controlling `mean_air_yards_per_target8` can remove either:
- genuine depth/role confounding, or
- part of a real delivery-quality pathway expressed through the kinds of targets a receiver earns.

Therefore a mediation failure is evidence that the raw signal is not independently separable under the frozen controls; it is not automatically proof that receiver-attributed CPOE is causally meaningless.

Also disclose unmodeled game-script / score-differential context as a V1 limitation. It is not added as a post-hoc control in V1.

## 3. Sealed-season reuse requires adversarial review

Any future hypothesis that proposes using a season that remained sealed after a failed or mediated development stage must, before that season is exposed:
1. freeze the materially new/narrower hypothesis, cohort, features, direction, gates, support rules, and stop rules; and
2. receive an independent adversarial pre-result review through the project collaboration process.

This makes the project's existing GPT-5.6 + Claude practice an explicit written requirement.

## 4. No science changes from this review

Claude explicitly found no blocking issue with:
- receiver-attributed CPOE novelty versus R7/M70/M71/C3;
- the positive predeclared direction;
- the 4-game / 16-valid-target support floor;
- Stage-A gates `+0.08 / +5 yd / 1.20` plus role coherence;
- the receiver-specific mediation unlock thresholds;
- Stage-B holdout gates;
- the R17 prior-roster GSIS identity bridge;
- the evidence-preservation framework's protection against post-hoc rescue.

Accordingly none of those frozen elements may be changed based on this review.
