# WR-R18 Receiver-Attributed Target CPOE V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY WR-R18 FOOTBALL RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE. NO PAID FULL SLATE. CLAUDE REVIEW REQUIRED BEFORE REAL-DATA STAGE A.**

## 1. Decision and lineage

This is the single next WR receiving-yard hypothesis after the valid WR-R17 target-depth-distribution null.

Canonical base at branch creation:
- `main`: `de6aed84d474867d81427f4e8277219868ac9d50`
- branch: `research-wr-r18-receiver-target-cpoe-v1`

WR-R17 authority/result:
- frozen plan commit: `7142c52fdee49e345ace9226df4dfeb40cc41a96`
- valid implementation head: `838e15389da8dd834745f35e799d3ce25a64e8f2`
- workflow run: `34898469962`
- artifact: `10369531993`
- digest: `sha256:d0c50a17b274677e6780207c77150d5c64e0c688536a6b8ce7f5ea7dacc0fe4b`
- result doc commit: `dd9d9cd39c67bc524ec5fd91771082a055a85db8`
- disposition: `NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`
- 2024 WR-R17 holdout was not scored.

Post-R17 anti-retest audit:
- branch: `research-wr-yard-efficiency-feature-audit-v1`
- commit: `862086baf2eeb946d78101331b45f70f166ecff9`
- `wr_completed_air_yards_per_target8` rejected as too close to R7 / completed-air persistence.
- old R16 `team_*` delivery features rejected as overlapping M70/M71.
- recent-minus-long delivery deltas rejected as a primary candidate because a different history window does not itself create a new information source and thin recent target counts raise noise risk.
- receiver-attributed target CPOE remained the only old-R16 variable with a plausible novelty case.

Claude independently audited the closed lanes and WR-R17 in GitHub Issue #535. Claude reproduced the WR-R17 raw CI metrics exactly and separately inspected C3. Claude's conclusion was that C3 is a joint target-mass/conservation study, R7 is raw realized-stat persistence, and M70/M71 are QB-attributed CPOE/efficiency studies; none directly tests receiver-attributed CPOE predicting that receiver's next-game receiving-yard residual. Claude therefore judged `wr_target_cpoe_mean8` genuinely open, while agreeing that completed-air should be dropped and recent-vs-long CPOE should not be a co-equal candidate.

No other signal is authorized in WR-R18 V1.

## 2. Football hypothesis

M38 + `WR_R15_PRODUCTION_MODEL_V1` remain the frozen WR opportunity authority. The unresolved weakness is receiving-yard translation conditional on opportunity.

WR-R18 asks one narrow question:

> Does the strictly-prior **completion-over-expectation state of passes targeted to a specific WR** contain reproducible next-game information about that WR's receiving-yard residual after the frozen M38/R15 opportunity projection?

The proposed mechanism is receiver-specific delivery/connection quality rather than generic raw YPT, YAC, explosiveness, mean target depth, target-depth distribution shape, or team/QB efficiency. A WR can receive similar projected target opportunity and similar raw target depth while the historical throws to that WR have converted above or below model-implied completion expectation.

The predeclared mechanism direction is **positive**:
- higher prior receiver-attributed target CPOE should associate with a more positive next-game receiving-yard residual;
- lower prior receiver-attributed target CPOE should associate with a more negative next-game receiving-yard residual.

A strong opposite-sign result is scientifically interesting but does **not** pass this preregistered mechanism.

## 3. Exact authority cohort

Use only the exact WR-R15 OOS authority artifact:
- run: `34238301577`
- artifact: `10061328722` (`wr-r15-wr1-anchor-participation-v1`)
- digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- variant exactly: `WR_R15_WR1_ANCHORED_PARTICIPATION`
- expected rows exactly: `4,193`
- 2023: `2,076` development rows
- 2024: `2,117` untouched holdout rows

Authority projection fields are immutable. In particular, do not alter `mc_rec_yards`, `mc_receptions`, `pred_targets`, `entitlement_tgt_share`, or `wr_rank`.

Primary outcome:
`receiving_yard_residual = actual_rec_yards - mc_rec_yards`.

No sportsbook line, price, odds, target-game result feature, or target-game PBP predictor may enter WR-R18.

## 4. Source, identity, and temporal contract

Historical target events come from nflverse/nflreadpy regular-season PBP.

For each authority row `(season S, week W, team, player)`:
- only events strictly before `(S,W)` may feed features;
- target-game PBP is forbidden as a predictor source;
- receiver history lookup must be GSIS/player-ID first;
- reuse the proven WR-R17 v1c identity pattern: exact authority full-name key -> strictly-prior nflverse weekly-roster stable GSIS ID -> prior PBP by GSIS ID;
- target-week roster rows are forbidden for identity construction;
- authority team may disambiguate multiple exact-name roster IDs only if exactly one stable ID remains;
- otherwise fail closed as unmatched/ambiguous;
- exact PBP-name fallback is permitted only when no stable GSIS anchor exists, must be reported separately, and may not use fuzzy matching;
- repeated target events must remain repeated events; never deduplicate targets merely because air yards/CPOE values repeat.

Every run must report identity-source counts, unmatched/ambiguous counts, target-game leakage count, source seasons, and sportsbook-input count.

## 5. Exact target-event definition

A receiver target event is an nflverse PBP row satisfying all of the following:
- regular season;
- official pass attempt (`pass_attempt == 1`);
- not a sack;
- not a two-point attempt;
- receiver identity resolves to the authority receiver under the ID-first contract;
- `cpoe` is non-null.

Use nflverse `cpoe` in its native source units. Do not rescale, winsorize, clip, transform, or recalculate expected completion probability in V1.

Incomplete targets remain eligible if nflverse provides a valid `cpoe`; the feature is mean completion over expectation on **all valid targeted official pass attempts**, not completed passes only.

## 6. History and support

For each target authority row, use the receiver's last **8 eligible prior target-bearing games** strictly before `(S,W)`.

Primary signal support requires:
- at least `4` prior target-bearing games; and
- at least `16` valid receiver target events with non-null `cpoe` across the last-8 history; and
- unambiguous receiver identity.

The `16`-event floor is frozen before football-result exposure to reduce thin-sample rate noise relative to the earlier 12-event depth-distribution floor. If this causes insufficient coverage, the correct disposition is data-blocked; do not lower the floor after seeing coverage or outcomes.

## 7. Single frozen candidate signal

Exactly one advancing signal is permitted:

### `WR_TARGET_CPOE_MEAN8`
Arithmetic mean of nflverse `cpoe` across all valid target events in the receiver's last-8 eligible prior target-bearing games.

There is no feature zoo and no priority search because there is only one candidate.

Explicitly excluded as advancing signals:
- completed air yards / target;
- recent-3 minus prior-8 CPOE;
- team/QB CPOE as a candidate;
- target-depth SD/IQR/deep-target share;
- mean aDOT as a candidate;
- YPT/YPR/YAC;
- EXP20/EXP40 outcomes;
- opponent explosive/YAC/air allowances;
- WR-CB assignment proxies;
- NGS separation/cushion/YACOE;
- sportsbook inputs.

## 8. Development / holdout separation

- 2023 = development / signal-existence season.
- 2024 = untouched confirmation holdout.
- Do not pool seasons to choose sign, thresholds, support, coefficient, transformation, or subgroup.
- 2024 outcomes remain sealed unless the 2023 primary signal clears every Stage-A raw gate **and** the receiver-specific novelty robustness in Section 10.

## 9. Stage A frozen gates — 2023 only

Report on valid 2023 rows:
- valid N and coverage over all 2,076 authority rows;
- Spearman correlation of `WR_TARGET_CPOE_MEAN8` with receiving-yard residual;
- 2023-only Q4 minus Q1 mean receiving-yard residual gap;
- Q4/Q1 actual-100+ receiving-yard game rate ratio;
- Q4/Q1 rate ratio for directional underprojection `receiving_yard_residual >= +30`;
- WR1 and WR2+ residual gaps when each slice has at least 150 valid rows;
- descriptive MAE and signed bias by quartile, non-advancing.

`DEVELOPMENT_SUPPORTED_RAW` requires **all**:
1. coverage >= `60%`;
2. Spearman >= `+0.08`;
3. Q4-Q1 receiving-yard residual gap >= `+5.0` yards;
4. either Q4/Q1 actual-100 rate ratio >= `1.20` **or** Q4/Q1 `residual >= +30` rate ratio >= `1.20`;
5. WR1 and WR2+ residual-gap directions are both positive when each has >=150 valid rows;
6. all identity/source/leakage audits pass.

Opposite-sign metrics may be reported as `DIRECTIONAL_CONTRADICTION` but do not satisfy the hypothesis.

If the raw gates fail, disposition is `NO_ACTIONABLE_WR_RECEIVER_TARGET_CPOE_SIGNAL`; 2024 remains sealed.

## 10. Receiver-specific novelty / mediation robustness

Because receiver-attributed CPOE can still proxy general team/QB accuracy or receiver role, a raw Stage-A pass does not unlock 2024 by itself.

For a raw-passing 2023 signal only, construct strictly-prior controls from the same history cutoff:
- `team_cpoe_mean8`: mean CPOE across all official team pass attempts in the team's last 8 prior games; **control only, never a candidate**;
- `mean_air_yards_per_target8`: receiver mean air yards per valid prior target; **control only**;
- frozen WR-R15 `entitlement_tgt_share`;
- `wr_rank_bucket = WR1` when `wr_rank == 1`, else `WR2PLUS`.

Fit on 2023 only:
`WR_TARGET_CPOE_MEAN8 ~ team_cpoe_mean8 + mean_air_yards_per_target8 + entitlement_tgt_share + C(wr_rank_bucket)`

Define `receiver_specific_cpoe` as the OLS residual.

Report:
- Spearman(`receiver_specific_cpoe`, receiving-yard residual);
- Q4-Q1 residual gap using 2023-only quartiles of `receiver_specific_cpoe`.

To unlock Stage B, both receiver-specific metrics must remain positive and retain at least half of the raw Stage-A gate magnitude:
- Spearman >= `+0.04`;
- Q4-Q1 residual gap >= `+2.5` yards.

If raw Stage A passes but this robustness fails, disposition is `WR_TARGET_CPOE_TEAM_ROLE_MEDIATED`; preserve the clue as partial evidence but do **not** inspect 2024 under WR-R18 V1.

Freeze the 2023 control-model coefficients and all 2023 quartile thresholds before Stage B.

## 11. Stage B holdout contract — 2024 untouched

Only if Sections 9 and 10 both pass, apply the 2023-frozen rules to 2024 with no refit/reselection.

Holdout confirmation requires all:
1. coverage >= `60%` of all 2,117 authority rows;
2. raw Spearman >= `+0.06`;
3. frozen-2023-threshold high-vs-low receiving-yard residual gap >= `+4.0` yards;
4. frozen high-vs-low actual-100 rate ratio >= `1.15` **or** `residual >= +30` rate ratio >= `1.15`;
5. WR1 and WR2+ residual-gap directions both positive when each has >=150 valid rows;
6. receiver-specific/orthogonalized CPOE remains positive on the frozen 2023 control model, with Spearman >= `+0.03` and high-vs-low residual gap >= `+2.0` yards;
7. zero target-game feature leakage, zero sportsbook inputs, and all identity/source audits pass.

PASS disposition:
`WR_RECEIVER_TARGET_CPOE_REPLICATED_SIGNAL`.

Holdout fail disposition:
`WR_RECEIVER_TARGET_CPOE_DEVELOPMENT_ONLY_SIGNAL`.

Data disposition:
`WR_RECEIVER_TARGET_CPOE_DATA_BLOCKED`.

A PASS is not a production model. It authorizes only a separately preregistered integration experiment that must protect M38/R15 opportunity/reception accuracy and improve receiving-yard football metrics.

## 12. Evidence preservation / season-consistency reporting

WR-R18 must report two separate concepts rather than collapsing everything into one PASS/FAIL label:

### A. Experiment disposition
Whether the exact preregistered gates passed.

### B. Scientific evidence disposition
What useful directional/seasonal evidence exists even when promotion gates fail.

Allowed WR-R18 evidence labels:
- `NO_DIRECTIONAL_EVIDENCE`: no coherent positive signal in the scored development evidence.
- `PARTIAL_DIRECTIONAL_EVIDENCE`: one or more meaningful descriptive components move in the hypothesized direction but the frozen experiment gate does not pass.
- `TEAM_ROLE_MEDIATED_EVIDENCE`: raw signal passes but receiver-specific orthogonalization does not.
- `DEVELOPMENT_ONLY_SIGNAL`: 2023 passes/unlocks holdout but 2024 does not replicate.
- `REPLICATED_2_OF_2_SIGNAL`: both 2023 development and untouched 2024 holdout confirm.
- `DIRECTIONAL_CONTRADICTION`: material opposite-sign evidence.
- `DATA_BLOCKED`: source/identity/support prevents an honest test.

These labels do not override the frozen promotion gates. They exist so a failed experiment does not erase informative sub-findings or season-specific behavior.

Do not call a failed season an `ANOMALY` merely because another season passes. A season-anomaly claim requires a separately identified, pregame-observable football regime or a later independently preregistered replication design.

## 13. Stop rules

After any WR-R18 football result is visible, do **not**:
- change the last-8 window;
- lower the 16-target support floor;
- add recent-3 or alternate recency windows;
- add completed-air features;
- transform, clip, winsorize, or reweight CPOE;
- change quartiles to tertiles/deciles;
- choose a negative sign after preregistering a positive mechanism;
- change Stage-A or Stage-B gates;
- add interactions or QB/team candidate features;
- inspect 2024 if Stage A + novelty robustness do not unlock it;
- rescue a Stage-A failure with WR1-only, WR2+-only, tail-only, matchup, or opponent slices;
- reopen R3, R7, R9-R11, M72/M75, ND3, C1/C2/C3, WR-R17 distribution shape, or the full old WR-R16 package under a new label;
- use sportsbook lines/odds upstream;
- alter M38/R15 opportunity science.

A descriptive positive sub-finding may be preserved under Section 12, but any follow-up hypothesis created from it must be separately frozen **before** new-season/holdout exposure.

## 14. Collaboration gate before Stage A

Claude must independently review this exact plan before any real-data Stage-A football result is run. The review should challenge:
- whether receiver-attributed CPOE is truly novel vs R7/M70/M71/C3;
- whether nflverse `cpoe` semantics and target-event construction match the claimed mechanism;
- whether 16 valid CPOE targets + 4 prior target-bearing games is adequate and not outcome-tuned;
- whether the R17 GSIS-first prior-roster bridge is appropriate here;
- whether team-CPOE/role/aDOT controls adequately test receiver-specific novelty without over-controlling the mechanism;
- whether the Stage-A/Stage-B gates and expected positive sign are defensible;
- whether the evidence-preservation labels create any hidden post-result degrees of freedom.

Any accepted amendment must be committed before any WR-R18 football result is exposed.
