# WR Post-R19 Read-Priority Source/Redundancy Audit V1

**STATUS: SOURCE/NOVELTY AUDIT ONLY. NO WR OUTCOME SCORING. NO PRODUCTION CHANGE.**

## Trigger

WR-R19 receiver catchability closed as `NO_ACTIONABLE_WR_RECEIVER_CATCHABILITY_SIGNAL` / `NO_DIRECTIONAL_EVIDENCE`. Claude's Issue #535 anti-retest audit rejected receiver-specific `is_contested_ball` outright and left receiver-specific `read_thrown` / first-read share as not closed but not yet clean enough to freeze because of apparent 2022-vs-2023 coding instability and possible redundancy with WR-R15 opportunity/entitlement.

## Official FTN semantics now frozen for this audit

Authoritative current nflverse FTN data dictionary semantics:
- `read_thrown == "0"` = first / primary read.
- `read_thrown == "1"` = second read.
- `read_thrown == "2"` = third read or later.
- `CHK` = checkdown.
- `DES` = designed read / no progression (e.g. screens and many RPOs).
- `SD` = scramble drill.
- The `"0"` code begins in 2023; **2022 primary reads are uncoded and appear as NA**.

Therefore the only allowed cross-season normalization is:
- 2022 `NA` -> `PRIMARY_READ`
- 2023+ `"0"` -> `PRIMARY_READ`
- `"1"` -> `SECOND_READ`
- `"2"` -> `THIRD_PLUS`
- `CHK`, `DES`, `SD` remain their own categories.

Any other non-null category is an unexpected schema value and must be reported. This rule is frozen before any receiver outcome is loaded.

## Candidate source feature for redundancy audit only

`PRIMARY_READ_SHARE8_ALL_TARGETS`:

> Among all official receiver-target events in the receiver's last 8 eligible strictly-prior target-bearing games, share whose normalized `read_thrown` category is `PRIMARY_READ`.

Rules:
- all official receiver targets are in the denominator, including incompletions;
- `CHK`, `DES`, `SD`, `SECOND_READ`, and `THIRD_PLUS` count as non-primary for this feature;
- last 8 target-bearing games are selected before aggregation;
- support floor mirrors the recent WR research program: >=4 prior target-bearing games and >=16 receiver targets;
- identity is GSIS-first using strictly-prior weekly roster bridge, with exact PBP-name fallback only when stable prior roster identity is absent; ambiguous identities fail closed;
- no target-week roster row;
- no fuzzy identity;
- no sportsbook inputs.

## What this audit is allowed to inspect

Source-only:
- nflverse FTN charting 2022-2024 for `read_thrown` semantics/population;
- nflverse PBP 2022-2024 only for exact play/receiver identity and official target population;
- weekly roster identity source;
- exact WR-R15 authority artifact solely for 2023 player identity plus `entitlement_tgt_share`, `pred_targets`, and `wr_rank` redundancy checks.

The authority loader must count/key-audit the full 2023/2024 contract for artifact parity but may materialize only 2023 non-outcome fields. It must not parse/materialize `actual_rec_yards`, `mc_rec_yards`, residuals, or any 2024 projection/outcome field.

## Redundancy diagnostics

On supported 2023 authority rows, report:
1. coverage of `PRIMARY_READ_SHARE8_ALL_TARGETS`;
2. Pearson and Spearman correlation versus `entitlement_tgt_share`;
3. Pearson and Spearman correlation versus `pred_targets`;
4. simple linear R^2 of primary-read share explained by `entitlement_tgt_share + pred_targets + WR1_indicator`;
5. primary-read-share dispersion within entitlement-target-share quartiles (N, mean, SD, IQR) to show whether materially different read-priority profiles exist at similar opportunity levels;
6. exact normalized category distribution by source season and unknown-category count;
7. identity/source audits and exact FTN->PBP join rate.

## Source/novelty decision rule

This is not a football predictive test and cannot promote a model. It only determines whether an R20 plan is scientifically defensible.

Call `READ_PRIORITY_SOURCE_REDUNDANCY_BLOCKED` if any of the following is true:
- exact FTN->PBP join rate <95% in any source season;
- normalized read population is not >=95% interpretable after the frozen mapping;
- supported 2023 authority coverage <60%;
- absolute Spearman(primary-read share, entitlement target share) >=0.75; or
- linear R^2 from entitlement + pred_targets + WR1 indicator >=0.60.

Otherwise call `READ_PRIORITY_R20_PLAN_ELIGIBLE`, while explicitly preserving the R17 target-depth adjacency risk and requiring a separate frozen R20 football plan before any outcome is exposed.

These thresholds are frozen before this audit runs. They are novelty/redundancy screens, not production or predictive gates.

## Hard stop rules

- No WR receiving-yard outcome or residual may be loaded/scored.
- No 2024 WR-R15 outcome/projection field may be parsed.
- No alternate normalization after seeing audit results.
- No `DES`/`CHK`/`SD` regrouping after seeing redundancy results.
- No contested-ball rescue; that lane is closed.
- No R17/R18/R19 reopening.
- No RB work.
- No production change.
- No paid Full Slate.
