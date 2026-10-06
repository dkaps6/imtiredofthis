# RB Opponent-Defender Injury Candidate V1 — Frozen Contract

**STATUS: FROZEN BEFORE ANY CANDIDATE SCORE. RESEARCH ONLY. NO PRODUCTION CHANGE.**

Readiness authority:
- branch: `research-rb-opponent-defender-injury-readiness-v1`
- result: `docs/research/RB_OPPONENT_DEFENDER_INJURY_READINESS_V1_RESULT.md`
- run: `37534490584` — SUCCESS
- artifact: `11446405827`
- digest: `sha256:01b4124be06a435da3048e931517cff81dbc036d3a23ad00c4cf9d49ed58e97b`
- disposition: `RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY`

Baseline authority:
- frozen right-tail run: `37493425352`
- artifact: `11427113291`
- digest: `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`
- baseline column: `distribution_right_tail_asymmetry_detail.csv::final_mean`

Position identity authority:
- preserved historical player logs from run `37527077327`
- artifact: `11444106835`
- digest: `sha256:6e2efdd9649bfa8c3043b74b07866cb08364409d3a7b2e72340d6fcca2957ec8`

## Purpose

Test one narrow football hypothesis:

> A target-week opponent defense missing more strictly-prior FRONT7 defensive snap mass via OUT/DOUBTFUL injury designations should, all else equal, be associated with higher RB rushing-yard production than the current football baseline predicts.

This is materially new pregame information. It is not another transform of aggregate rush EPA, box rate, pass rate, PROE, coverage, or historical defensive tendency.

## Candidate

Candidate ID:

`RBDI-F7-1`

Eligible player rows:
- season 2024 or 2025;
- Week 2-18;
- market `rush_yards`;
- position RB/HB/FB from preserved player-history identity;
- exact frozen right-tail baseline identity.

No QB rushing rows.

## Frozen injury feature

For each target defensive team-week, define:

`front7_out_doubtful_snap_mass`

as the sum of the most recent strictly-prior `defense_pct` for every target-week injury row satisfying:
- defensive group = DL or LB;
- target-week game status contains OUT or DOUBTFUL;
- strictly-prior snap join exists;
- chronology is valid.

Important fail-close rule:

If a target defensive team-week has one or more OUT/DOUBTFUL FRONT7 injury identities but any such identity lacks a strictly-prior snap join, that team-week is **not candidate-scoreable**. Do not silently treat the missing player as zero burden.

If the team-week has zero OUT/DOUBTFUL FRONT7 injury identities, burden is exactly `0.0`.

No threshold, cap, nonlinear transform, status weight, player-quality weight, position-specific coefficient, or interaction is allowed.

## Train / confirmation split

Coefficient training:
- 2024 Weeks 2-18 only.

Independent confirmation:
- 2025 Weeks 2-18 only.

2026 is forbidden from candidate fitting or scoring.

## Frozen functional form

Let:

`residual = actual_rush_yards - baseline_final_mean`

Fit 2024 zero-intercept OLS:

`residual = beta_train * front7_out_doubtful_snap_mass`

Then freeze `beta_train`.

2025 candidate:

`candidate_mean = baseline_final_mean + beta_train * front7_out_doubtful_snap_mass`

No intercept.

No refit in 2025.

No sign flip.

If `beta_train <= 0`, the directional hypothesis is contradicted and the candidate closes immediately.

A zero-burden row must remain bitwise/effectively unchanged:
`candidate_mean == baseline_final_mean`.

## Primary 2025 confirmation metrics

Compare the frozen candidate with the frozen baseline on the exact same scoreable 2025 RB rows.

Required support:
- at least 500 scoreable player-games;
- at least 100 unique NFL games.

Primary error metrics:
- MAE;
- RMSE;
- signed bias;
- 75+ yard absolute-error count;
- 100+ yard absolute-error count.

Dependence-aware inference:
- game-cluster bootstrap;
- 10,000 replicates;
- deterministic seed `20261006`;
- estimand = mean(`baseline_AE - candidate_AE`);
- positive favors candidate.

## PASS gate

`RBDI_F7_1_CONFIRMED` requires ALL:

1. `beta_train > 0`;
2. support floor met;
3. 2025 candidate MAE < baseline MAE;
4. game-cluster bootstrap 95% CI lower bound for paired MAE improvement > 0;
5. candidate RMSE <= baseline RMSE;
6. absolute signed bias <= baseline absolute signed bias;
7. 75+ yard AE count <= baseline;
8. 100+ yard AE count <= baseline;
9. zero-burden rows unchanged within `1e-10`;
10. all chronology / identity / baseline-authority gates pass;
11. sportsbook inputs used = 0;
12. 2026 target outcomes read = 0;
13. production changed = false.

With adequate support, failure of any scientific gate yields:

`RBDI_F7_1_CLOSED`

Do not rescue a failed result with:
- OUT-only;
- DOUBTFUL-only;
- DL-only;
- LB-only;
- top-N players;
- minimum snap thresholds;
- alternate caps;
- interactions with box/rush EPA/pass tendency;
- high-burden subgroups;
- post-hoc status weighting;
- sportsbook lines;
- 2026 outcome inspection.

## Interpretation boundary

A PASS would establish only that this exact opponent-personnel burden contains stable incremental RB rushing-yard mean information across 2024 -> 2025.

It would **not** authorize direct production integration. A separate integration/mechanism plan would still be required to determine whether the effect should enter:
- team rushing volume;
- RB allocation;
- rushing efficiency;
- or another explicit production seam.

A FAIL closes this exact one-feature additive mean mechanism. It does not invalidate the source itself.

Candidate models fit before this contract: **0**  
Sportsbook inputs authorized: **0**  
2026 outcomes authorized: **0**  
Production mutations authorized: **0**
