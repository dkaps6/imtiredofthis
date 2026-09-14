# WR-R17 Target-Depth Distribution V1 — Stage A Result

**STATUS: FAILED IN 2023 DEVELOPMENT. DO NOT SCORE THE 2024 HOLDOUT.**

## Authority / frozen contract

- plan: `docs/research/WR_R17_TARGET_DEPTH_DISTRIBUTION_V1_PLAN.md`
- frozen-plan commit: `7142c52fdee49e345ace9226df4dfeb40cc41a96`
- branch: `research-wr-r17-target-depth-distribution-v1`
- WR-R15 authority run: `34238301577`
- WR-R15 authority artifact: `10061328722` (`wr-r15-wr1-anchor-participation-v1`)
- authority digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- authority cohort: 4,193 rows = 2,076 (2023 development) + 2,117 (2024 sealed holdout)
- target outcome: `actual_rec_yards - mc_rec_yards`
- frozen signal priority: `DEPTH_SD8`, `DEPTH_IQR8`, `DEEP15_TARGET_SHARE8`

No sportsbook input was used. No target-game PBP feature was used. The 2024 holdout was not scored.

## Pre-result mechanical lineage

The first implementation never produced a valid football result.

1. `7c23a08ca329cfe2604366429cacd4c816ae717f` — initial evaluator.
2. `ebc89f3faf6d0e9e3cc714fa06a881fd4981935c` — synthetic test.
3. `74da5315745817f8a81303c37c36fffd425aec17` — pre-result correction preserving repeated same-depth target events.
4. run `34897896248` — failed at the synthetic import gate before any real-data Stage A execution.
5. run `34897984434` — mechanically completed but had 0/2,076 matched identities because the authority uses full canonical WR names while the PBP receiver-name field is abbreviated. This run is **DATA-BLOCKED / scientifically invalid**, notwithstanding its early JSON label. It exposed zero usable feature rows and did not score 2024.
6. `a50e9a5917e48e48d9b994a12506a0ba7d290b0c` plus subsequent wiring commits — added strictly-prior weekly-roster GSIS identity bridge, using the repo's existing identity-only roster-source pattern. No football feature/gate/cohort changed.
7. authoritative implementation head: `838e15389da8dd834745f35e799d3ce25a64e8f2`.

The identity repair is exact-name -> stable prior roster GSIS ID -> prior PBP by GSIS ID. It does not use fuzzy matching or target-week roster evidence. Synthetic tests explicitly verify that target-week roster rows cannot influence the mapping.

## Authoritative Stage A evidence

- workflow: `WR R17 Target Depth Distribution V1`
- run: `34898469962`
- head SHA: `838e15389da8dd834745f35e799d3ce25a64e8f2`
- artifact: `10369531993` (`wr-r17-target-depth-distribution-stage-a-v1`)
- artifact digest: `sha256:d0c50a17b274677e6780207c77150d5c64e0c688536a6b8ce7f5ea7dacc0fe4b`
- implementation: `v1c_prior_roster_gsis_bridge`
- mechanical workflow conclusion: `success`
- scientific disposition: `NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`

### Identity / source audit

- 2023 authority rows: 2,076 / 2,076
- stable-ID resolved rows: 2,055
- unmatched rows: 21
- rows team-disambiguated from exact-name collisions: 16
- rows with >=4 prior target-bearing games: 1,872
- rows with >=12 valid prior target events: 1,799
- valid depth-signal rows: 1,775
- signal coverage: 1,775 / 2,076 = **85.500963%**
- authority display-key mismatches: 0
- target-game leakage rows: 0
- sportsbook inputs: 0
- PBP history seasons: 2022, 2023
- identity-roster seasons: 2022, 2023
- `holdout_2024_scored`: `false`

Coverage therefore clears the frozen 60% gate. This is a valid scientific Stage A result, not a source-blocked result.

## Frozen Stage A results

| Signal | n | Coverage | Spearman | Q4-Q1 residual gap | Actual-100 directional ratio | 30+ miss directional ratio | WR1 gap | WR2+ gap | Supported |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `DEPTH_SD8` | 1,775 | 85.50% | -0.004662 | +1.562 yd | 0.9600 | 0.8870 | -3.072 yd | +2.801 yd | No |
| `DEPTH_IQR8` | 1,775 | 85.50% | -0.025313 | -0.315 yd | 0.9622 | 0.9189 | -3.522 yd | +0.726 yd | No |
| `DEEP15_TARGET_SHARE8` | 1,775 | 85.50% | -0.022587 | +0.111 yd | 0.8110 | 0.7985 | -1.164 yd | +0.101 yd | No |

Frozen development requirements were:
- coverage >=60%;
- abs(Spearman) >=0.08;
- abs(Q4-Q1 residual gap) >=5.0 yd with matching sign;
- actual-100 or 30+ miss directional rate ratio >=1.20;
- coherent WR1 / WR2+ direction where each slice has >=150 rows;
- identity/source/leakage audit pass.

All three candidates miss the core signal-strength gates by a wide margin. `DEPTH_SD8` is nearly zero correlation; `DEPTH_IQR8` and `DEEP15_TARGET_SHARE8` are likewise near-null. No candidate advances.

## Scientific interpretation

The tested mechanism is closed:

> Strictly-prior within-WR target-depth distribution shape — dispersion, IQR, or deep-target mass — does not provide actionable next-game receiving-yard residual information after the frozen M38/R15 opportunity projection on the 2023 development cohort.

This does **not** say route depth is irrelevant to football. It says this exact historical targeted-pass distribution family, under the frozen support rules and authority cohort, does not clear even the development signal-existence gates and therefore is not entitled to a 2024 confirmation test.

The result also strengthens the anti-retest boundary:
- do not retune the SD/IQR/deep15 thresholds;
- do not try alternate percentiles, skew, upper-tail mass, or deep thresholds as rescue;
- do not inspect 2024 to choose a better distribution statistic;
- do not reopen R16's `wr_target_depth_sd8` as a separate lane;
- do not combine these failed signals with team/QB delivery variables to rescue them post hoc.

## Disposition

`NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`

Per the frozen stop rule, **2024 remains sealed and Stage B is not authorized**.

M38 / WR-R15 remain unchanged production authorities. No production change, no RB work, and no paid Full Slate run is authorized by this result.
