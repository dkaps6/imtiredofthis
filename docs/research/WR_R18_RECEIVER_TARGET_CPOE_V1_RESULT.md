# WR-R18 Receiver-Attributed Target CPOE V1 — Stage A Result

**STATUS: 2023 DEVELOPMENT COMPLETE. FORMAL EXPERIMENT FAIL. SCIENTIFIC EVIDENCE = PARTIAL_DIRECTIONAL_EVIDENCE. 2024 HOLDOUT SEALED. NO PRODUCTION CHANGE.**

## Canonical execution

- branch: `research-wr-r18-receiver-target-cpoe-v1`
- workflow head: `fa7ffbb819b5ffd3fa9999758db0b5b554c3025a`
- workflow run: `34905163992`
- workflow conclusion: `success`
- artifact: `10371569919`
- artifact name: `wr-r18-receiver-target-cpoe-stage-a-v1`
- artifact digest: `sha256:0f46d544091949be62797f22009e456d01d4169e27e326c42c7c1a90c7354bf0`

Frozen WR-R15 authority re-verified in-run:
- run: `34238301577`
- artifact: `10061328722`
- digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- 2023 authority rows: `2,076`
- 2024 authority rows: `2,117`

The workflow passed the final corrected synthetic suite before real data, verified the exact authority artifact lineage, scored only 2023, asserted `holdout_2024_scored == false`, and uploaded the result artifact.

## Formal experiment disposition

`NO_ACTIONABLE_WR_RECEIVER_TARGET_CPOE_SIGNAL`

Raw 2023 Stage A did not clear every preregistered gate. Therefore:
- receiver-specific mediation/orthogonalization was **not evaluated**;
- 2024 remains sealed;
- WR-R18 does not advance to Stage B;
- no production change is authorized.

## Exact frozen-gate results

| Gate | Frozen requirement | 2023 result | Disposition |
| --- | ---: | ---: | --- |
| Coverage | >= 60% | `1667 / 2076 = 80.30%` | PASS |
| Spearman | >= +0.08 | `+0.02245` | **FAIL** |
| Q4-Q1 yard residual gap | >= +5.0 yd | `+5.2211 yd` | PASS |
| Tail evidence | >= 1.20 on either frozen tail ratio | `2.8889x` actual-100; `1.3824x` residual>=+30 | PASS |
| WR1 direction | positive if n>=150 | n=`495`, gap=`+14.2239 yd` | PASS |
| WR2+ direction | positive if n>=150 | n=`1172`, gap=`+0.8399 yd` | PASS |
| Identity/source/leakage | all clean | clean | PASS |

The only frozen gate failure is the monotonic rank-correlation gate. It is not a marginal miss: `+0.02245` is far below the required `+0.08`.

Frozen CPOE quartiles:
- Q1 boundary: `-5.77344`
- Q4 boundary: `+5.29947`

Tail rates:
- Q4 actual 100+ yards: `12.4700%`
- Q1 actual 100+ yards: `4.3165%`
- ratio: `2.88889x`
- Q4 residual >= +30 yards: `22.5420%`
- Q1 residual >= +30 yards: `16.3070%`
- ratio: `1.38235x`

## Descriptive quartile evidence

These were frozen as descriptive-only outputs and do not alter the experiment disposition.

| Quartile | n | Signed residual bias | MAE |
| --- | ---: | ---: | ---: |
| Q1 | 417 | `+4.66 yd` | `20.71 yd` |
| Q2 | 417 | `+8.95 yd` | `25.47 yd` |
| Q3 | 416 | `+8.95 yd` | `26.45 yd` |
| Q4 | 417 | `+9.88 yd` | `25.94 yd` |

Interpretation: there is coarse high-vs-low / tail separation, especially among WR1 rows, but not enough rank-monotonic relationship across the full cohort to satisfy the preregistered mechanism.

## Identity / temporal / source audit

- stable prior-roster GSIS resolution: `2055 / 2076`
- unmatched: `21`
- team-disambiguated rows: `16`
- rows with >=4 prior target-bearing games: `1872`
- rows with >=16 valid CPOE targets: `1679`
- rows with valid CPOE signal: `1667`
- authority display-key mismatches: `0`
- receiver target-game leakage rows: `0`
- team-control target-game leakage rows: `0`
- sportsbook inputs: `0`
- PBP history seasons: `2022, 2023`
- identity roster seasons: `2022, 2023`
- 2024 holdout scored: `false`

## CPOE missingness audit

Among resolved receiver target events used for the selection audit:
- resolved WR target events: `18,976`
- non-null CPOE events: `18,976`
- null CPOE events: `0`
- null CPOE rate: `0.0%`

Thus the pre-result concern that missing nflverse CPOE might selectively remove difficult receiver targets does not manifest in this resolved receiver-target cohort.

## Scientific evidence disposition

`PARTIAL_DIRECTIONAL_EVIDENCE`

Why this is not `NO_DIRECTIONAL_EVIDENCE`:
- the frozen Q4-Q1 residual-gap gate passes;
- both frozen tail diagnostics pass;
- WR1 and WR2+ directions are both positive;
- the source/identity/leakage contract passes.

Why this is not a replicated/actionable signal:
- full-cohort rank monotonicity is near zero and fails badly;
- the strongest subgroup magnitude is concentrated in WR1s, while WR2+ magnitude is very small;
- Stage A therefore does not unlock mediation or 2024.

This evidence label does **not** override the failed experiment gate and carries no production eligibility.

## Holdout protection / stop rule

Do not inspect 2024 under WR-R18 V1.

Do not rescue the Stage-A failure by:
- switching to WR1-only;
- inventing a CPOE threshold after seeing these quartiles;
- changing the CPOE window/support floor;
- replacing monotonicity with only the tail result;
- adding interactions or matchup slices;
- changing the positive direction or frozen gates.

A separately frozen follow-up may be considered only if the 2023 descriptive pattern maps to a genuinely distinct football mechanism and receives adversarial pre-result review before any new-season exposure.

## Collaboration status

Claude independently passed the WR-R18 plan and implementation before real-data exposure, including an independent parity calculation and fresh last-8-game mechanical test. After this result, GPT-5.6 requested an independent Claude audit of run `34905163992` / artifact `10371569919`, including whether `PARTIAL_DIRECTIONAL_EVIDENCE` is the correct scientific-evidence label and whether the high-vs-low / WR1 / tail pattern merits any genuinely new preregistered hypothesis.
