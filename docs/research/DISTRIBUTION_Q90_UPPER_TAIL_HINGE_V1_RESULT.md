# DISTRIBUTION Q90 UPPER-TAIL HINGE V1 — DEVELOPMENT RESULT

Date: 2026-10-13

## Decision

**Q90_UPPER_TAIL_HINGE_DEVELOPMENT_NULL — CLOSED**

The exact frozen candidate did not qualify in any market. No prospective
shadow and no production promotion. Do not rescue with another hinge, multiplier,
market subgroup, or mean shift.

## Canonical execution

- GitHub Actions run: `38414015553` — SUCCESS
- Exact head: `d11688326e4e9ec125120f0dff14b715ec60709d`
- Artifact: `11825047006`
- Digest: `sha256:81c977681c7bfcc7daced9b0f6b91256406d9dc8533511480c034dbb0097f5f4`
- Frozen parent exact SHA: `80bc5a2768c9be031757e830e4d8ae8d958d5a93`
- Parent parity: 39,690 rows; maximum numerical differences <1.5e-14.
- Football means unchanged, maximum absolute shift 2.85e-14.
- 0 sportsbook inputs, 0 OddsAPI calls, 0 fitted parameters, 0 threshold searches, 0 2026 outcomes, 0 production changes.

2024/2025 are **development**, not independent confirmation, because the
right-tail signal had already been identified in these seasons. The experiment
was pre-frozen and scored exactly once.

## Full results

CRPS gain is baseline minus candidate, so positive means improvement.

| Market | Year | Rows | Baseline CRPS | Candidate CRPS | Gain | 95% game-cluster CI | Upper q90 baseline -> candidate | Decision |
|---|---|---:|---:|---:|---:|---|---|---|
| rush_yards | 2024 | 5270 | 6.054225 | 6.034359 | +0.019866 | [+0.011719,+0.027869] | 12.8843% -> 13.4725% | FAIL |
| rush_yards | 2025 | 5322 | 5.679679 | 5.661717 | +0.017962 | [+0.010784,+0.025261] | 11.6498% -> 12.5893% | FAIL |
| rec_yards | 2024 | 4606 | 12.700545 | 12.711762 | -0.011217 | [-0.027031,+0.005812] | 18.4976% -> 20.2779% | FAIL |
| rec_yards | 2025 | 4647 | 12.230245 | 12.216577 | +0.013669 | [-0.002656,+0.030461] | 17.1078% -> 18.6787% | FAIL |
| receptions | 2024 | 4606 | 0.997154 | 1.001398 | -0.004245 | [-0.006008,-0.002504] | 14.2206% -> 15.7186% | FAIL |
| receptions | 2025 | 4647 | 0.937276 | 0.940167 | -0.002891 | [-0.004483,-0.001211] | 13.1267% -> 14.9128% | FAIL |
| rush_rec_yards | 2024 | 5270 | 16.907214 | 16.902498 | +0.004717 | [-0.009475,+0.018131] | 30.0190% -> 31.3093% | FAIL |
| rush_rec_yards | 2025 | 5322 | 15.931820 | 15.921119 | +0.010701 | [-0.003538,+0.024751] | 27.0763% -> 28.5983% | FAIL |

## Why this matters

Rush yards improved proper CRPS significantly in both seasons, but its
upper-q90 break rate worsened, directly violating the parent calibration
question and frozen gate. Receiving yards and rush+receiving lacked stable
cluster confidence. Receptions worsened proper CRPS in both seasons.

A universal one-sided q90 excess expansion followed by exact mean realignment
is not a valid fix for the observed upper-tail asymmetry. Mean-neutral
rescaling can lower the effective central/upper quantile threshold even while
increasing the far-right extreme; a tail extension is not necessarily a better
q90 calibration.

No tuning or subgroup rescue is permitted. Future work must use materially
different, pregame football information, not another universal distribution
shape adjustment.

## Boundary

- Parent descriptive right-tail asymmetry signal remains historically valid.
- Candidate development result is NULL, not a scientific rejection of the
  parent descriptive finding.
- WR/TE receiver conversion V1 and player-efficiency-tail stability V1 remain
  independently closed.
- Preserve protected QB/WR/TE/RB opportunity and efficiency science.
- Week-6 roster authority gate remains independently binding.
