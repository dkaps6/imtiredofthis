# RB Rush + Receiving Conservation V2 — Mean Result

Date: 2026-09-24

Status: `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`

## Canonical execution

- frozen plan commit: `5bc4be462ec3da8398a0cf186fb52e1dffb8c0d1`
- scientific head: `16949affde76131d7ff7231d923c80aac7990e36`
- run: `36005675177`
- job: `107653075505`
- artifact: `10809812602`
- digest: `sha256:b24030e52632a3c7315d16301735ccb4c26b9709920428d047b8dd6d042322fe`

Integrity:
- exact PR #549 production-order trace;
- 2,787 complete RB player-game triples;
- actual rush+receiving identity max gap = **0.0**;
- fitted parameters = 0;
- sportsbook inputs = 0;
- 2026 outcomes = 0;
- production changed = false.

## Candidate

Baseline:
`ensemble_proj(rush_rec_yards)`

Candidate:
`ensemble_proj(rush_yards) + ensemble_proj(rec_yards)`

No coefficient, blend, clip, threshold or router.

## Pooled 2024-2025

n = **2,787**

- MAE: **27.6853 -> 25.5718** (**+2.1135 yd improvement**)
- RMSE: **39.8437 -> 36.4030**
- signed bias: **+14.3388 -> +9.3891**
- p90 absolute error: **64.7742 -> 57.2604**
- 30+ yard misses: **841 -> 768**
- candidate closer rate: **59.13%**
- existing combo-minus-component projection gap:
  - mean: **-4.9497 yd**
  - mean absolute: **6.1408 yd**
  - max absolute: **38.2056 yd**

## 2024

n = 1,394

- MAE: **28.2918 -> 25.9442** (+2.3476)
- RMSE: 40.1083 -> 36.3497
- bias: +14.9639 -> +9.7249
- p90: 66.0704 -> 57.6875
- 30+ misses: 440 -> 408
- candidate closer: 59.83%

## 2025

n = 1,393

- MAE: **27.0783 -> 25.1991** (+1.8792)
- RMSE: 39.5771 -> 36.4562
- bias: +13.7134 -> +9.0532
- p90: 63.5755 -> 56.3850
- 30+ misses: 401 -> 360
- candidate closer: 58.44%

## Frozen gates

All PASS:
- actual identity exact
- pooled MAE improves
- 2024 MAE improves
- 2025 MAE improves
- pooled RMSE non-worse
- pooled absolute bias non-worse
- pooled p90 non-worse
- pooled 30+ misses non-increase
- candidate closer >50%
- no season degradation >0.50 yd

## Interpretation

The independently calibrated `rush_rec_yards` mean is losing useful information and violating an exact outcome identity. The algebraically coherent component sum is materially better out of sample with no fitted parameter.

This is not an M96 router and does not use Week-2 outcomes to define the candidate.

A separate draw-level integration test is now authorized. It must conserve:
- final standalone rush mean;
- final standalone receiving mean;
- pathwise rush+receiving = rush + receiving;
- all unrelated markets;
- sportsbook separation.

No production change is authorized by the mean result alone.
