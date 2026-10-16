# DISTRIBUTION Q90 UPPER-TAIL HINGE V1 — FROZEN DEVELOPMENT CONTRACT

Date: 2026-10-13

Status: **FROZEN BEFORE CANDIDATE SCORING / DEVELOPMENT QUALIFICATION ONLY / NO PRODUCTION CHANGE**

## Parent authority

Historical parent:
`DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_SIGNAL_CONFIRMED`

Canonical parent:
- branch `research-distribution-right-tail-asymmetry-v1`
- exact source head `80bc5a2768c9be031757e830e4d8ae8d958d5a93`
- run `37493425352` — SUCCESS
- artifact `11427113291`
- digest `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`

Replicated in BOTH 2024 and 2025:
- rush_yards
- rec_yards
- receptions
- rush_rec_yards

Pass yards did not replicate and is excluded.

The parent diagnostic used no sportsbook inputs, fit no parameters and changed no
football mean.

## Why 2023 is NOT a confirmation holdout

The current receiving ensemble weights for rec_yards/receptions were fit on 2023
and frozen for 2024-2025. Therefore scoring this candidate on 2023 with the
current final-mean stack would leak target-season outcomes through the mean
authority.

2023 MUST NOT be called an independent holdout for this candidate.

2024/2025 are also not independent confirmation seasons because their outcomes
already established the right-tail parent signal.

Therefore this experiment is explicitly **development qualification only**.

A positive result may authorize a prospective future-week shadow. It is not
historical confirmation and cannot promote production.

## Frozen candidate

For each eligible individual player-market row:

1. reconstruct the exact parent baseline empirical football distribution;
2. mean-align it to the exact parent final football mean;
3. compute that row's baseline empirical q90;
4. for each draw `x`:
   `candidate_pre = x + max(x - q90, 0)`
5. multiplicatively re-align `candidate_pre` to the exact same football mean.

Interpretation:
- below/equal q90: no direct tail-excess addition;
- above q90: existing excess above q90 is doubled;
- no fitted skew coefficient;
- no market-specific factor;
- no player subgroup;
- no threshold search;
- the q90 hinge is inherited directly from the confirmed parent diagnostic.

This is the smallest integer no-fit upper-tail-excess expansion:
one additional copy of the existing excess above the already-frozen q90 hinge.

## Market scope

Exactly:
- rush_yards
- rec_yards
- receptions
- rush_rec_yards

No pass_yards.
No rush_att.
No anytime TD.

## Parent-baseline reproducibility

Candidate scoring MUST reconstruct with the exact parent code head
`80bc5a2768c9be031757e830e4d8ae8d958d5a93`.

Before any candidate disposition:
- exact 2024/2025 parent season scope;
- exact 2,000 draws per row;
- exact row identity;
- raw reconstructed MC mean matches stored mc_proj <=1e-8;
- mean-aligned baseline equals frozen final football mean <=1e-8;
- parent result artifact must be downloadable;
- regenerated parent detail must match the frozen parent detail on row keys and
  final_mean/q10/q90 within 1e-8.

A parity failure is mechanical/integrity failure, not science.

## Proper scoring

Primary per-row score:
- empirical CRPS.

For each season/market:
`crps_gain = baseline_crps - candidate_crps`

Positive is better.

Game-cluster bootstrap:
- 5,000 reps
- fixed seed 20261013
- resample authoritative game_id clusters
- statistic = mean CRPS gain.

## Tail/calibration diagnostics

Report baseline and candidate:
- q10/q90 break rates;
- q05/q95 break rates;
- central 80% interval coverage;
- central 90% interval coverage;
- interval widths;
- row SD ratio;
- exact mean shift.

The primary calibration target inherited from the parent is upper q90 escape.

## Development qualification — independently per market

A market earns:
`Q90_UPPER_TAIL_HINGE_DEVELOPMENT_QUALIFIED`

only if BOTH 2024 and 2025 independently satisfy:

1. support >=200 rows and >=50 games;
2. mean CRPS gain >0;
3. game-cluster 95% CI lower bound for CRPS gain >0;
4. candidate q90 upper-break rate is lower than baseline;
5. absolute q90 upper-break calibration error vs 10% is lower than baseline;
6. central 80% coverage absolute error vs 80% is non-worse by more than 0.5pp;
7. central 90% coverage absolute error vs 90% is non-worse by more than 0.5pp;
8. absolute row mean shift <=1e-10.

No pooled-season rescue.
No majority-season rescue.
No subgroup rescue.

If a market fails:
`Q90_UPPER_TAIL_HINGE_DEVELOPMENT_NOT_QUALIFIED`.

Overall:
- >=1 market qualifies:
  `Q90_UPPER_TAIL_HINGE_DEVELOPMENT_SURVIVOR_EXISTS`
- none:
  `Q90_UPPER_TAIL_HINGE_DEVELOPMENT_NULL`
- any lineage/integrity violation:
  `Q90_UPPER_TAIL_HINGE_INTEGRITY_FAILURE`

## Prospective boundary

A development survivor is NOT validated.

It may only authorize a separately frozen future-week shadow beginning with the
first future week for which:
- authoritative current roster/player state is valid before kickoff;
- canonical football means/distributions are generated outcome-blind;
- the exact V1 transform is locked before outcomes.

Week-6 remains separately blocked if exact Week-6 roster authority is stale.
Do not substitute Week-5 identity merely to obtain a shadow.

## Anti-rescue

After candidate results are exposed, do NOT search:
- 1.25x / 1.5x / 3x tail factors;
- q80/q85/q95 hinge alternatives;
- market-specific factors;
- position/player/role subsets;
- OVER/UNDER betting rules;
- sportsbook-conditioned transforms;
- mean boosts;
- asymmetric transformations with fitted parameters;
- pass-yards inclusion.

A null closes this exact q90 hinge family.

Sportsbook inputs: **0**
OddsAPI calls: **0**
2026 outcomes: **0**
Football means changed: **false**
Production changed: **false**
