# TE Player Allocation Joint Dirichlet V1 — Result

Date: 2026-10-14

Status: **COMPLETE / NO ACTIONABLE TE JOINT DIRICHLET TRANSMISSION / RESEARCH ONLY**

Frozen contract:
- `docs/research/TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `e34430812854588ade7d87b5fbfd5aed59c59f65`
- workflow: `Research TE Player Allocation Joint Dirichlet V1`
- run: `38656885725` — SUCCESS
- job: `116168645524`
- artifact: `11937013082`
- digest: `sha256:b6c8b885eb3185fc864bc49410dbe37c87f544ccbefc357447343f80ddfb04d4`

Parent:
- `TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`
- allocation-only run `38656442803`

## Frozen disposition

`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_JOINT_DIRICHLET`

Support and every integrity gate passed.

The candidate failed scientifically.

## Support / integrity

Parent:
- 1,476 TE allocation-only rows / 114 TEs in 2024-2025
- 1,088 player-rows / 90 TEs had mathematically valid overdispersion relative to ordinary multinomial sampling

Rooms:
- 799 team-game rooms seen
- 735 candidate-applied rooms scored
- 64 rooms had no overdispersed eligible TE
- zero rooms failed team-target-count inference
- zero probability failures
- zero count-conservation failures

2024:
- 376 rooms
- 32 teams
- 204 games
- 547 eligible TE marginals / 72 TEs

2025:
- 359 rooms
- 32 teams
- 202 games
- 541 eligible TE marginals / 79 TEs

Integrity:
- exact room target conservation: PASS
- probabilities sum to one: PASS
- candidate and baseline expected TE counts equal within max `1.78e-15`
- no sportsbook / OddsAPI / 2026 outcomes / production mutation

## Joint room score — FAILED

Positive joint gain means lower NLL for the candidate.

2024:
- mean joint gain: `-0.00200`
- game-cluster 95% CI: `[-0.13144,+0.13393]`
- team-cluster 95% CI: `[-0.13529,+0.13106]`

2025 independent confirmation:
- mean joint gain: `-0.07130`
- game-cluster 95% CI: `[-0.19259,+0.05725]`
- team-cluster 95% CI: `[-0.21767,+0.08956]`

Neither season passed the frozen room-level proper-score gate.

## Marginal TE score — FAILED

2024:
- baseline Binomial CRPS: `1.52590`
- candidate Beta-Binomial CRPS: `1.64828`
- candidate is worse by `0.12239`

2025:
- baseline Binomial CRPS: `1.42409`
- candidate Beta-Binomial CRPS: `1.54869`
- candidate is worse by `0.12460`

Thus the exact overdispersion transmission worsens individual TE target-count proper scores even though it preserves the mean exactly.

## Named-player 2025 confirmation — FAILED

Among 49 TEs with >=6 candidate-applied games:
- median per-player marginal CRPS gain: `-0.12159`
- fraction positive: `14.29%`

Player-clustered 2025 marginal gain:
- observed: `-0.12460`
- 95% CI: `[-0.15259,-0.09771]`
- P(gain>0): `0.000`

This is a strong rejection of this exact transmission law.

## Scientific interpretation

This result does **not** invalidate:
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`; or
- `TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

Those two results establish that the named TE's own history improves player-level uncertainty relative to pooled TE uncertainty, including after team pass-volume error is removed.

This result answers a different question:

> Can that confirmed player-level width be converted directly into a Dirichlet/Beta-Binomial overdispersion parameter and inserted into a finite conserved target room?

For the exact frozen conversion, the answer is **no**.

The normal-MAE player width is not automatically a valid beta-binomial/Dirichlet overdispersion parameter. Equal mean and an algebraically matched variance do not preserve probabilistic calibration across distribution families.

## Closure

Close this exact:
- normal-width -> count-variance;
- count-variance -> `rho_i`;
- weighted room `rho`;
- Dirichlet-multinomial / Beta-binomial

transmission.

Do not rescue by:
- scaling or clipping rho;
- fitting a post-hoc conversion coefficient;
- TE1/TE2 subsets;
- changing prior windows/support;
- pooling 2024/2025;
- weakening player-level gates.

## Remaining legitimate implication

The confirmed TE individual uncertainty remains usable scientific evidence, but any future integration must preserve its **marginal calibration semantics** rather than assume a different parametric joint law is equivalent.

A future lane is authorized only if it is scientifically distinct from this failed variance-matching Dirichlet conversion and is frozen before scoring.

No production change.
