# NFL HANDOFF — 2026-10-16 — PRISM STAGE-2 BRIDGE QUALIFIED PARTIAL

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model name:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Canonical research status:** `PLAYER_INDIVIDUALIZATION_PARTIAL`

## 1. Non-negotiable objective

PRISM is an **individual-player projection model**.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-level MAE/RMSE is downstream QA, not the modeling objective.

No sportsbook information may enter the upstream football model.
Do not spend another Week-6 OddsAPI credit without explicit user authorization.

## 2. Stage 1 — qualified

`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Canonical scored authority:
- run `38907459996` — SUCCESS;
- job `116994537005`;
- exact head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

Qualified named-player state:

`P(any offensive snap | named player's strictly-prior same-team participation state)`.

Per-position 2025 Brier improvements:
- QB +0.092720;
- RB +0.038602;
- WR +0.015566;
- TE +0.006296.

## 3. Stage-2 source audit — ready partial

Canonical source result:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

Authority:
- run `38955368643` — SUCCESS;
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`;
- artifact `12070806828`;
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`.

Source-ready:
- QB passing;
- RB rushing;
- RB receiving;
- TE receiving.

Source-blocked:
- WR receiving because 2025 WR-R15 cannot legally be treated as independent OOS authority.

## 4. Stage 2 — historical bridge result

Frozen-before-scoring:
- commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md`.

Canonical scored authority:
- run `38957575645` — SUCCESS;
- job `117164201650`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

Overall:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Qualified:
- `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Closed:
- `QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`

Not scored:
- WR receiving.

Result:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_RESULT.md`.

## 5. 2025 independent confirmation

RB rushing:
- CRPS +0.385893;
- expected-count AE +0.267115;
- room error +0.008439;
- player-cluster CI low +0.297996;
- game-cluster CI low +0.333263;
- equal-player CI low +0.277780;
- 86.78% of evaluable players improved.

RB receiving:
- CRPS +0.034723;
- expected-count AE +0.050076;
- room error +0.000975;
- player-cluster CI low +0.019115;
- game-cluster CI low +0.027288;
- equal-player CI low +0.015982;
- 47.11% of evaluable players improved.

TE receiving:
- CRPS +0.018198;
- expected-count AE +0.018672;
- room error +0.000395;
- player-cluster CI low +0.009962;
- game-cluster CI low +0.013674;
- equal-player CI low +0.008875;
- 57.89% of evaluable players improved.

Preserve named-player heterogeneity. Qualification does not imply every player improves.

## 6. QB bridge — terminal for exact V1

QB V1 is closed.

It improved pooled CRPS but worsened expected pass-attempt mean and room allocation in both seasons and failed 2025 player/game/equal-player confirmation.

Do not rescue with:
- p_active thresholds;
- correlations;
- starter carveouts;
- backup reassignment;
- new pass-share coefficients;
- pooled evidence.

Preserve M89/M90/C2 and all existing QB stopping rules.

## 7. WR — blocked, not failed

WR was not scored.

2024 OOS authority exists, but 2025 WR-R15 final fit has training overlap and cannot be relabeled OOS.

Do not score 2024 alone or weaken the source contract.

## 8. Protected terminal lanes

Do not reopen:
- simple high-value-opportunity persistence V1;
- role-elasticity V1 source block;
- generic recent-share allocators;
- RB carry/snap or receiving-room formulas;
- WR/TE target-share trajectory;
- generic QB mean/YPA/attempt families;
- generic defense-vs-position / Vegas / game-script rescues;
- terminal lanes catalogued in Issue #535.

WR/TE difficulty trust qualifications and Week-6 trust locks remain frozen.

Generic ATD remains uncertified.

## 9. Exact next action

Do **not** production-promote the historical bridge.

Freeze a separate **2026 Week-6 prospective PRISM Stage-2 shadow** for:
- RB rushing;
- RB receiving;
- TE receiving.

The prospective shadow must:
1. use football-only pregame state;
2. use the exact qualified Stage-1 probability semantics;
3. preserve the current canonical entitlement and team opportunity volume;
4. apply the exact qualified V1 active-set / complement bridge with no refit;
5. lock player rows and outputs before target-game outcomes;
6. record immutable source/run hashes;
7. make zero sportsbook inputs upstream;
8. make zero new OddsAPI calls;
9. read zero Week-6 outcomes at lock time;
10. make zero production mutations.

QB must remain baseline under this bridge.
WR must remain baseline under this bridge.

First reconcile whether the preserved Week-6 production artifacts provide all required football-only inputs without consuming sportsbook fields. If not, build a no-live-odds football-only source path; do not spend another paid credit.

After the prospective lock, continue the PRISM hierarchy. The next distinct science layer after qualified opportunity quantity is opportunity quality/topology, but any experiment must be genuinely distinct from the closed simple high-value-persistence V1.

## 10. Integrity

Stage-2 historical run:
- target-game outcomes used as predictors: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- new share models fit: 0;
- threshold searches: 0;
- production changed: false.

GitHub remains canonical.
Fail closed rather than weaken a gate.
