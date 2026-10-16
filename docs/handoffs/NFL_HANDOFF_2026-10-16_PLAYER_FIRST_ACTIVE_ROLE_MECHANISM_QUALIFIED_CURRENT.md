# NFL HANDOFF — 2026-10-16 — PLAYER-FIRST ACTIVE-ROLE MECHANISM QUALIFIED ALL

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Canonical research status:** `PLAYER_INDIVIDUALIZATION_PARTIAL`

## 1. Non-negotiable objective

This is an **individual-player projection model**.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position MAE/RMSE is downstream QA. Named-player football state is the scientific object.

No sportsbook information may enter the upstream football model.
Do not spend another Week-6 OddsAPI credit without explicit user authorization.

## 2. Current scientific milestone

Stage 1 active-role / zero-state participation is now scientifically qualified.

Canonical result:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Per position:
- `QB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `RB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `WR_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `TE_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`

Frozen-before-scoring contract:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE.md`

Pre-score freeze commit:
`16df6e95388a5dd7d682a619ea0b6f49d50f37b4`

Result:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_RESULT.md`

Canonical scoring authority:
- run `38907459996` — SUCCESS;
- job `116994537005`;
- exact scored head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

Parent source authority remains:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`;
- run `38895205962`;
- job `116953914726`;
- source head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`;
- artifact `12044155086`;
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`.

## 3. Exact qualified Stage-1 mechanism

Pregame universe:
weekly roster ACT + stable identity - exact official inactives, restricted to valid scheduled team-weeks with a valid snap source.

Target:
- 1 = any offensive snap;
- 0 = zero offensive snaps.

Target-game participation is an outcome only and never defines pregame membership.

Baseline:
strictly-prior same-season position-family participation rate with Jeffreys smoothing:

`p_baseline = (K_pos + 0.5) / (N_pos + 1)`

with `N_pos >= 100`.

Named-player state:
- same stable player;
- same current team;
- most recent up to 4 eligible strictly-prior games;
- minimum 3;
- binary prior participation occurrence only.

Candidate:

`p_candidate = (k_player + 2 * p_baseline) / (n_player + 2)`

The prior strength 2 and every support/inference gate were frozen before scoring.

## 4. 2025 independent confirmation

Brier improvement, baseline minus candidate:
- QB: +0.092720
- RB: +0.038602
- WR: +0.015566
- TE: +0.006296

Every position passed:
- 2025 support;
- positive Brier improvement;
- non-negative log-loss improvement;
- player-cluster 95% CI lower bound > 0;
- game-cluster 95% CI lower bound > 0;
- equal-player mean improvement > 0;
- equal-player bootstrap 95% CI lower bound > 0.

Named-player share improved in 2025:
- QB 88.41%;
- RB 89.47%;
- WR 82.22%;
- TE 88.89%.

2024 development and pooled corroboration also passed all frozen gates.

The result is not universal at every named player. Individual counterexamples remain and are preserved.

## 5. Scientific meaning

The model now has qualified evidence for a separate upstream player state:

`P(any offensive snap | named player's strictly-prior same-team participation history)`.

This directly supports the Player Projection First Principle.

Stage 1 does not answer opportunity amount.

Keep the architecture two-stage:

1. **Stage 1:** active-role / zero-state participation probability — QUALIFIED.
2. **Stage 2:** conditional opportunity allocation given participation — NOT YET FROZEN/QUALIFIED as this new bridge.

Do not multiply projections by Stage-1 probability, remove players, or zero them automatically without a separately frozen deployment bridge.

## 6. Protected research progression

Do not reopen:
- player role elasticity from teammate availability V1 — `PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`, run `38847501580`, artifact `12024335417`; do not lower the <90% strictly-prior same-team history gate;
- simple high-value opportunity persistence V1 — `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`, run `38848538570`, artifact `12024037594`; do not rescue with alternate bands/windows/subsets/TD outcomes;
- Availability -> Opportunity rule-order gap;
- M78/M79 generic inactive-QB identity;
- RB Vacancy;
- RB carry/snap allocation;
- RB receiving-room share;
- WR/TE target-share trajectory;
- WR-R15 / TE-R5P snap entitlement;
- route-participation/YPRR source work;
- generic QB mean/YPA after M90;
- all other terminal anti-retest lanes in Issue #535.

High-value scoring-opportunity source remains source-ready but simple persistence is closed.

## 7. WR / TE trust continuity

Preserve:
- `WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`;
- frozen Week-6 WR trust lock;
- `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`;
- frozen Week-6 TE trust lock.

Do not change frozen Week-6 labels from target-week outcomes.

Generic ATD remains uncertified.

## 8. Week-6 OddsAPI guardrail

Exactly one authorized paid Week-6 acquisition has already been used.

Do not call OddsAPI again without explicit user authorization.

The active-role research used:
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production mutation: false.

## 9. Exact next scientific action

The existing allocation roadmap has now been reconciled.

Do **not** launch a generic recent-share allocator. That would duplicate or
reopen existing RB carry/receiving, WR/TE trajectory, WR Dirichlet, TE
allocation-difficulty, personnel-conditioned and high-value-persistence work.

The next distinct seam is:
`qualified Stage-1 active-role probability -> active-set state -> existing canonical conditional entitlement -> finite opportunity allocation`.

Frozen source/lineage plan:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md`.

Before any bridge scoring, audit 2024/2025 historical as-of/OOS-compatible
canonical parent authority separately for:
- QB passing opportunity;
- RB rushing opportunity;
- RB receiving opportunity;
- WR target opportunity;
- TE target opportunity.

A cell whose current canonical historical entitlement cannot be proven is
source-blocked. Do not substitute a new share model to rescue it.

Only source-ready cells may advance to a separately frozen scored bridge.
That later freeze must predeclare:
- active-set sampling law from the already-qualified Stage-1 probability;
- all-off / degenerate-room fallback;
- exact conditional renormalization of existing canonical entitlement;
- finite team/room conservation;
- score and support gates;
- player/game dependence-aware inference.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.
Do not collapse Stage 1 and Stage 2.

## 10. Operating style

GitHub is canonical.
Fail closed rather than weaken gates.
Do not restart completed research.
Do not advertise source readiness as predictive proof.
Do not advertise qualification as production promotion.
Record terminal failures and move forward.

The objective remains to project **this player, in this role, in this game, against this environment**.
