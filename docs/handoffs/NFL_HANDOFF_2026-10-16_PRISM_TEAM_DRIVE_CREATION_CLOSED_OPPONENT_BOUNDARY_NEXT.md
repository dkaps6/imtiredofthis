# NFL HANDOFF — 2026-10-16 — PRISM TEAM DRIVE CREATION CLOSED / OPPONENT-ENVIRONMENT BOUNDARY NEXT

Model:
**PRISM — Player Role & Individual Situation Model**

Repository:
`dkaps6/imtiredofthis`

Research branch:
`research-individual-opportunity-roadmap-2026-10-08`

PR:
#672 — open / draft / research only.

This handoff supersedes the prior drive-topology-next checkpoint.

## 1. Canonical philosophy

PRISM is an **INDIVIDUAL PLAYER PROJECTION MODEL**.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA. The objective remains the named player's actual current football situation.

## 2. Protected upstream state

Do not reopen:

- Stage 1 active-role participation: **QUALIFIED ALL**.
- Stage 2 conditional allocation:
  - RB rushing: QUALIFIED;
  - RB receiving: QUALIFIED;
  - TE receiving: QUALIFIED.
- QB Stage-2 exact V1: CLOSED.
- WR Stage-2: source-blocked / not scored.
- player recent high-value scoring-area share persistence: CLOSED.
- Cleveland identity blocker: CLOSED.
- canonical parent denominator persistence: PASS.
- direct team recent high-value event-rate persistence: CLOSED.
- team scoring-territory drive creation V1: CLOSED.

Week-6 Sunday/Monday Stage-2 shadow remains locked:
- 13 included games;
- 248 player-cell rows;
- SEA-DEN excluded;
- zero Week-6 outcomes read at lock.

Do **not** grade it until all 13 included games are final.

No paid OddsAPI credit is authorized.

## 3. Parent denominator authority

Disposition:

`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`

Canonical:
- run `38991915503` — SUCCESS;
- job `117275946409`;
- exact head `02e96ad9faabee62c1ecb31303b48746e4945a8c`;
- artifact `12085194208`;
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`.

14,284 rows.
Exact keys / finite flags / ready cells / scientific disposition.
Max canonical entitlement-q drift:
`2.0816681711721685e-16`.

Persisted denominator:
`authority_denominator_mean`.

## 4. Opportunity-quality raw/source authority

Raw finite topology:
- run `38991605252`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`;
- disposition `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`.

Reconciled:
- run `38991737241`;
- artifact `12085278127`;
- digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`;
- disposition `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`.

Drive topology is source-ready:
- `offensive_drives`;
- `drives_reached20`;
- `drives_reached10`.

Strict-prior opponent-state source attachment is also source-ready, but was intentionally not used in the two team-side mechanism tests below.

## 5. First team-side opportunity-quality mechanism — CLOSED

Frozen contract:
`docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_FREEZE.md`

Binding first freeze:
`b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`.

Result:
`docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_RESULT.md`

Canonical score:
- run `38999998874` — SUCCESS;
- exact head `2f5ff029f017bfd7b25d667be9d56b3905f87c73`;
- artifact `12087739890`;
- digest `sha256:ebee1d07e0edf5ed92335818aa8bb5c2995ccdaa6bdb40be15c267987b0d2b20`.

Disposition:

`PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`

All:
- RB rush inside-10: CLOSED;
- RB receiving red-zone: CLOSED;
- TE receiving red-zone: CLOSED.

Candidate was the team's own last-4/min-3 same-season high-value event rate, conditional on the exact canonical parent opportunity mean and unchanged qualified player allocation.

It worsened team proper score in both 2024 development and 2025 confirmation.

A later overlapping freeze `02e25a29...` / run `38999926446` is **NONCANONICAL SENSITIVITY ONLY** because it occurred after the first binding freeze and changed scientific mechanics. It also closed all cells and does not contradict the authority.

Do not retest this family with:
- alternate windows;
- smoothing/shrinkage;
- alternate yardline bands;
- lead-back/TE1 subsets;
- TD rescue;
- weakened gates.

## 6. Second genuinely distinct team-side mechanism — CLOSED

Frozen contract:
`docs/research/PRISM_TEAM_SCORING_TERRITORY_DRIVE_CREATION_V1_FREEZE.md`

Freeze:
`82f74487e68729efdbcc0868968fd6ddccd4eef1`.

Result:
`docs/research/PRISM_TEAM_SCORING_TERRITORY_DRIVE_CREATION_V1_RESULT.md`

Canonical execution:
- run `39000352640` — **SUCCESS**;
- exact head `2baba745aa0fe50a9bcb88600051f28a783a197d`;
- artifact `12088459232`;
- digest `sha256:4816dbb88dd85911999171bad60700ab4eccbd62266396af302d689f661be881`;
- strict repository audit: PASS.

Disposition:

`PRISM_TEAM_DRIVE_CREATION_CLOSED`

All:
- RB rush inside-10: CLOSED;
- RB receiving red-zone: CLOSED;
- TE receiving red-zone: CLOSED.

Mechanism:
- baseline = strictly-prior same-season league high-value rate;
- candidate = baseline × team/league scoring-territory drive-entry ratio;
- RB rushing territory = drives reaching opponent 10;
- receiving territory = drives reaching opponent 20;
- same canonical parent opportunity mean;
- same qualified Stage-1 + Stage-2 player allocation;
- Poisson count law;
- 2024 development / 2025 independent confirmation;
- 5,000 dependence-aware bootstrap replicates.

Key confirmation:

RB rushing 2025:
- team CRPS gain `-0.022685`;
- team AE gain `+0.007145`;
- player CRPS gain `-0.000314`;
- player AE gain `+0.002544`;
- team-cluster CI `[-0.066222, 0.023085]`.

RB receiving 2025:
- team CRPS gain `-0.076567`;
- team AE gain `-0.098317`;
- player CRPS gain `-0.001266`;
- team-cluster CI `[-0.157742, 0.006453]`.

TE receiving 2025:
- team CRPS gain `-0.076567`;
- team AE gain `-0.098317`;
- player CRPS gain `+0.000770`;
- player AE gain `+0.003882`;
- equal-player mean `+0.000360`;
- team game-cluster CI `[-0.155746, -0.000712]`;
- player/equal-player dependence-aware intervals cross zero.

The small positive TE player point estimate is **not** a qualification. The frozen finite-pool chain requires upstream team-pool improvement and dependence-aware player confirmation. Do not cherry-pick TE.

## 7. What the two closures mean

The problem is now narrowed.

PRISM's:
- canonical parent opportunity quantity remains protected;
- active-role participation remains protected;
- conditional player allocation remains protected.

The failure appears in creating the **team finite high-value opportunity pool** from simple same-team recent persistence.

Two distinct team-side states failed:
1. direct recent high-value event-rate persistence;
2. scoring-territory drive-entry persistence.

The drive mechanism was generally less harmful than direct event-rate persistence, but “less bad” is not predictive qualification.

Do not reopen these formulas.

## 8. Integrity

Across the canonical drive run:
- target-game outcomes used as predictors: 0;
- fitted predictive coefficients: 0;
- threshold searches: 0;
- direct recent-team high-value-rate candidate: false;
- player recent high-value-share candidate: false;
- opponent/environment used: false;
- TD outcomes used: false;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false;
- denominator drift baseline/candidate: 0;
- player-allocation drift baseline/candidate: 0.

## 9. Current boundary / exact next action

The next conceptual architecture layer is:

`OPPONENT / ENVIRONMENT`.

However the governing user instruction for this opportunity-quality phase explicitly said:

> Do NOT add opponent/environment transmission yet.

That boundary has been respected.

Therefore:
- do not score an opponent/environment candidate until this “yet” boundary is deliberately crossed in a subsequent continuation;
- do not invent a third same-team persistence rescue;
- do not swap drives for plays-inside-territory merely to salvage the failed drive formula;
- do not use TD outcomes;
- do not use sportsbook information;
- do not call OddsAPI;
- do not use 2026 outcomes for retrospective mechanism discovery;
- do not mutate production;
- do not grade Week 6 before all 13 locked games are final.

When the opponent/environment boundary is opened, freeze **one** genuinely distinct mechanism before scoring, with:
- 2024 development;
- 2025 independent confirmation;
- exact support;
- exact dependence-aware inference;
- anti-rescue;
- same canonical parent denominator;
- same qualified player allocation.

Do not tune after seeing results.
