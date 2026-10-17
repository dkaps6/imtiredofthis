# NFL HANDOFF — 2026-10-16 — PRISM TEAM HIGH-VALUE CREATION CLOSED / DRIVE-TOPOLOGY MECHANISM NEXT

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Issue:** #535 — live research ledger

This handoff supersedes the denominator-persistence-next checkpoint.

## 1. Player-first mandate

PRISM is an **individual-player projection model**.

Canonical reasoning order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA. The named player is the scientific
object.

The user's "Construct NFL Tickets" clarification is binding context:
- know who the player has historically been;
- establish his current role, availability and trajectory;
- account for teammates and finite opportunity competition;
- identify the type/quality of opportunity he is likely to receive;
- evaluate the actual opponent/game environment and measurable matchup;
- translate that into the specific market being projected;
- preserve player-specific reliability/volatility;
- downstream tickets must be built from approved PRISM player plays rather than
  raw edge sorting.

A team or position aggregate may describe the named player's environment. It is
not the modeling object.

## 2. New canonical milestone

`PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`

Frozen contract:
`docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_FREEZE.md`

First/binding freeze:
`b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`

Canonical execution:
- run `38999998874` — **SUCCESS**;
- job `117300709440`;
- exact head `2f5ff029f017bfd7b25d667be9d56b3905f87c73`;
- artifact `12087739890`;
- digest `sha256:ebee1d07e0edf5ed92335818aa8bb5c2995ccdaa6bdb40be15c267987b0d2b20`;
- strict repo audit PASS;
- result `docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_RESULT.md`.

All three cells are terminally closed:
- RB rush inside-10;
- RB receiving red-zone targets;
- TE receiving red-zone targets.

The candidate was the team's own last-4/min-3 strictly-prior same-season
high-value event rate conditional on the **same** canonical PRISM parent
opportunity mean.

The player side used the **same qualified Stage-1 + Stage-2 active-set expected
allocation** on baseline and candidate.

### 2025 confirmation direction

RB rush inside-10:
- team CRPS gain -0.091510;
- player CRPS gain -0.008304;
- team/game and player/game/equal-player confirmation intervals all below zero.

RB receiving red-zone:
- team CRPS gain -0.155009;
- player CRPS gain -0.000635;
- team confirmation intervals below zero;
- all named-player qualification gates failed.

TE receiving red-zone:
- team CRPS gain -0.155009;
- player CRPS gain -0.001554;
- small expected-count AE improvement did not rescue worsened proper score,
  negative median supported-player gain or failed clustered inference.

Interpretation:

> Recent team scoring-area event rate is not a useful persistence state for
> PRISM's next-game high-value opportunity pool under the frozen construction.

Do not retest it with alternate windows, bands, shrinkage, player subsets,
touchdown outcomes or gate changes.

## 3. Concurrency authority reconciliation

A later overlapping freeze:
`02e25a29e26c0562f298fdd71a517f7602f424cf`

and run:
`38999926446`

also returned CLOSED, but are **NONCANONICAL CONCURRENCY SENSITIVITY ONLY**.

Chronology is decisive:
- `b15573...` freeze: 2026-10-17T03:38:37Z;
- `02e25...` freeze: 2026-10-17T03:40:56Z;
- `02e25...` is a direct child of `b15573...`.

The user required ONE mechanism to be frozen before scoring. Therefore the later
implementation is not an independent second trial and may not be used for
model selection or replication claims.

## 4. Protected prior state

Still protected:
- Stage 1 active-role: QUALIFIED ALL;
- Stage 2 conditional allocation: QUALIFIED for RB rushing / RB receiving /
  TE receiving;
- QB Stage-2 exact V1: CLOSED;
- WR Stage-2: source-blocked/not scored;
- parent denominator persistence: PASS;
- player recent high-value-share persistence: CLOSED;
- Week-6 Sunday/Monday Stage-2 shadow: LOCKED;
- 13 included games / 248 player-cell rows / SEA-DEN excluded;
- **do not grade Week-6 shadow until all 13 included games are final**.

No paid OddsAPI credit is authorized.

## 5. What the negative result means

Do not conclude that opportunity quality is irrelevant.

The failure is specifically the generic shortcut:

`recent team high-value event rate -> next-game high-value count`.

That shortcut worsened the team pool before player allocation.

For PRISM, that is useful because it prevents a noisy team-average trend from
being inserted upstream of named-player projections.

## 6. Exact next action

Do **not** rescue the closed high-value event-rate family.

The next genuinely different, already source-ready opportunity-quality frontier
is **team scoring-territory drive creation**.

Source-ready raw fields already exist:
- `offensive_drives`;
- `drives_reached20`;
- `drives_reached10`.

The next mechanism must be frozen **before** scoring and must:
- remain conditional on the same canonical `authority_denominator_mean`;
- preserve the same qualified Stage-1 + Stage-2 named-player allocation;
- use 2024 development / 2025 independent confirmation;
- use drive penetration as the team quality state rather than direct recent
  high-value event rate;
- use no player recent high-value share;
- use no opponent/environment transmission yet;
- use no TD outcomes;
- use no sportsbook;
- call no OddsAPI;
- use no 2026 outcomes;
- make no production change;
- predeclare support, proper score, dependence-aware inference and anti-rescue
  before scoring.

This must be genuinely distinct:
the team-specific candidate may use scoring-territory drive penetration, but
may **not** use the team's recent inside-10 rush or red-zone target rate as its
predictive state.

## 7. Integrity

Canonical closed run:
- target outcomes used as predictors: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- fitted coefficients: 0;
- production changed: false.

Continue autonomously from the drive-topology freeze. Do not restart completed
research and do not grade Week 6 early.
