# NFL HANDOFF — 2026-10-16 — PRISM OPPONENT BOUNDARY AUDITED / PLAYER RELIABILITY FORWARD

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Issue:** #535

This handoff supersedes the earlier drive-creation-closed checkpoint.

## 1. Non-negotiable model philosophy

PRISM is an **INDIVIDUAL PLAYER PROJECTION MODEL**.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position averages are downstream QA.

The named player in his current football situation is the modeling object.

The prior “Construct NFL Tickets” discussion is reflected in this architecture:
- know the player historically;
- establish current role, usage, availability and trajectory;
- model teammate competition and finite opportunity;
- distinguish opportunity quantity from opportunity quality;
- evaluate the opponent only through a valid player-conditioned mechanism;
- map to the specific market;
- preserve named-player reliability;
- only then compare to the market and construct tickets.

Reliability is not over/under direction.

## 2. Protected opportunity/allocation state

- Stage 1 active-role: `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.
- Stage 2 conditional allocation:
  - RB rushing QUALIFIED;
  - RB receiving QUALIFIED;
  - TE receiving QUALIFIED.
- QB Stage-2 V1 CLOSED.
- WR Stage-2 source-blocked/not scored.
- parent denominator persistence PASS:
  - run `38991915503`;
  - artifact `12085194208`;
  - field `authority_denominator_mean`.
- Week-6 Sunday/Monday Stage-2 shadow remains LOCKED:
  - 13 games;
  - 248 player-cell rows;
  - SEA-DEN excluded;
  - zero target outcomes read at lock.

Do not grade Week 6 until all included games are final.

## 3. Opportunity-quality mechanism 1 — CLOSED

`PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`

Canonical:
- freeze `b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`;
- run `38999998874` SUCCESS;
- exact head `2f5ff029f017bfd7b25d667be9d56b3905f87c73`;
- artifact `12087739890`.

The team's own last-4/min-3 high-value event rate worsened the finite team pool.

Do not rescue with new windows/bands/shrinkage/subsets.

## 4. Opportunity-quality mechanism 2 — CLOSED

`PRISM_TEAM_DRIVE_CREATION_CLOSED`

Canonical:
- freeze `82f74487e68729efdbcc0868968fd6ddccd4eef1`;
- run `39000352640` SUCCESS;
- exact head `2baba745aa0fe50a9bcb88600051f28a783a197d`;
- artifact `12088459232`;
- digest `sha256:4816dbb88dd85911999171bad60700ab4eccbd62266396af302d689f661be881`.

The scoring-territory drive-entry mechanism also failed all three cells.

Small positive TE player point estimates did not qualify because the upstream
team pool worsened and dependence-aware confirmation failed.

## 5. Opponent/environment boundary audit — COMPLETE, NOT OPENED

Audit:
`docs/research/PRISM_OPPONENT_ENVIRONMENT_BOUNDARY_V1_AUDIT.md`

Commit:
`dcab49fbdae58d0a54fc422321936b1f45cb6ca0`

Disposition:

`PRISM_OPPONENT_ENVIRONMENT_BOUNDARY_AUDIT_COMPLETE__NO_CLEAN_CANDIDATE_YET`

No opponent candidate was scored.

The explicit “do not add opponent/environment yet” instruction remains intact.

### Why no generic opponent test is authorized

Historical Football Matchup Transmission already established:
- opponent context can contain real residual signal;
- simple integration can still make the player model worse.

Phase B/C:
- run `37514137803`;
- artifact `11436786668`.

Frozen integrations:
- run `37519246784`;
- artifact `11440195795`.

All three simple integration candidates failed closed:
- RB opponent pass-rate-faced;
- WR offensive true PROE pass-share replacement;
- TE opponent pass-success multiplier.

Also protected:
- M95A/M95B generic role × run-defense;
- QB M56/M83 generic defense adaptation;
- defender-injury source parity;
- WR-CB assignment;
- public position-YPT / coverage / box proxies without exact parity.

Phase C usage × matchup interactions did not replicate across both seasons.

Therefore:
- direct opponent high-value rate is source-ready but too generic / symmetric
  with newly closed team persistence to authorize;
- opponent territory-drive allowance is source-ready but equally generic;
- “bad defense vs position => boost player” remains forbidden.

A future opponent mechanism must be genuinely player-conditioned on current
role/opportunity subtype with exact historical/live parity.

## 6. Player reliability frontier — RECONCILED

Audit:
`docs/research/PRISM_PLAYER_RELIABILITY_FRONTIER_V1_AUDIT.md`

Disposition:

`PRISM_PLAYER_RELIABILITY_FRONTIER_RECONCILED__PROSPECTIVE_LOCKS_GOVERN_NEXT`

### WR opportunity trust — QUALIFIED

- run `38823063793`;
- artifact `12013153664`;
- 2024 HIGH_TRUST abs error `14.78 yd`;
- LOW_TRUST `30.45 yd`;
- player/game clustered confirmation strongly positive.

### WR efficiency trust — QUALIFIED

- run `38826418315`;
- artifact `12015303274`;
- incremental beyond opportunity difficulty;
- 2024 HIGH_EFFICIENCY_TRUST `15.72 yd`;
- LOW_EFFICIENCY_TRUST `29.60 yd`.

### WR dual trust — HISTORICALLY SUPPORTED / PROSPECTIVE REQUIRED

- run `38827426902`;
- artifact `12015229072`;
- fixed 50/50 opportunity + efficiency fragility score;
- 2024 dual association `0.32334` vs opportunity-only `0.28453`;
- clustered incremental gain passed.

Frozen prospective plan:
`docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1_PLAN.md`.

Target weeks:
**2026 Weeks 7-10**.

Week 6 is explicitly not part of the composite.

### TE opportunity/allocation trust — QUALIFIED

- run `38821176218`;
- artifact `12013570473`;
- 2025 HIGH_TRUST abs error `8.70 yd`;
- LOW_TRUST `23.60 yd`;
- dependence-aware confirmation passed.

TE efficiency trust exact V1 CLOSED.

Generic player efficiency-volatility V1 CLOSED.

## 7. Week-6 prospective trust locks — IMMUTABLE

WR opportunity trust:
- run `38823520954`;
- artifact `12012884262`;
- supported WRs 141.

WR efficiency trust:
- run `38826804491`;
- artifact `12015596169`;
- supported WRs 141.

TE allocation trust:
- run `38822346088`;
- artifact `12013018536`;
- supported TEs 103.

Every lock read zero Week-6 outcomes.

Do not:
- grade early;
- change labels;
- make a Week-6 WR dual composite;
- infer over/under from trust.

## 8. Ticket-layer interpretation

Preserve this order:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

Trust can eventually help decide:
- which valid projection is strong enough to anchor a ticket;
- which apparently large edge is too fragile;
- where an alt line is preferable;
- how much concentration on one player is reasonable.

Trust is not an upstream football mean and is not a sportsbook-conditioned
selector.

## 9. Exact forward boundary

Right now:
- do not score another opponent mechanism;
- do not reopen either team opportunity-quality persistence family;
- do not grade Week 6 early;
- do not call OddsAPI;
- do not use sportsbook upstream;
- do not mutate production.

The next predetermined prospective player-reliability action is:
- begin WR dual reliability confirmation with **Week 7** under the already-frozen
  Weeks 7-10 plan;
- no Week-6 dual backfill;
- no optional stopping;
- Week-7 target lock must be frozen before Week-7 outcomes and may use Week-6
  history only after Week 6 is legitimately complete.

Before that target artifact can exist, implementation mechanics may be prepared
but no Week-7 scientific row may be frozen from incomplete Week-6 history.

Parallel source work may investigate richer **player-conditioned** opponent
objects, but source readiness alone may not cross the opponent boundary.

## 10. Integrity

Newest audits:
- opponent candidates scored: 0;
- Week-6 outcomes read: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- fitted coefficients: 0;
- production changed: false.

No paid OddsAPI credit is authorized.
