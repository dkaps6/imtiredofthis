# PRISM 2026 Player-First Replay — Weeks 4-5 V1 Protocol

Date: 2026-10-17

Status: **FROZEN DIAGNOSTIC REPLAY PROTOCOL / RESEARCH ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

## 1. Governing question

This replay exists to test the model at its actual scientific object:

> Did PRISM understand this named player, in this player's specific pregame football situation, well enough to produce the correct market-specific football projection?

The replay is not a position-average optimization exercise.

Canonical player-first order:

`PLAYER IDENTITY / HISTORY -> CURRENT ROLE / STATE / TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE / QUALITY -> PLAYER-CONDITIONED OPPONENT / ENVIRONMENT -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position MAE/RMSE remains downstream QA only.

## 2. Why this replay starts at Week 4

Weeks 1-3 provide current-season state before Week 4, while prior-season / career history may provide legitimate player-identity context when that source existed before kickoff.

Week 4 is therefore the first replay target for the broader PRISM player-first diagnostic.

The separately frozen WR dual-reliability mechanism requires a minimum of four completed same-player current-season games. Under that exact rule, its first eligible retrospective target is Week 5. That requirement does **not** prohibit broader player-first Week-4 analysis.

Do not manufacture a Week-4 dual-reliability score that did not satisfy its own minimum-history contract.

## 3. Target weeks

Initial replay target weeks:
- 2026 Week 4;
- 2026 Week 5.

Weeks 1-3 are strict-prior current-season context for Week 4.
Weeks 1-4 are strict-prior current-season context for Week 5.

Week 6 remains protected by its existing immutable prospective/shadow protocols and is not graded by this replay before the canonical Week-6 completion boundary.

## 4. Preserved authorities

### Week 4

Canonical recovered pregame board:
- recovery run `36935917903`;
- artifact `11197776900`;
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`;
- 873 player-market rows / 16 games.

Canonical expanded postmortem:
- run `37485600879`;
- artifact `11423535484`;
- digest `sha256:59185d2aa4178971f6b0cc806eddd172ca0e7fbae559ad44da37a2b99656d791`.

### Week 5

Canonical preserved pregame full slate:
- run `37860592615`;
- artifact `11585973062`;
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`.

Canonical full-model postmortem:
- run `38395384228`;
- artifact `11812200405`;
- digest `sha256:5888cab39506ae0bf70034dfb60f297defd28dd3219ea8118870738706f55ac3`.

Canonical Week-5 player failure casebook:
- run `38397361891`;
- artifact `11812403368`;
- digest `sha256:4cdef29912f35368473d57e15b38efdc3dd0f75e897bbbdc3b97fe1ff8f67670`.

No live odds reacquisition is permitted for this replay.

## 5. Unit of analysis

Primary unit:
**named player x target game x market**.

Every diagnostic must retain:
- stable player identity;
- target week/game;
- team/opponent;
- position only as contextual metadata;
- exact market;
- frozen pregame projection;
- realized outcome only after the pregame state is fixed.

Do not collapse the scientific conclusion to a position average.

## 6. Required player-first audit layers

For every prioritized replay case, inspect in order:

1. **Player identity / historical playbook**
   - prior seasons/career behavior available pregame;
   - historical role ranges;
   - historical opportunity/efficiency behavior;
   - known player-specific volatility/reliability.

2. **Current role / state / trajectory**
   - strictly-prior 2026 participation;
   - recent snaps/routes/carries/targets where source-authorized;
   - starter/backup/committee/hierarchy state;
   - role trend without target-game leakage.

3. **Teammate competition / availability**
   - finite room/team opportunity;
   - injuries/inactives/returns;
   - competition for the same opportunity subtype;
   - QB/offensive personnel state where relevant.

4. **Opportunity allocation**
   - team opportunity denominator;
   - named-player share/allocation;
   - whether PRISM under/overestimated actual opportunity.

5. **Opportunity type / quality**
   - rush vs receiving vs pass;
   - scoring-area/high-value role when source-authorized;
   - route/carry subtype only when exact historical/live parity exists.

6. **Opponent/environment**
   - only player-conditioned football mechanisms;
   - no generic defense-vs-position multiplier;
   - identify source gaps when the correct player-conditioned matchup object is unavailable.

7. **Individual efficiency**
   - named-player conversion assumptions;
   - opportunity-to-yard/reception conversion;
   - distinguish opportunity error from efficiency error.

8. **Reliability**
   - use qualified same-player trust evidence where legally available;
   - Week 5 may evaluate the exact four-current-game minimum where the frozen component semantics are reconstructable;
   - Week 4 may use broader historical identity/reliability context but may not fake the later four-current-game dual score.

9. **Exact market mapping**
   - carries/targets/receptions/yards/rush+rec/pass yards/TD concepts are distinct questions;
   - do not infer one market from another without a valid bridge.

## 7. Repeat-player failure registry

A core replay output is a repeat-player registry.

Question:

> Did a named player who produced an extreme PRISM miss earlier in the season appear again among later extreme player-market misses?

This is intentionally player-specific.

For every repeat identity report:
- prior extreme-miss week(s);
- prior market(s);
- target-week market;
- direction of error;
- opportunity vs efficiency attribution where available;
- whether the later miss repeats the same mechanism or changes mechanism.

A repeat identity is **not** automatically a mean correction.
Examples may indicate:
- persistent role under-allocation;
- persistent efficiency misread;
- unstable/fragile player identity;
- changing mechanism across weeks.

## 8. Outcome-use boundary

Outcomes may be used to:
- score frozen pregame projections;
- decompose projection failure;
- identify repeat-player failures;
- label diagnostic failure modes.

Outcomes may **not** be used to:
- select a coefficient;
- select a threshold;
- invent a new role rule;
- choose a matchup multiplier;
- fit a player-specific correction;
- rewrite a historical projection;
- promote production.

Any new candidate mechanism discovered by the replay requires a separate anti-retest audit, freeze, and independent historical/prospective evaluation.

## 9. Market / betting boundary

Archived sportsbook lines may be retained only as downstream comparison evidence.

They may not feed upstream player-state reconstruction.

Order remains:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

No paid OddsAPI call.
No sportsbook refetch.
No production mutation.

## 10. Initial outputs

V1 must produce:

1. named-player repeat-failure registry across Weeks 1-4 -> Week 5;
2. Week-4 prioritized player forensic queue;
3. Week-5 prioritized player forensic queue;
4. player-level opportunity vs efficiency decomposition where authority exists;
5. explicit `KNOWN_BUT_NOT_TRANSMITTED`, `SOURCE_NOT_AVAILABLE`, `STATE_CHANGE_NOT_CAPTURED`, `EFFICIENCY_TAIL`, `RELIABILITY_WARNING`, or `NO_CLEAR_MECHANISM` annotations only when evidence supports them;
6. a mechanism-frequency summary **after** individual cases are adjudicated.

The mechanism summary is downstream of the player cases, not the reverse.

## 11. Scientific interpretation boundary

The purpose is not to prove that a position is bad or that every large miss is fixable.

The purpose is to find recurring football-information failures such as:

> PRISM repeatedly under-allocated this named player's opportunity despite pregame evidence that his current role supported more.

or:

> PRISM had the opportunity approximately right but this named player's efficiency was historically fragile and the projection was treated too confidently.

or:

> The relevant opponent effect depended on this player's actual role/opportunity subtype, but no source-parity object currently represented it.

Those are PRISM-level findings.

## 12. Integrity

At protocol freeze:
- new paid OddsAPI calls: **0**;
- sportsbook refetches: **0**;
- Week-6 outcomes read for this replay: **0**;
- production changes: **0**;
- projection means changed: **0**;
- new coefficients fit: **0**;
- new thresholds fit: **0**.

Disposition:

`PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_PROTOCOL_FROZEN`.
