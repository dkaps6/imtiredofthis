# NFL HANDOFF — 2026-10-17 — PRISM WR DUAL RELIABILITY MECHANICS READY / WEEK-7 LOCK PENDING

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Issue:** #535

This handoff supersedes the 2026-10-16 opponent-boundary / reliability-forward
checkpoint for the active frontier.

## 1. Non-negotiable PRISM philosophy

PRISM is an **individual-player projection model**.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-level MAE/RMSE is downstream QA. The named player in his actual current
football situation is the scientific object.

The prior **Construct NFL Tickets** discussion is binding context:
- know this player historically;
- establish his current role, usage, availability and trajectory;
- account for teammate competition and finite opportunity;
- distinguish opportunity quantity from opportunity quality;
- evaluate matchup only through a valid player-conditioned football mechanism;
- map to the exact market;
- preserve player-specific projection reliability;
- only then compare against the market and construct tickets.

Downstream order:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

Reliability is **not** over/under direction.

## 2. Protected opportunity state

Stage 1 active-role:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Stage 2 conditional allocation remains qualified for:
- RB rushing;
- RB receiving;
- TE receiving.

QB exact Stage-2 V1 remains CLOSED.
WR Stage-2 historical scoring remains source-blocked/not scored.

Parent denominator persistence remains PASS:
- run `38991915503`;
- artifact `12085194208`;
- persisted field `authority_denominator_mean`.

Week-6 Sunday/Monday Stage-2 shadow remains immutable:
- 13 games;
- 248 player-cell rows;
- SEA-DEN excluded;
- zero target outcomes read at lock.

Do not grade it until all included games are final.

## 3. Opportunity-quality closures remain terminal

Direct team high-value creation:
`PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`
- freeze `b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`;
- run `38999998874` SUCCESS;
- artifact `12087739890`.

Scoring-territory drive creation:
`PRISM_TEAM_DRIVE_CREATION_CLOSED`
- freeze `82f74487e68729efdbcc0868968fd6ddccd4eef1`;
- run `39000352640` SUCCESS;
- artifact `12088459232`.

Do not rescue either family with alternate windows, bands, shrinkage, subgroups,
TD outcomes, opponent terms or renamed variants.

The overlapping later high-value-rate run from concurrent work is not an
independent trial and does not reopen the canonical closure.

## 4. Opponent/environment boundary remains audited but unopened

Audit:
`docs/research/PRISM_OPPONENT_ENVIRONMENT_BOUNDARY_V1_AUDIT.md`.

Disposition:
`PRISM_OPPONENT_ENVIRONMENT_BOUNDARY_AUDIT_COMPLETE__NO_CLEAN_CANDIDATE_YET`.

No opponent candidate has been scored.

Do not use:
- generic defense-vs-position;
- direct recent opponent high-value allowance;
- symmetric opponent drive allowance;
- simple usage × matchup weakness;
- previously failed Football Matchup Transmission integrations.

A future opponent mechanism must be genuinely player-conditioned on the named
player's current role/opportunity subtype with exact historical/live source parity.

## 5. Player reliability remains the active forward layer

WR opportunity trust:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`.

WR efficiency trust:
`WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`.

WR dual trust:
`WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`.

Frozen dual prospective plan:
`docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1_PLAN.md`.

Fixed target horizon:
**2026 Weeks 7, 8, 9, 10**.

Week 6 is explicitly not a dual-composite target week.
No Week-6 dual backfill.
No optional stopping.

TE allocation trust remains QUALIFIED.
TE efficiency trust remains CLOSED.
Generic efficiency-tail stability remains CLOSED.

## 6. NEWEST MILESTONE — WR dual prospective mechanics READY

Canonical successful mechanics execution:
- run `39001244942` — **SUCCESS**;
- job `117304434689`;
- exact execution head `0fbfd104cbea803d8afbbcc9c56b23414b309c88`;
- artifact `12088664427`;
- digest `sha256:384850e42c28b53171a8f3062998cdf910dcc1714e36d1abea0a1e4acbee7418`;
- focused tests: **4 / 4 PASS**;
- strict repository audit: **PASS**.

Result:
`docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_MECHANICS_V1_RESULT.md`
(on the active research branch).

Disposition:

`WR_DUAL_RELIABILITY_PROSPECTIVE_MECHANICS_READY__NO_WEEK7_LOCK`.

Implemented:
- `scripts/research/lock_wr_dual_reliability_prospective_v1.py`;
- `tests/test_wr_dual_reliability_prospective_v1.py`;
- `.github/workflows/research-wr-dual-reliability-prospective-mechanics-v1.yml`.

### Frozen mechanics

For target Week W:
- exact target-week WR cohort supplied pregame;
- same named player only;
- trailing maximum 8 qualifying games;
- minimum 4 opportunity-history games;
- minimum 4 efficiency-history games;
- opportunity/efficiency MAEs remain separate;
- percentile ranks use `method="average"`;
- `dual_fragility_score = (opportunity_fragility_score + efficiency_fragility_score) / 2`;
- frozen WR-R15 receiving-yard projection is retained on the row;
- HIGH trust <= 0.25;
- LOW trust >= 0.75;
- no fitted weights.

### Strong chronology guard

The builder fails closed unless current-season history is complete through
**exactly target_week - 1**.

For Week 7:
- history stopping at Week 5 fails;
- history must reach legitimately completed Week 6;
- any Week-7-or-later outcome row fails;
- Week-6 dual-composite backfill is prohibited.

This means the mechanics are ready, but no Week-7 scientific row can be created
early.

## 7. Week-6 trust locks remain immutable

WR opportunity trust:
- run `38823520954`;
- artifact `12012884262`;
- supported WRs 141.

WR efficiency trust:
- run `38826804491`;
- artifact `12015596169`;
- supported WRs 141.

TE trust:
- run `38822346088`;
- artifact `12013018536`;
- supported TEs 103.

Every lock read zero Week-6 outcomes at freeze.

Do not:
- grade early;
- relabel trust as over/under;
- manufacture a Week-6 WR dual composite.

## 8. Exact next action

**Do not create the Week-7 lock yet.**

The next scientific action occurs only after:
1. Week 6 is legitimately complete;
2. strict-prior Week-6 player history is available and validated;
3. Week-7 roster/slate/player identity is ready pregame.

Then execute the frozen Week-7 WR dual-reliability lock using the already-built
mechanics.

The Week-7 lock must:
- contain zero Week-7 outcomes;
- retain the exact frozen WR-R15 projection;
- preserve the exact fixed 50/50 reliability formula;
- use no sportsbook;
- call no OddsAPI;
- fit no model;
- change no projection mean or distribution;
- make no production change.

Weeks 7-10 remain the fixed prospective horizon.

## 9. Integrity at this checkpoint

Canonical mechanics run:
- Week-7 scientific rows frozen: **0**;
- Week-6 outcomes read by the mechanics run: **0**;
- Week-7 outcomes read: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- models fit: **0**;
- projection/distribution changes: **0**;
- production changed: **false**.

No paid OddsAPI credit is authorized.

GitHub remains canonical over chat memory.
