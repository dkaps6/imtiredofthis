# NFL HANDOFF — 2026-09-26 — SYSTEMS AUDIT CLOSED / DISCRETE COUNT V1 PROMOTED CURRENT

GitHub is canonical. This handoff supersedes the prior Week-3 repaired-systems handoff for immediate execution. Do not recursively load older handoffs unless this file explicitly directs it.

## Canonical production state

Production code main before the docs-only continuity commit:
`d658a176d86654e3c153d09d2fcf1e69eca4aaec`

Production lineage:
- PR #645 — Week-3 Full Slate production repair — MERGED.
- PR #646 — Rush-Att Zero-MC Allocation Lineage V1 diagnostic closure — MERGED.
- PR #647 — Discrete Count Mean Alignment V1 production promotion — MERGED.

Post-promotion verification on `d658a176...`:
- Repo CI `36286648937` = **SUCCESS**
- canonical no-live-odds Full Slate `36286648933` = **SUCCESS**
- strict repository audit inside Full Slate = **SUCCESS**
- sportsbook/pricing steps were intentionally skipped because live-odds mode was off.

No paid `fetch_live_odds=true` Full Slate has been launched after this checkpoint.

## Actual betting-board gate

The next operational betting-board gate remains:

`fetch_live_odds=true`

That consumes OddsAPI credits.

**Do not launch it without explicit user authorization in the active chat.**

The no-live stack is green; there is no known football-stack blocker requiring paid debugging.

## Rush-Att Zero-MC Allocation Lineage V1 — CLOSED

Canonical result:
`docs/research/RUSH_ATT_ZERO_MC_ALLOCATION_LINEAGE_V1_RESULT.md`

Authority:
- run `36285507895` = SUCCESS
- artifact `10920712168`
- digest `sha256:487e9a59efffaae45494cf830e30650582e130baae2831507fae7dea9da5959a`

The frozen historical cohort reproduced exactly:
- total zero-MC / nonzero-ensemble rush-att rows = **7,137**
- 2024: QB 271 / RB 74 / TE 1,087 / WR 2,117
- 2025: QB 249 / RB 87 / TE 1,122 / WR 2,130

Every blocked row first becomes zero at:
`TOP5_EXCLUDED`

Hard parity:
- output-row vs simulator-selected rushing-share mismatches = 0
- canonical MC vs keyed lookup mismatches = 0
- keyed lookup vs realized carry-vector mismatches = 0
- reproduced top-five vs in-simulator top-five mismatches = 0

Disposition:
`CLOSED_AS_HARD_SUPPORT_CONFLICT_NO_REPAIR_AUTHORIZED`

Do not rescue with:
- ensemble injection;
- QB/RB carveouts;
- alternate top-N;
- depth/role exceptions;
- share thresholds;
- revived Rush Pool Evidence Guard variants.

## Discrete Count Mean Alignment V1 — PRODUCTION ACTIVE

Frozen research authority now lives on main:
- `docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_PLAN.md`
- `docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_MECHANICAL_AMENDMENT.md`
- `docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_RESULT.md`

Production integration:
- `docs/production/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PLAN.md`
- `docs/production/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_RESULT.md`

Authoritative current-main integration:
- run `36286418528` = SUCCESS
- artifact `10920344026`
- digest `sha256:ac0834f33755eb273f91f4d1f25f12d2a0de088995b14bb95c7d3f09ac161b4c`
- rows checked = 19,845
- V1-applied rows = 12,698
- exact zero/nonfinite-MC no-op rows = 7,147
- integer failures = 0
- max array gap vs frozen research = 0
- all non-count legacy invariance = true

Production behavior:
- applies only to `receptions` and `rush_att`;
- deterministic largest-remainder integer support is applied after the existing continuous mean-alignment guard;
- zero/nonfinite-MC rows remain exact no-ops;
- all non-count markets retain legacy continuous semantics;
- no football mean, coefficient, ensemble weight, rule, or sportsbook-upstream feature changed.

## Frozen prospective lanes — do not redesign pregame

### RB Vacancy Opportunity V1

DEN:
- Jonah Coleman unavailable
- label `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- lead `NONE_CLEAR`

PIT:
- Rico Dowdle unavailable
- label `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
- lead `JAYLEN_WARREN`
- confidence MEDIUM

After games:
1. grade RB Vacancy V1 independently;
2. grade frozen public-intent labels against actual carry/snap concentration;
3. never rewrite pregame labels.

### Receiving Rule Semantics V1

Frozen Week-3 cells:
- A0B0 = current production
- A1B0 = middle-open unit repair
- A0B1 = slot-alignment repair
- A1B1 = combined repair

Confirmed defects remain:
- `middle_open_rate` is percentage-point scaled while production interprets it as 0-1;
- SWR alignment exists upstream but is dropped before production SLOT labeling.

Historical Stage 2 remains:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`

After games, attach outcomes to the unchanged cells and grade. No postgame redesign/rescue.

### Availability -> Opportunity Rule Order

Disposition remains:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Week-3 frozen authority:
- 11 definitive-unavailable skill players
- 6 WR / 3 TE / 2 RB-FB
- zero survived into eligible roles / PlayerForm / ModelContext / production-reachable injury-redistribution rows.

Do not resurrect 60/30/10 and do not fit vacancy coefficients from Week-3 results.

## Production-active improvement that remains locked

RB Rush+Receiving Conservation V2 remains production-active.

Historical 2024-25 RB MAE:
`27.6853 -> 25.5718`

Do not reopen absent a concrete new defect/evidence.

## Closed / do-not-repeat

Do not reopen without genuinely new evidence:
- Receiver Room Targets-per-Play V1
- WR Anchor / Role-Transmission
- TE Receiving-Yards Width V2
- Rush Pool Evidence Guard V1
- Rush Post-Ensemble Reconciliation V1
- residual/top-five normalization rescue
- generic copula rescue
- old receiving-market missing-weight experiment
- retrospective M96 RB rushing family
- rush-att zero-MC repair variants after the lineage closure above.

## Exact next actions

Before Week-3 outcomes are final, there is no additional authorized repair implied by the closed zero-MC lane or the promoted count-support lane.

Valid next actions are:
1. if the user explicitly authorizes OddsAPI spend, run the controlled canonical Full Slate with `fetch_live_odds=true` from live `main` and inspect the real betting-board output;
2. otherwise preserve all Week-3 prospective cells/labels untouched until outcomes are final;
3. after games, grade RB Vacancy/Public Intent and Receiving Rule Semantics against the frozen pregame authorities;
4. continue systems-integrity work only when a genuinely new concrete implementation/data-lineage contradiction is identified; do not manufacture a new candidate merely to keep experimenting.

Sportsbook data remains downstream of football-model science.

## Continuity anchors

Issue #535:
- rush-att final closure: comment `5851689287`
- Discrete Count current-main integration pass: comment `5851733502`
- Discrete Count promotion + post-merge green: comment `5851762896`

If the chat times out, recover from this file + those Issue #535 comments + live GitHub state. Do not make the user re-explain.
