# NFL HANDOFF — 2026-09-27 — PAID WEEK-3 BOARD CERTIFIED / RB LINEAGE SEMANTICS CURRENT

GitHub is canonical. This handoff supersedes `docs/handoffs/NFL_HANDOFF_2026-09-26_SYSTEMS_AUDIT_COUNT_PROMOTED_CURRENT.md` for immediate execution.

Do not recursively load older handoffs unless this file explicitly directs it.

## Canonical current state

Production code main before the continuity/docs-only commit:
`6a1de05842a03272e51269d35a7a01ca8dd1ab12`

Current production verification on that exact code main:
- Repo CI `36321552702` = **SUCCESS**
- no-live Full Slate `36321552693` = **SUCCESS**
- Archive Market Track Record `36321720427` = **SUCCESS**
- strict repo/readiness audits inside Full Slate = **SUCCESS**

Production lineage added since the prior handoff:
- PR #650 — Zach Ertz Week-3 live-prop quarantine — **MERGED**
- PR #651 — verified post-registry player identity aliases — **MERGED**
- PR #652 — Week-aware RB market-lineage/workbook semantics — **MERGED**

No football mean, model weight, projection rule, frozen Week-3 prospective label, or sportsbook-upstream feature was changed by #650/#651/#652.

## Canonical paid Week-3 betting-board run

The user explicitly authorized the OddsAPI spend.

Initial paid run:
- Full Slate `36292654942` = **FAILURE**
- failure occurred only after the complete football-first stack passed
- step: live prop identity gate
- exact unresolved core rows = **2**
- player = Zach Ertz
- markets = receptions + receiving yards

The source/roster fact was verified: Philadelphia's 2026-09-25 team update had Ertz on the practice squad and explicitly not on the active roster.

Mechanical repair:
- PR #650 merged
- production diff: one verified Zach Ertz entry in `data/manual_prop_quarantine.csv`
- treatment: remove Ertz sportsbook offers before model-facing pricing
- do **not** invent an active-roster promotion or projection

Canonical successful paid rerun:
- Full Slate `36293274478` = **SUCCESS**
- event = `workflow_dispatch`
- head = `0982b62276303403e2ca58b16e6f4fc3e041f65d`
- artifact = `10923570170`
- digest = `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- run created approximately 2026-09-27 00:05 ET; treat lines/odds as a preserved snapshot, not current prices

Successful paid-board counts:
- active games/events = **15**
- eligible football teams = **30**; ATL/GB Thursday game correctly withheld
- raw actual prop rows = **998**
- compact model-facing rows = **856**
- quarantined sportsbook rows = **152**
- pricing offer rows = **1,692**
- priced side rows = **3,384**
- priced players = **365**
- sportsbook inputs used to define football distributions = **false**

All critical paid path stages passed:
- live odds gate
- compact sportsbook boundary
- opponent map
- live identity semantics
- data-quality classifier execution
- bookmaker offer assembly
- deterministic metrics
- certified football-first pricing stack
- final-board quarantine
- strict repo/readiness audits
- master betting workbook
- artifact upload

Master workbook:
`outputs/NFL_BETTING_MODEL_MASTER.xlsx`
inside artifact `10923570170`.

## Player-identity certification blocker — CLOSED

The successful paid board initially surfaced one data-quality certification blocker:
- Joshua Palmer -> historical Josh Palmer / GSIS `00-0036988`
- Matt Hibner -> historical Matthew Hibner / GSIS `00-0040879`

The issue was identity-registry semantics: verified aliases could be inserted and then lost when the current identity registry collapsed to strict-prior latest rows.

PR #651 merged:
- current identity-only alias configuration is persistent
- no model-feature prior window changed
- no football mean/weight/rule changed

Authoritative exact-paid-snapshot offline verification:
- run `36294114610` = **SUCCESS**
- artifact `10923765458`
- digest `sha256:6e2baf1644e0631b13d81422a7ea94319db5f0022a6cd522e366897ebe224e09`

Result on the preserved paid snapshot:
- sportsbook current-roster mismatches = **0**
- temporary possible historical aliases = **0**
- certification blockers = **0**
- disposition = `FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS`

Declared limitations are explicit and non-blocking:
- optional TeamForm features unavailable: `neutral_pace_score`, `slot_rate`, `personnel_12_rate_z`
- direct WR-CB matchup rows unavailable/gated off; team-scheme coverage remains available
- explicitly new/unmapped players remain labeled as such

No second OddsAPI purchase was necessary for this identity closeout.

## Week-3 RB workbook/lineage semantic bug — CLOSED

Inspection of the successful paid workbook found a governance contradiction:
- actual Week-3 RB rush-yards priced rows had `rb_synthesis_applied=0`, correctly indicating Week-1 P3 was not active
- actual Week-3 RB rush+receiving rows had `rb_rush_rec_conservation_v2_applied=1`
- but `market_model_lineage_current.csv` and the workbook's Market Science presentation still hard-coded Week-1 P3 language

This did **not** change pricing. It was a downstream semantic/display bug that overstated Week-3 RB specialist authority.

PR #652 merged:
- merge commit `6a1de05842a03272e51269d35a7a01ca8dd1ab12`
- lineage is now explicitly week-aware
- Week 1 preserves the frozen P3 labels unchanged
- outside Week 1, standalone RB rush yards are labeled as generic calibrated rush-yards ensemble + joint MC
- outside Week 1, RB rush+receiving is labeled as `RB_RUSH_REC_CONSERVATION_V2`
- fail-closed checks verify actual priced-row flags/version

Authoritative exact-paid-snapshot offline verification:
- run `36321385122` = **SUCCESS**
- artifact `10932413506`
- digest `sha256:bda081cd4d8c97407bae3c29cf3a43fcf7d5fabc233ae13c6336f9eef3b9234a`

Verified preserved paid snapshot:
- `priced_week=3`
- `rb_p3_consumed_this_week=false`
- `rb_rush_rec_conservation_v2_consumed_this_week=true`
- RB rush-yards science status = `GENERIC_CANONICAL_ACTIVE`
- RB rush+rec science status = `PROMOTED_RB_RUSH_REC_CONSERVATION_V2_ACTIVE`

Final-head PR Repo CI `36321488848` = SUCCESS.

The legacy PR workflow `Replay Paid Full Slate Artifact Once` still fails at its first download step because historical Week-2 artifact `run_35282021679` is expired/not found. That is a known mechanical CI artifact-retention issue; it does not reach science or this Week-3 verification.

## Betting-board interpretation: do not confuse raw EV with confidence

The master workbook's `Best Snapshot Edges` / Snapshot Signal fields are raw model-vs-market signals, not a validated confidence tier or staking policy.

The real Week-3 board contains many very large RB rushing and rush+receiving under edges. Do **not** promote those merely because the raw gaps are large.

Reason:
- P3 is not active in Week 3
- standalone Week-3 RB rushing is generic canonical authority
- the archived 2026 Weeks 1-2 descriptive market record does not support treating large RB edges as elite:
  - QB pass yards: **36-16 = 69.23%**, +15.99 units
  - rush yards: **70-66 = 51.47%**
  - rush+receiving yards: **32-33 = 49.23%**
  - receiving yards: **132-144 = 47.83%**
  - receptions: **125-143 = 46.64%**

RB position-specific descriptive results:
- RB rush yards: **48-46 = 51.06%**
- RB rush+receiving yards: **32-33 = 49.23%**

Therefore the giant Week-3 RB unders are a signal to inspect, not a permission to manufacture a new selection rule.

## Current strongest live-season market evidence: QB passing yards

The 2026 Weeks 1-2 archived QB pass-yards record is:
- 52 settled bets
- 36 wins / 16 losses
- 69.23% descriptive hit rate
- +15.99 units

Descriptive edge buckets:
- edge >=5%: 33-14 / 70.21% / +15.31 units
- edge >=7.5%: 29-12 / 70.73% / +13.77 units
- edge >=10%: 23-9 / 71.88% / +11.44 units
- edge >=15%: 13-6 / 68.42% / +5.55 units
- edge >=20%: 9-3 / 75.00% / +5.00 units

This is **descriptive evidence only**. Do not retroactively convert one of these thresholds into a production selection rule without a frozen prospective plan.

Top preserved Week-3 QB passing-yard snapshot signals from paid run `36293274478`:
1. Justin Herbert LAC vs BUF — UNDER 228.5 DK -112 — model 192.19 — raw edge 28.28 pp
2. Baker Mayfield TB vs MIN — UNDER 216.5 FD -113 — model 182.43 — raw edge 27.81 pp
3. Cam Ward TEN vs NYG — OVER 176.5 DK -112 — model 213.51 — raw edge 24.11 pp
4. Jared Goff DET vs NYJ — UNDER 261.5 DK -112 — model 235.33 — raw edge 19.27 pp
5. Bo Nix DEN vs LAR — OVER 211.5 FD -113 — model 242.34 — raw edge 18.21 pp
6. Deshaun Watson CLE vs CAR — OVER 187.5 FD -113 — model 215.34 — raw edge 17.81 pp
7. Jameis Winston NYG vs TEN — OVER 200.5 DK -112 — model 227.40 — raw edge 16.72 pp
8. Kyler Murray MIN vs TB — UNDER 213.5 DK -111 — model 195.88 — raw edge 15.56 pp
9. Malik Willis MIA vs KC — OVER 173.5 FD -113 — model 196.59 — raw edge 15.10 pp
10. C.J. Stroud HOU vs IND — OVER 236.5 DK -112 — model 262.39 — raw edge 14.78 pp

All are **snapshot lines** from the preserved paid run, not current odds. Re-fetching current odds requires a new explicitly authorized OddsAPI spend.

One board caveat:
- Marcus Mariota WAS vs SEA also had a large raw QB pass edge, but the current PlayerForm role showed QB2 in the preserved snapshot. Do not elevate it without resolving starter authority/current state.

## Anytime TD remains non-certified science

ATD remains:
`ATD_EXECUTION_PROVEN_SCIENCE_NOT_YET_CERTIFIED`

Current method:
- generic joint MC offensive TD rate + red-zone modifiers
- no dedicated walk-forward probability certification

Do not present ATD raw edges as equivalent to the promoted/certified market lanes.

## Frozen prospective Week-3 lanes remain untouched

### RB Vacancy Opportunity V1

DEN:
- Jonah Coleman unavailable
- `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- lead `NONE_CLEAR`

PIT:
- Rico Dowdle unavailable
- `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
- lead `JAYLEN_WARREN`
- confidence MEDIUM

After games:
1. grade RB Vacancy V1 independently
2. grade frozen public-intent labels against actual carry/snap concentration
3. never rewrite pregame labels

PR #642 remains the frozen prospective authority.

### Receiving Rule Semantics V1

Frozen Week-3 cells:
- A0B0 current production
- A1B0 middle-open unit repair
- A0B1 slot-alignment repair
- A1B1 combined repair

Confirmed defects remain:
- `middle_open_rate` percentage-point scaled while production interprets 0-1
- SWR alignment is calculated upstream then dropped before production SLOT labeling

Historical Stage 2 remains:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`

After games attach outcomes to unchanged cells and grade. No redesign/rescue.

### Availability -> Opportunity

Disposition remains:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Do not resurrect 60/30/10 and do not fit Week-3 vacancy coefficients from outcomes.

## Other production locks remain unchanged

Rush-Att Zero-MC Allocation Lineage V1:
`CLOSED_AS_HARD_SUPPORT_CONFLICT_NO_REPAIR_AUTHORIZED`

Discrete Count Mean Alignment V1:
- production-active for receptions + rush_att
- zero/nonfinite-MC exact no-op
- non-count legacy-invariant

RB Rush+Receiving Conservation V2:
- production-active and locked
- historical 2024-25 RB MAE `27.6853 -> 25.5718`
- Week-3 workbook lineage now correctly identifies V2 on the combined market

Closed/do-not-repeat families remain closed:
- Receiver Room Targets-per-Play V1
- WR Anchor / Role-Transmission
- TE Receiving-Yards Width V2
- Rush Pool Evidence Guard V1
- Rush Post-Ensemble Reconciliation V1
- residual/top-five normalization rescue
- generic copula rescue
- old receiving-market missing-weight experiment
- retrospective M96 RB rushing family
- rush-att zero-MC repair variants after lineage closure

## Exact next action

No additional paid OddsAPI fetch is authorized by this handoff.

Valid immediate work:
1. use the preserved paid artifact `10923570170` for board analysis and diagnostics;
2. treat QB pass yards as the strongest current **descriptive** live-season evidence lane, without inventing a post-hoc threshold;
3. treat giant RB Week-3 raw edges cautiously; P3 is not active and Weeks 1-2 RB market results are approximately coin-flip;
4. if a concrete RB implementation/weighting/data-lineage contradiction is found, freeze it before scoring and audit it mechanically; do not rescue based on the Week-3 sportsbook gaps alone;
5. preserve all frozen Week-3 prospective lanes until outcomes are final;
6. after games, grade RB Vacancy/Public Intent and Receiving Rule Semantics against frozen pregame authority.

If the user explicitly authorizes a fresh OddsAPI spend later, a new controlled `fetch_live_odds=true` Full Slate can refresh current lines. Otherwise do not refetch merely because the prior snapshot has aged.

## Continuity anchors

Issue #535:
- prior systems continuity: `5852304320`
- paid launch checkpoint: `5852427642`
- paid success / Ertz closeout: `5852547875`
- identity certification closeout: `5856067868`
- paid-board + RB lineage final closeout: `5856188328`

Memory-efficient next-chat read order:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this handoff
4. Issue #535 from `5852547875` onward
5. live main / PR #642 / relevant Actions state

Then work. Do not recursively load older handoffs.
