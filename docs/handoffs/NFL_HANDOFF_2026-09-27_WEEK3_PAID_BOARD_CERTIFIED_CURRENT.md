# NFL HANDOFF — 2026-09-27 — WEEK-3 PAID BOARD CERTIFIED CURRENT

GitHub is canonical. This handoff supersedes the 2026-09-26 systems-audit/count-promotion handoff for immediate execution. Do not recursively load older handoffs unless this file explicitly directs it.

## Canonical current main

Current main after all paid-board mechanical repairs:
`6a1de05842a03272e51269d35a7a01ca8dd1ab12`

Production science lineage remains the previously promoted stack. The commits since the prior checkpoint are operational / identity / governance repairs only; they did not change football means, ensemble weights, model rules, frozen Week-3 prospective labels, or sportsbook-upstream football inputs.

Post-merge current-main verification:
- Repo CI `36321552702` = **SUCCESS**
- no-live Full Slate `36321552693` = **SUCCESS**
- these runs are on exact current main `6a1de058...`
- Archive Market Track Record triggered after the no-live run; check its live state if needed

## Canonical paid Week-3 betting board

User explicitly authorized one OddsAPI-paid Full Slate in the active chat.

Initial paid attempt:
- Full Slate `36292654942` = **FAILURE**
- sportsbook acquisition itself completed, but the live identity gate failed closed on exactly 2 core rows, both Zach Ertz:
  - `player_reception_yds`
  - `player_receptions`
- reason: sportsbook had CHI-PHI Ertz offers while the football-only current Ourlads / eligible-role / PlayerForm universe had no certifiable Ertz production row.

Verified authority:
- Philadelphia Eagles 2026-09-25 update states Ertz was signed to the practice squad and explicitly said he was not on the active roster.

Mechanical repair:
- PR #650 = **MERGED**
- merge commit `bd20a8ab3d8484a7e0c26b1a60c19bd951fe6201`
- one production data change only: verified Zach Ertz row added to `data/manual_prop_quarantine.csv`
- no active-roster override was invented
- no football science changed

Canonical successful paid rerun:
- Full Slate `36293274478` = **SUCCESS**
- event = `workflow_dispatch`
- head = `0982b62276303403e2ca58b16e6f4fc3e041f65d`
- artifact = `10923570170`
- digest = `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`

Critical paid-path steps all passed:
- football eligibility
- live odds gate
- compact sportsbook boundary
- opponent map
- live identity semantics
- data-quality classifier execution
- exact bookmaker offer assembly
- deterministic metrics
- certified football-first pricing
- final-board quarantine
- strict repo/readiness audits
- master betting workbook
- artifact upload

Paid artifact counts:
- active games/events = **15**
- current football teams = **30**; ATL/GB Thursday game correctly withheld
- raw actual prop rows = **998**
- compact model-facing rows = **856**
- quarantined sportsbook rows = **152**
- book-line pricing rows = **1,692**
- priced side rows = **3,384**
- priced players = **365**
- sportsbook inputs used to define football distributions = **false**

Ertz final identity state:
- `LIVE_PROP_IDENTITY_READY`
- unresolved core rows = **0**
- manual prop quarantine removed compact 5 / raw 16 / enriched 16 rows

No second OddsAPI purchase is needed merely to inspect, certify, or analyze this preserved paid board.

## Paid-board identity certification — CLOSED

The successful paid artifact surfaced one certification blocker in historical player identity:
- Joshua Palmer -> historical Josh Palmer / GSIS `00-0036988`
- Matt Hibner -> historical Matthew Hibner / GSIS `00-0040879`

This was an identity-registry lineage defect only.

Repair:
- PR #651 = **MERGED**
- merge/main commit `e3fac9fb56ba0c72d34311c44c5f8c6fae459a2b`
- verified aliases are now re-applied after the Player Identity v3 registry collapse
- persistent current alias configuration added
- no targets/carries/yards, football means, weights, rules, or sportsbook-upstream inputs changed

Exact successful paid-snapshot offline verification:
- run `36294114610` = **SUCCESS**
- artifact `10923765458`
- digest = `sha256:6e2baf1644e0631b13d81422a7ea94319db5f0022a6cd522e366897ebe224e09`

After repair on the exact paid snapshot:
- sportsbook current-roster mismatches = **0**
- temporary possible historical aliases = **0**
- certification blockers = **0**
- disposition = `FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS`

Declared non-blocking limitations:
- optional TeamForm features unavailable: `neutral_pace_score`, `slot_rate`, `personnel_12_rate_z`
- direct WR-CB matchup rows unavailable; direct matchup consumption is gated off while team-scheme coverage remains available
- explicitly new/unmapped identities remain labeled, but there is no unresolved alias certification blocker

## Week-3 RB workbook lineage semantics — FIXED

While inspecting the real paid workbook, a governance/presentation contradiction was found:

Actual Week-3 priced behavior:
- RB/FB `player_rush_yds`: `rb_synthesis_applied=0` for all rows; Week-1 P3 is correctly off
- RB/FB `player_rush_reception_yds`: `RB_RUSH_REC_CONSERVATION_V2` is applied

But the market-lineage artifact / workbook hard-coded Week-1 P3 labels and therefore falsely described Week-3 RB rows as P3-specialist active.

This was **lineage metadata only**, not a projection defect.

Repair:
- PR #652 = **MERGED**
- current main after merge = `6a1de05842a03272e51269d35a7a01ca8dd1ab12`
- lineage generator is now week-aware
- Week 1 preserves frozen P3 semantics
- outside Week 1:
  - standalone RB rush yards = `GENERIC_CANONICAL_ACTIVE`
  - RB rush+rec = `PROMOTED_RB_RUSH_REC_CONSERVATION_V2_ACTIVE`
- added fail-closed checks and regression tests
- no projections, arrays, sportsbook prices, means, weights, rules, or OddsAPI calls changed

Exact preserved-paid-snapshot verification:
- run `36321385122` = **SUCCESS**
- artifact `10932413506`
- digest = `sha256:bda081cd4d8c97407bae3c29cf3a43fcf7d5fabc233ae13c6336f9eef3b9234a`
- restored paid run `36293274478` with zero OddsAPI refetch
- focused regression tests = PASS
- canonical market lineage regeneration = PASS
- corrected master betting workbook = PASS

Correct Week-3 lineage authority now states:
- `priced_week=3`
- `rb_p3_week1_only=true`
- `rb_p3_consumed_this_week=false`
- `rb_rush_rec_conservation_v2_consumed_this_week=true`
- version `RB_RUSH_REC_CONSERVATION_V2`

## Decision-grade interpretation of the paid board

Important: workbook raw EV / Snapshot Signal is a **snapshot signal**, not a calibrated confidence or staking tier.

Anytime TD:
- current execution is mechanical/football-only
- scientific status remains `ATD_GENERIC_ACTIVE_NOT_DEDICATED_SCIENCE_CERTIFIED`
- do not treat ATD as a decision-grade certified lane

RB rushing / rush+rec:
- many of the largest raw board edges are RB unders
- standalone Week-3 RB rushing is generic calibrated, not P3
- rush+rec is production-active V2 conservation
- the workbook has **no retrospective trust score authority** for these markets
- do not equate their huge raw EV with the mature trust level of QB passing

QB passing:
- science status = `PROMOTED_QB_MEAN_PLUS_DISTRIBUTION_SPECIALISTS_ACTIVE`
- current retrospective workbook evidence class = descriptive / healthier directional diagnostic
- this is the cleanest mature lane on the current board

TE receptions / receiving yards:
- promoted TE entitlement specialist is active
- workbook evidence is descriptive / healthier directional diagnostic
- notable secondary lane

WR receiving yards:
- promoted M38 + R15 entitlement is active, but workbook retrospective directional diagnostic remains historically weaker than QB passing / TE receiving

## Leading board snapshot — descriptive, not a staking recommendation

Top QB passing snapshot rows from the successful paid board:
- Justin Herbert LAC UNDER 228.5 DK -112 — model 192.19 — raw EV ROI ~0.482
- Baker Mayfield TB UNDER 216.5 FD -113 — model 182.43 — raw EV ROI ~0.467
- Cam Ward TEN OVER 176.5 DK -112 — model 213.51 — raw EV ROI ~0.403
- Jared Goff DET UNDER 261.5 DK -112 — model 235.33 — raw EV ROI ~0.311
- Bo Nix DEN OVER 211.5 DK -113 — model 242.34 — raw EV ROI ~0.286
- Deshaun Watson CLE OVER 187.5 FD -113 — model 215.34 — raw EV ROI ~0.278
- Jameis Winston NYG OVER 200.5 -112 — raw EV ROI ~0.263
- Kyler Murray MIN UNDER 213.5 -111 — raw EV ROI ~0.242
- Malik Willis MIA OVER 173.5 -113 — raw EV ROI ~0.227
- Daniel Jones IND OVER 212.5 -111 — raw EV ROI ~0.227

Notable TE snapshot rows:
- Terrance Ferguson LAR receptions UNDER 3.5 FD +128 — raw EV ROI ~0.836
- Erick All receiving yards OVER 5.5 FD -102 — raw EV ROI ~0.730
- Terrance Ferguson receiving yards UNDER 49.5 DK -111 — raw EV ROI ~0.709
- Trey McBride receiving yards UNDER 68.5 -113 — raw EV ROI ~0.596

Do not mechanically rank these across markets solely by raw EV because market science maturity differs.

## Frozen prospective Week-3 science remains unchanged

RB Vacancy Opportunity V1:
DEN:
- Jonah Coleman unavailable
- label `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- lead `NONE_CLEAR`

PIT:
- Rico Dowdle unavailable
- label `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
- lead `JAYLEN_WARREN`
- confidence MEDIUM

PR #642 remains the frozen prospective cohort. Do not rewrite it pregame.

Receiving Rule Semantics V1 remains frozen:
- A0B0 current production
- A1B0 middle-open unit repair
- A0B1 slot-alignment repair
- A1B1 combined repair
- no postgame redesign/rescue

Availability -> Opportunity remains:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Rush-Att Zero-MC Allocation Lineage V1 remains:
`CLOSED_AS_HARD_SUPPORT_CONFLICT_NO_REPAIR_AUTHORIZED`

Discrete Count Mean Alignment V1 remains production-active.

RB Rush+Receiving Conservation V2 remains production-active and locked.

## Closed / do-not-repeat

Do not reopen without genuinely new evidence:
- 30-team/32-team Full Slate theory
- Receiver Room Targets-per-Play V1
- WR Anchor / Role-Transmission
- TE Receiving-Yards Width V2
- Rush Pool Evidence Guard V1
- Rush Post-Ensemble Reconciliation V1
- residual/top-five normalization rescue
- generic copula rescue
- old receiving-market missing-weight experiment
- retrospective M96 RB rushing family
- rush-att zero-MC repair variants
- QB/RB carveouts, alternate top-N, depth/role exceptions, share-threshold rescues for zero-MC

## Exact next actions

Before games:
1. use the preserved successful paid artifact / corrected workbook for Week-3 board analysis; do not spend OddsAPI credits again merely to inspect the same snapshot;
2. preserve the frozen RB Vacancy/Public Intent, Receiving Rule Semantics, and Availability -> Opportunity Week-3 authorities unchanged;
3. do not treat raw EV as calibrated confidence/staking authority;
4. keep ATD outside the decision-grade certified board.

After games:
1. grade RB Vacancy V1 independently;
2. grade the frozen public-intent labels against actual carry/snap concentration;
3. attach outcomes to the unchanged Receiving Rule Semantics A/B cells and grade;
4. do not fit rescue thresholds or vacancy coefficients after seeing Week-3 outcomes.

Sportsbook data remains downstream of football-model science.

## Continuity anchors

Issue #535:
- pre-paid systems continuity: comment `5852304320`
- paid launch: `5852427642`
- paid rerun success + initial identity caveat: `5852547875`
- identity certification blocker closed: `5856067868`
- add the final PR #652 / current-main closure comment from this session as the newest authority

Memory-efficient next-chat read order:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this file
4. Issue #535 from comment `5852547875` onward
5. live main / PRs / Actions

Do not recursively load older handoffs unless this file explicitly directs it.
