# NFL HANDOFF — 2026-09-29 — WEEK-3 POSTMORTEM COMPLETE / PROSPECTIVE RESULTS CURRENT

GitHub is canonical. This handoff is the authoritative continuity document for the next chat.

Do **not** recursively read older handoffs unless a named artifact/result below is missing. Start here, query GitHub live, then continue execution.

---

## 1. EXECUTIVE STATE

Repository:
`dkaps6/imtiredofthis`

Canonical main at handoff:
`8a965f2754b4ccfa960a2a05517fd40861f49a3d`

Active Week-3 postmortem/research branch:
`research-week3-postmortem-execution-v1`

Current branch head:
`fb80bb6c4af6c280aee53d3013315632c69a7c9b`

Current status:
- Week 3 is fully final in nflverse.
- The canonical paid Week-3 board is settled with **zero unresolved rows**.
- Weeks 1-3 cumulative production postmortem is complete.
- Projection-authority attribution is complete.
- Frozen Week-3 RB Vacancy V1 is graded.
- Frozen Week-3 public-intent labels are graded.
- Frozen Week-3 Receiving Rule Semantics A/B cells are graded.
- RB-PD2 Week-3 prospective Observation #1 is attached and scored.
- No new production coefficient, carveout, side rule, probability rescale, or semantic repair was promoted from the Week-3 outcomes.
- Production code remains frozen with respect to these postmortem findings.
- No paid OddsAPI acquisition was used for any of the postgame work below.
- No relevant Week-3 postmortem workflow is currently required to finish; the latest runs are complete.

---

## 2. READ ORDER FOR THE NEXT CHAT

Read only:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-09-29_WEEK3_POSTMORTEM_COMPLETE_CURRENT.md`
4. Issue #535 from comment `5897966236` onward, especially:
   - `5897966236` — Weeks 1-3 postmortem
   - `5898292164` — RB Vacancy Week-3 result
   - `5898483120` — public-intent Week-3 result
   - `5898648022` — Receiving Rule Semantics Week-3 result
   - `5899001957` — RB-PD2 Week-3 Observation #1
   - `5881240970` — GSIS execution-ready checkpoint / private filenames
5. Query GitHub live for:
   - current `main`
   - `research-week3-postmortem-execution-v1`
   - `repair-specialist-rng-isolation-v1`
   - PR #662
   - latest Actions

Then work.

Do not ask the user to re-explain anything.

---

## 3. CANONICAL WEEK-3 BOARD / SETTLEMENT

Canonical paid Week-3 Full Slate:
- run `36293274478` = SUCCESS
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- source SHA `0982b62276303403e2ca58b16e6f4fc3e041f65d`
- priced quote-side rows: 3,384

Frozen pregame publication gate recovered from original paid-run logs:
- 16 games total
- 15 eligible
- ATL-GB already kicked off, so ATL and GB were correctly withheld
- live identity gate had 0 core unresolved player rows
- final priced-board quarantine removed 0 additional rows

Week-3 postgame source gate:
- all games final
- player weekly stats complete
- weekly rosters complete
- snap counts complete

One settlement edge case was resolved fail-closed:
- Tyson Bagent, CHI, rush_yards 17.5 UNDER, DraftKings
- roster status ACT but no game participation
- exact sportsbook participation rule + postgame nonparticipation evidence -> VOID
- evidence file:
  `data/research/week3_postmortem/WEEK3_MANUAL_SETTLEMENT_EVIDENCE_V1.json`

Canonical Week-3 settlement/result run:
`36620072764` = SUCCESS

Do not reacquire Week-3 odds merely to re-grade this board.

---

## 4. WEEKS 1-3 PRODUCTION POSTMORTEM — COMPLETE

Result:
`docs/research/WEEKS1_3_POSTMORTEM_FINDINGS_V1.md`

Authority:
- run `36622608145` = SUCCESS
- artifact `11059171429`
- digest `sha256:5e05afca1f94afaa04a4da6f93fa396714df2234d283cc75302b87eac97d38a6`

Canonical cumulative record:
- selected settlement rows: 1,260
- decided bets: 1,240
- voids: 20
- record: **629-611**
- win rate: **50.7%**
- units: **-40.72u**
- ROI: **-3.28%**
- model MAE: **17.55**
- selected market-line MAE: **16.36**
- model closer than line: **45.8%**
- model signed bias: **-5.36**
- line signed bias: **-2.20**

By week:
- W1: 204-205, -20.92u, 49.9%
- W2: 192-198, -22.06u, 49.2%
- W3: **233-208, +2.26u, 52.8%**

Cumulative by position:
- QB: 80-62, +10.63u, 56.3%
- RB: 221-214, -15.00u, model bias -8.17
- WR: 230-224, -16.11u
- TE: 98-109, -18.24u

Cumulative by market:
- pass_yards: 47-30, +11.82u, 61.0%
- rush_yards: 111-101, -0.11u
- rec_yards: 216-219, -26.60u
- receptions: 205-210, -19.08u
- rush_rec_yards: 50-51, -6.75u, model bias -17.63
- RB rush_rec bias specifically: -17.99

Confidence/calibration problem:
- mean stated fair probability ~68.7%
- actual win rate 50.7%
- 70%-100% stated band:
  - 502 bets
  - mean stated probability 81.0%
  - realized 51.8%
  - calibration gap -29.2pp

Clustered inference:
- 54 eligible slices tested
- game-cluster-aware BH-FDR q=0.10
- **0 slices survived**
- disposition:
  `NO_SLICE_SURVIVES_MULTIPLE_COMPARISONS_CORRECTION`

Nominal pass-yards strength does **not** authorize a QB/pass-yards production carveout.

Do not:
- fit a Week-3-only rescue;
- add a side rule;
- add a QB/pass-yards carveout;
- globally rescale probability from three weeks;
- reopen closed historical calibration lanes under new names.

---

## 5. PROJECTION AUTHORITY ATTRIBUTION — COMPLETE

Plan:
`docs/research/WEEKS1_3_PROJECTION_AUTHORITY_ATTRIBUTION_V1_PLAN.md`

Result:
`docs/research/WEEKS1_3_PROJECTION_AUTHORITY_ATTRIBUTION_V1_RESULT.md`

Authority:
- run `36626121661` = SUCCESS
- artifact `11060101978`
- digest `sha256:8833b5611cb979d585932d94ea071b26d9dfbad3784c196381ff376681d23e3`

Disposition:
`MIXED_BY_MARKET_OR_POSITION`

Overall MC -> final on 1,240 settled bets:
- MC MAE 17.9683
- final MAE 17.5501
- paired improvement +0.4182
- MC bias -8.0912
- final bias -5.3625
- movement toward actual on moved rows 56.19%
- game-cluster bootstrap 95% CI for paired improvement:
  [-0.0631, +0.8862]

By market:
- pass_yards: 56.8491 -> 53.2781 MAE; +3.5710 yd improvement
- rush_yards: 21.4290 -> 20.2504; +1.1786 yd
- rec_yards: 21.7300 -> 21.7600; slightly worse
- receptions: 1.6903 -> 1.6739; tiny improvement
- rush_rec_yards: 31.7461 -> 31.7461 in archived mc_proj -> model_proj comparison

Important rush_rec interpretation:
RB Rush+Receiving Conservation V2 replaces the combo draw array before `mc_proj` is recorded for Week 3. Therefore `mc_proj == model_proj` does **not** imply V2 did nothing.

Bottom line:
- upstream MC is already low-biased;
- late-stage stack meaningfully helps QB passing and rushing;
- receiving means are mostly untouched;
- global rollback of synthesis/state/ensemble is unsupported.

---

## 6. RB VACANCY OPPORTUNITY V1 — WEEK 3 GRADED

Result:
`docs/research/RB_VACANCY_OPPORTUNITY_V1_WEEK3_RESULT.md`

Frozen pregame authority:
- run `36205758768`
- artifact `10893588588`
- digest `sha256:54437c69f0a66c2c9bb97c520f3e8d9c933e0203dac39fe1fde5352dafa3eeea`

Postgame authority:
- run `36626894564` = SUCCESS
- artifact `11060248043`
- digest `sha256:b53ef1935253737a967198caf2d62eebd66d4c7d922fd73501198fa538d22c1b`

Disposition:
`WEEK3_OBSERVATIONAL_MIXED`

Full locked cohort:
Rush attempts:
- MAE 4.0279 -> 3.8549 improved
- absolute bias 0.2607 -> 0.4110 worsened

Rush yards:
- MAE 28.1177 -> 26.6182 improved
- absolute bias 3.1227 -> 1.9380 improved

Direct recipients improved in both MAE and absolute bias.

Team split:
- DEN worsened in both markets
- PIT improved in both markets
- no player reached frozen 20+ / 25+ actual-carry slice

Do not rescue with a PIT-only rule or change the transfer formula.
Continue future qualifying vacancy events under unchanged V1.

---

## 7. PUBLIC INTENT — WEEK 3 GRADED

Result:
`docs/research/PUBLIC_INTENT_WEEK3_PROSPECTIVE_CAPTURE_V1_RESULT.md`

Frozen pregame commit:
`0170f89a42ca5c10e57c06685cd2736474dd45a2`

Grade authority:
- run `36627524836` = SUCCESS
- artifact `11060069037`
- digest `sha256:5d695cfa82b5a7dac5798277ec542bd80cc2a76ed952287bca1eb7a344c47c41`

Disposition:
`WEEK3_PUBLIC_INTENT_DIRECTIONALLY_INFORMATIVE`

Frozen labels:
- DEN: `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- PIT: `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`

Actual:
DEN:
- 20 RB/FB carries
- 4 participants
- Dobbins 17 carries = 85.0%
- carry HHI 0.7350

PIT:
- 19 RB/FB carries
- 4 participants
- Warren 17 carries = 89.47%
- carry HHI 0.8061
- frozen lead identity was correct

Nuance:
DEN itself was highly concentrated. Do not call the DEN no-clear-concentration label a clean literal hit.

Only defensible statement:
PIT was slightly more concentrated than DEN by both HHI and top share, and Warren correctly led PIT.

No public-intent coefficient is authorized.

---

## 8. RECEIVING RULE SEMANTICS V1 — WEEK 3 GRADED

Result:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_WEEK3_RESULT.md`

Frozen Stage-1:
- run `36276736046`
- artifact `10917254062`
- digest `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

Postgame:
- run `36629181044` = SUCCESS
- artifact `11061636348`
- digest `sha256:bbe8a30cdeb68b2b1b4df952aa6dc8632c74c1588df7f86a1b2c0b1476b1ceeb`

Disposition:
`WEEK3_PROSPECTIVE_EVIDENCE_ONLY_CONTINUE_UNCHANGED_CELLS`

A1B0 — middle_open fix:
- TE MAE improved for target share, receptions, rec yards
- TE p90 worsened for receptions and rec yards
- pooled WR/TE rec-yards MAE worsened slightly
- frozen gates do not clear

A0B1 — slot carry:
- targeted SWR MAE worsened on all three scored metrics

A1B1:
- changed-row receptions and rec-yards MAE worsened
- all targeted p90 deltas worsened

No candidate is promoted.
No rescue variant.
No threshold retuning.
Do not partially promote A1B0 from one week.

Postgame identity evidence used only for grading:
- Matt Hibner -> Matthew Hibner via existing verified alias registry
- Drew Ogletree -> Andrew Ogletree via research-only official-source evidence
- Hollywood Brown -> Marquise Brown via research-only official-source evidence

These did not alter pregame model identity/features.

---

## 9. RB-PD2 YARD-DIFFICULTY WIDTH — WEEK 3 OBSERVATION #1 COMPLETE

Result:
`docs/research/RB_PD2_FORWARD_WEEK3_OBSERVATION_V1_RESULT.md`

Frozen prospective authority:
- original paid shadow run `36330757181`
- recovered immutable lock run `36331514633`
- artifact `10935529140`
- digest `sha256:94f168125dc889b6a747da5f3b3e3829d2fd4bf7a9b6b3d9f4d2d79ac0b59b6e`

First postgame authority:
- run `36629683048` = SUCCESS
- artifact `11061697097`
- digest `sha256:1472ccd4290be2c3841265af68c68c5b5523d18eaef8ef11be8349233124ec1b`

Latest verification rerun on branch head:
- run `36631186678` = SUCCESS
- artifact `11062119094`
- digest `sha256:295195e150fb676389bc6abde92028c264ce4fe2cda6e8ec780de02462b36297`

Disposition:
`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`

This is HOLD / underpowered.
It is **not** PASS and **not** scientific FAIL.

Support:
- 1 / 8 required prospectively locked weeks
- 46 / 400 required unique eligible player-games
- 15 games

Week-3 observation:
Baseline:
- CRPS 16.513152
- 80% coverage 52.17%
- 90% coverage 69.57%
- Brier >=50 0.184883
- Brier >=75 0.168414
- Brier >=100 0.041522
- point MAE 22.213108

Candidate:
- CRPS 16.305948
- 80% coverage 58.70%
- 90% coverage 73.91%
- Brier >=50 0.185754
- Brier >=75 0.160870
- Brier >=100 0.040266
- point MAE 22.213108

Observed CRPS gain baseline-candidate:
**+0.207204**

Game-cluster bootstrap:
- 10,000 reps
- 95% CI [+0.074031, +0.345209]

Crossed player×game:
- P(candidate CRPS - baseline CRPS < 0) = 0.9374
- eventual frozen gate is >=0.95

High-Q75 slice:
- CRPS 22.672719 -> 21.979044
- 80% coverage gap 30.0pp -> 5.0pp
- 90% coverage gap 15.0pp -> 1.67pp

Mean neutrality:
- max rowwise mean gap 1.42e-14
- pooled point-MAE difference 0.0

Week-3-only unsatisfied items:
- Brier >=50 non-worse failed slightly
- crossed robustness 0.9374 < eventual 0.95 gate

Do **not** call those scientific failures because the frozen support floor is nowhere near met.

Continue exact frozen PD2 prospectively.
No width/onset/window/reference/threshold change.
No subgroup rescue.

---

## 10. AVAILABILITY -> OPPORTUNITY RULE-ORDER GAP — CONFIRMED, NOT REPAIRED

Historical/current audit authority from prior frozen work:
- run `36275905038`
- artifact `10917451964`
- digest `sha256:340b5992a58ff591baf7a4ccf92c36482c1e22a082fabdc14e2f4e7dbcb1f6fb`
- disposition:
  `AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Week-3 audit facts:
- 11 definitive-unavailable skill players
  - 6 WR
  - 3 TE
  - 2 RB/FB
- surviving eligible roles = 0
- surviving PlayerForm = 0
- surviving ModelContext = 0
- production-reachable legacy `rules_injury_redistribution` rows = 0

Meaning:
definitive unavailable players are correctly removed before the legacy vacancy rule can see them.

Affected teams still normalize modeled team opportunity mass back to 0.95 + residual 0.05.
The unresolved question is successor identity/concentration, not missing team mass.

No repair was promoted.
Do not resurrect 60/30/10.
Do not fit Week-3 transfer coefficients.

### Immediate remaining Week-3 task
This is the one frozen Week-3 lane not yet given a clean postgame closure in the current postmortem branch.

Next chat should:
1. inspect the original frozen Availability->Opportunity audit/result/contract at its authority branch/commit;
2. determine whether it actually predeclared a valid postgame descriptive grade;
3. if yes, attach Week-3 outcomes **descriptively only** with no fitting;
4. if no predeclared postgame contract exists, explicitly document `NO_VALID_POSTGAME_GRADING_CONTRACT` rather than inventing thresholds after seeing outcomes.

Do not redesign the rule.

---

## 11. SPECIALIST RNG REPAIR — SECONDARY OPEN MECHANICAL LANE

Research already established:
- specialist finite-MC global RNG path coupling exists;
- core RNG isolation passed;
- QB C2 downstream isolation passed;
- paid-board counterfactual stayed within ordinary MC resampling envelope.

Production-repair branch:
`repair-specialist-rng-isolation-v1`

Current branch head:
`41dba6e410509fa48b7031326db4e30c946ae03c`

Latest run:
`36504441918` — FAILURE

This newest run has **not** been diagnosed in the Week-3 postmortem work.

Older failure `36435652561` had:
- tests green
- live workflow untouched green
- paid artifact authority green
- source/state parity green
- specialist arrays intentionally changed
- but exact historical board-equivalence constants did not reproduce exactly

The Week-3 postmortem took priority before the latest run was diagnosed.

If resuming RNG:
1. fetch jobs/logs for `36504441918`;
2. identify the exact current failure;
3. make the smallest mechanical repair only if justified;
4. do not merge merely because core isolation science passed;
5. do not alter production model science.

---

## 12. GSIS — CURRENT PRIVATE DATA / PROSPECTIVE BOUNDARY

Draft PR:
#662

Branch:
`research-gsis-point-in-time-archive-v1`

The public branch contains code/provenance only.
Do **not** commit raw GSIS report cells to the public repo.

Private ChatGPT Library filenames:
- `NFLGSIS_2026_REG_Week03_Point_in_Time_2026-09-28.json.gz`
  - SHA-256 `779b1bd6c5d4dbd390c992d1e02e007edb80dac7d5e19b8b9045b161000c8a23`
- `NFLGSIS_2026_REG_Week03_Point_in_Time_2026-09-28_manifest.json`
- `NFLGSIS_2025_REG_Lineup_Detail_Pilot_2026-09-28.json.gz`
  - SHA-256 `166589a5ee5a8e5388818068a18bab0632ca0839b09cc3a35e073701d1d2f5eb`
- `NFLGSIS_2025_REG_Lineup_Detail_Pilot_2026-09-28_manifest.json`

Current audit result from Issue #535 comment `5881240970`:

Lineup Detail:
- `INCREMENTAL_CURRENT_STATE_CONFIRMED`
- exact 11-player co-occurrence is genuinely incremental over marginal snaps/depth
- temporal disposition:
  `NEEDS_PROSPECTIVE_HISTORY`

Formation Usage:
- `INCREMENTAL_CURRENT_STATE_CONFIRMED`
- current #TE/#WR personnel by down/distance and personnel-conditioned play choice are incrementally descriptive

Lineup Combinations:
`MOSTLY_REDUNDANT_ONE_NARROW_FIELD`

Only prospectively retain:
`Unique Starting Lineups`

Derive instead:
- Unique Lineups
- Pct of Plays Featuring Most Common Lineup

Important boundary:
- 2026 snapshot is cumulative through Week 3.
- It is **not** a valid pregame reconstruction for Weeks 1-3.
- It is baseline Snapshot #1 for prospective temporal work.
- Next useful temporal evidence requires a later immutable weekly snapshot.
- Do not reopen Player Game-by-Game, Team By Game, Down Analysis, Play Propensity, or broad historical GSIS acquisition.

---

## 13. PRODUCTION-ACTIVE / PROTECTED SCIENCE

Do not casually reopen:

### RB Rush+Receiving Conservation V2
Production-active.
Historical 2024-25:
- MAE 27.6853 -> 25.5718
- RMSE 39.8437 -> 36.4030
- p90 AE 64.7742 -> 57.2604
- candidate closer 59.13%

Week-2 observational confirmation:
- MAE 30.3620 -> 28.9014
- signed bias +20.1339 -> +9.8670

Week-3 postmortem still shows rush_rec means low overall, but that is **not** evidence to undo V2; V2 is already an improvement relative to its old baseline.

### Discrete Count Mean Alignment V1
Production-active after earlier integration/promotion.
Protect count support for receptions/rush_att.
Do not reinterpret the Week-3 postmortem as an instruction to remove it.

---

## 14. CLOSED / DO-NOT-REPEAT

Do not rerun or rescue these without genuinely new information:

- Receiver Room Targets-per-Play V1 — failed 2024-25 confirmation
- WR Anchor / Role-Transmission V1 — closed
- TE Receiving-Yards Width V2 — failed frozen gates
- Rush Pool Evidence Guard V1 integration — failed full-stack science
- Rush Post-Ensemble Reconciliation V1 — failed
- generic residual/top-five normalization rescue
- generic copula/dependence idea as a fix for marginal mean error
- historical missing receiving-market weights as an explanation
- M96 retrospective RB rushing search
- generic distribution widening
- strong-gate probability calibration
- global probability rescale from Weeks 1-3
- QB/pass-yards carveout from nominal three-week performance
- RB-only / TE-only rescue variants for failed Opportunity Authority work
- different top-N/share/depth-chart hacks
- Week-3-only semantic repair promotion
- public-intent coefficient fit from DEN/PIT

---

## 15. EXACT NEXT EXECUTION ORDER

The next chat should continue in this order unless live GitHub state has materially changed:

1. **Close Availability -> Opportunity Week-3 postgame status**
   - read the exact original contract/result;
   - grade descriptively only if predeclared;
   - otherwise record no valid postgame contract;
   - no coefficient fitting.

2. **Write one consolidated Week-3 prospective disposition matrix**
   covering:
   - RB Vacancy
   - public intent
   - Receiving Rule Semantics
   - Availability -> Opportunity
   - RB-PD2 Observation #1
   This should make explicit what is:
   - closed,
   - continue prospectively unchanged,
   - underpowered/HOLD,
   - production-ineligible.

3. **Then choose the next genuinely new science from the current error**
   using the Weeks 1-3 postmortem + authority attribution.
   The strongest unresolved structural facts are:
   - probability confidence remains badly overconfident;
   - upstream MC already contains a large low-mean bias;
   - late stack improves QB/rushing but barely changes receiving;
   - rush_rec/RB mean construction remains the largest mean-bias area;
   - no simple market/position carveout survives correction.
   Do not simply reopen an older failed calibration lane.

4. **Secondary mechanical lane: RNG**
   - diagnose run `36504441918` before any code edit;
   - no merge until production candidate verification is actually clean.

5. **GSIS**
   - preserve current Week-3 private snapshot as baseline;
   - next temporal work requires a later immutable weekly snapshot;
   - no raw GSIS public upload.

6. **Full Slate / paid odds**
   - do not launch any new paid OddsAPI run without explicit user authorization.

---

## 16. USER EXPECTATIONS / OPERATING STYLE

The user wants:
- actual model/science progress, not admin loops;
- visible GitHub execution when work is occurring;
- hypotheses frozen before scoring;
- no hindsight rescue;
- failures closed cleanly;
- no repeated experiments;
- GitHub canonical over chat memory;
- Issue #535 paper trail;
- brief, concrete status updates;
- seamless continuation after chat limits/timeouts;
- no asking them to re-explain prior work;
- no paid OddsAPI spend without explicit approval.

When the user says:
- "update?" -> query GitHub live and report exact current state.
- "keep going" -> perform the next authorized action; do not merely describe it.
- "timed out" -> recover from GitHub and continue.
- "I see nothing running" -> check Actions live before claiming something is active.

Never promise background work. Only say something is running when a real workflow is running.

---

## 17. FINAL HANDOFF STATE

At the time this handoff was written:

- main: `8a965f2754b4ccfa960a2a05517fd40861f49a3d`
- active postmortem branch:
  `research-week3-postmortem-execution-v1@fb80bb6c4af6c280aee53d3013315632c69a7c9b`
- latest RB-PD2 verification:
  `36631186678` = SUCCESS
- Week-3 settlement: complete
- Weeks 1-3 postmortem: complete
- projection authority attribution: complete
- RB Vacancy Week-3 grade: complete
- public intent Week-3 grade: complete
- Receiving Rule Semantics Week-3 grade: complete
- RB-PD2 Observation Week #1: complete / HOLD for insufficient support
- Availability -> Opportunity: structural gap confirmed; postgame closure still the immediate unfinished Week-3 item
- RNG production candidate: open mechanical repair lane; latest run failed and needs exact log diagnosis
- GSIS: incremental current-state value confirmed; temporal use awaits repeated weekly snapshots
- production changes from Week-3 outcome work: **NONE**
