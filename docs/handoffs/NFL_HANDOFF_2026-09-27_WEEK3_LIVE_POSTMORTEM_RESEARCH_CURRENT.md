# NFL HANDOFF — 2026-09-27 — WEEK-3 LIVE / POSTMORTEM + RESEARCH CURRENT

**STATUS: CURRENT CANONICAL CONTINUITY HANDOFF**

This handoff supersedes the earlier Week-3 paid-board checkpoint for immediate execution. GitHub is canonical. Chat memory is secondary.

## 1. Current canonical repository state

Repository:
`dkaps6/imtiredofthis`

Current main:
`d2a1366f04d38cfd1a4ad3a10a49d8f2102afafd`

Current-main verification:
- Repo CI `36332026225` = **SUCCESS**
- no-live Full Slate `36332026072` = **SUCCESS**
- Archive Market Track Record `36332250341` = **SUCCESS**

Recent merged research / mechanical lineage:
- PR #656 — QB M89/M90 opportunity-efficiency decomposition — **MERGED**
- PR #657 — RB-PD2 forward-lock mechanical hardening / Week-3 lock preservation — **MERGED**
- PR #655 — Week-3 specialist non-target MC path-drift audit — **MERGED**

Do not infer that a merged research diagnostic means a production model change. These merges primarily preserve diagnostic machinery/results and mechanical contracts. Production science changes remain separately governed.

## 2. Canonical Week-3 paid betting board remains unchanged

Canonical paid Full Slate:
- run `36293274478` = **SUCCESS**
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`

Paid-board data quality remains:
`FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS`

Mechanical paid-board repair chain remains closed:
- PR #650 — Zach Ertz sportsbook prop quarantine — MERGED
- PR #651 — verified identity alias repair — MERGED
- PR #652 — Week-3 RB market-lineage/workbook semantics — MERGED

Important:
- no additional OddsAPI spend is required to inspect or grade the preserved Week-3 board;
- raw workbook EV is a snapshot signal, not a calibrated confidence/staking tier;
- ATD is still execution-capable but **not dedicated-science certified**.

2026 Weeks 1-2 settled descriptive market record before Week 3:
- QB pass yards: **36-16 (69.23%), +15.99u**
- RB rush yards: **48-46**
- RB rush+rec: **32-33**
- RB receiving yards: **32-31**
- WR receiving yards: **72-72**
- WR receptions: **70-75**
- TE receptions: **30-34**
- TE receiving yards: **28-41**

The user reports that the live Week-3 board performed very poorly today. Treat that as an urgent reason for an objective postmortem, **not** as authorization for post-hoc tuning. Final scientific conclusions must come from canonical grading after Week-3 outcomes are complete.

## 3. Week-3 RB-PD2 prospective shadow observation is VALID

Earlier paid board `36293274478` did **not** capture the RB-PD2 shadow because the research switch was off.

A later explicitly authorized paid Full Slate was run with:
- `fetch_live_odds=true`
- `rb_pd2_shadow_capture=true`

Paid shadow-capture Full Slate:
- run `36330757181`
- head `37deff5b51b5ec48117556b91d12dba1d4ca1fad`
- sportsbook acquisition succeeded
- canonical pricing later failed separately
- **this run is not a replacement betting-board authority**

Same-process pregame capture:
- artifact `10934814338`
- digest `sha256:f4f215eaa1ba4fd6931dd327300599e4052cf984ba0f887d0bd1755c2c0a16ba`
- 54 RB/HB/FB rush-yard baseline rows
- 25,000 draws per captured row
- no outcomes present
- sportsbook inputs to candidate = false
- production mutated = false

Two mechanical defects were fixed without changing science:
1. forward-history restore was success-gated and could skip after unrelated pricing failure;
2. the history producer/consumer used inconsistent pre-CSV vs persisted-CSV digest semantics.

Mechanical recovery authority:
- run `36331514633` = **SUCCESS**
- artifact `10935529140`
- digest `sha256:94f168125dc889b6a747da5f3b3e3829d2fd4bf7a9b6b3d9f4d2d79ac0b59b6e`

Recovered Week-3 lock:
- valid = true
- target = 2026 Week 3
- locked rows = **46**
- exclusions = 8, all frozen insufficient-prior-game exclusions
- integrity failures = 0
- width multiplier > 1.00 on 34/46 rows
- max width multiplier = 1.2942418426x
- persisted `2026-09-27T15:59:26.120129Z`
- earliest kickoff `2026-09-27T17:00:00Z`
- pregame buffer = **60.56 minutes**
- outcome present at lock = false
- production changed = false

**Disposition: Week 3 is Observation Week #1 for the frozen RB-PD2 forward/shadow confirmation.**

Do not declare a scientific PASS/FAIL until the frozen support floor is reached:
- at least **8 distinct prospectively locked weeks**
- at least **400 unique eligible player-games**

Week-3 outcomes may be attached to the locked rows when final, but no promotion/rejection decision may be made from Week 3 alone.

## 4. QB M89/M90 opportunity-efficiency decomposition completed

Authoritative diagnostic:
- run `36330107392` = **SUCCESS**
- artifact `10935138252`
- digest `sha256:8699230a732a754c3487af650d793503c3a6683eb070b730e604c2cbb32251c1`
- exact clean 2024-2025 M89 authority
- 894 rows / 61 QBs
- candidate models fit = 0
- sportsbook inputs = false
- production changed = false
- max decomposition identity gap = 5.68e-14

Disposition:
`QB_M89_OPP_EFF_DECOMPOSITION_COMPLETE`

Remaining promoted M89 error mass:
- opportunity / passing attempts = **42.37%**
- efficiency / YPA = **34.01%**
- non-factor residual = **23.62%**

Oracle MAE from M89 baseline 60.4901:
- actual attempts + predicted YPA = 49.5476
- predicted attempts + actual YPA = 49.4583
- both actual primitives = 29.7664

100+ yard catastrophic misses:
- 160 total
- **98 opportunity-dominant**
- 56 efficiency-dominant
- 6 residual-dominant

Interpretation:
- the strongest unresolved QB information need is genuinely new **week-specific pregame passing-opportunity / intent information**;
- M82 anti-reinvention ledger remains binding;
- no generic pass-rate/game-script retune, catastrophic router, efficiency-volatility retry, M89 coefficient retune, or C2 change is authorized;
- `next_candidate_authorized=false` from this diagnostic alone.

## 5. Week-3 Specialist Non-Target MC Invariance V1 — drift CONFIRMED

Frozen disposition:
`SPECIALIST_NONTARGET_MC_PATH_DRIFT_CONFIRMED`

Authority:
- run `36330399450` = **SUCCESS**
- artifact `10935587149`
- digest `sha256:f1bf35878758478a3871436b48531432c02a8102663c361942d7e7ff0669db39`

All entitlement/provenance integrity gates passed. No new OddsAPI fetch. No Week-3 outcomes. No production mutation.

Result:
- TE-R5P: 560 / 745 protected semantically unrelated keys drifted (75.17%)
- all 75 protected pass-yard keys moved at the TE stage
- max absolute protected pass mean drift = 0.8089 yd
- all 335 protected rush-yard keys moved
- WR-R15: 515 / 655 protected semantically unrelated keys drifted (78.63%)
- all 75 protected pass-yard keys moved at the WR stage
- max absolute protected pass mean drift = 1.0445 yd
- all 290 protected rush-yard keys moved

Interpretation is narrow:
- finite-MC RNG/path dependence is real;
- this does **not** prove the theoretical marginal football law is wrong;
- no repair is authorized yet.

The only authorized follow-up from this result is a separately frozen **downstream materiality audit**:
- fair probability movement
- side changes
- EV movement
- HAS EDGE / PASS changes
- publishability/rank movement
- compare against ordinary finite-MC resampling noise under identical football inputs

Do **not** implement split RNG streams, common-random-number routing, cache/splice fixes, seed changes, or larger iteration counts before that materiality study.

## 6. Bayesian Current-State Transmission V1 — systemic mismatch confirmed

Authority:
- run `36330210860` = **SUCCESS**
- artifact `10935860713`
- digest `sha256:0423d3e2e4044880d8894297c154319b26dc9f07d41ed05da3a5851708228293`
- 6,785 scored rows
- parameters fit = 0
- candidate variants = 0
- sportsbook inputs = 0
- 2026 outcomes = 0

Disposition:
`BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`

Production PlayerForm four-game pseudo-prior beat the downstream empirical-Bayes posterior for fast-moving opportunity metrics in both 2024 and 2025:
- RB rush share:
  - 2024 0.124324 vs Bayes 0.135835
  - 2025 0.112230 vs Bayes 0.125110
- WR target share:
  - 2024 0.062149 vs 0.064143
  - 2025 0.061479 vs 0.063904
- TE target share:
  - 2024 0.048414 vs 0.050345
  - 2025 0.045134 vs 0.046843

Exact two-completed-game / Week-3 analogue also favored PlayerForm for all three:
- RB rush share delta +0.010749; clustered 95% CI [0.002179, 0.019754]
- WR target share delta +0.003876; CI [0.001159, 0.006594]
- TE target share delta +0.003399; CI [0.000815, 0.005900]

Mechanism:
- PlayerForm current-state weight at two games = 33.33%
- veteran production Bayes current-state weight = 18.18%
- simulation rules prefer downstream Bayes values, so validated fast current-state opportunity is re-shrunk.

This finding remains valid, but it did **not** itself authorize Bayes retuning.

## 7. Opportunity Authority Priority V1 — FAILED CLOSED

Parent diagnostic above motivated one frozen full-stack candidate: consume exact PlayerForm fast-state opportunity for:
- RB/HB/FB rush share
- WR target share
- TE target share
while preserving Bayesian efficiency metrics and all other rules/specialists.

Authoritative execution:
- branch `research-opportunity-authority-priority-v1`
- run `36332720457` = **SUCCESS mechanically**
- head `818b30d2e7369654527298f6014badc7e2ffa4a4`
- artifact `10935999653`
- digest `sha256:4fde0a72c3aef7ae2412580333a7fefcd3078e53a31f0e9ebb945fd6a6ed1bb5`
- candidate variants = 1
- fitted parameters = 0
- sportsbook inputs = 0
- production changed = false

Frozen scientific disposition:
`OPPORTUNITY_AUTHORITY_PRIORITY_V1_FAILED_CLOSED`

Direct opportunity:
- RB improved both seasons:
  - 2024 4.4447 -> 4.1258
  - 2025 4.3840 -> 3.9677
- TE improved slightly both seasons:
  - 2024 1.6736 -> 1.6521
  - 2025 1.5732 -> 1.5622
- WR worsened both seasons:
  - 2024 2.0735 -> 2.1572
  - 2025 2.0544 -> 2.0907

Pooled downstream:
- RB rush att: 3.5402 -> 3.4649
- RB rush yards: 21.0496 -> 20.5132
- RB rush+rec yards: 25.5307 -> 24.7569
- TE receptions: 1.2922 -> 1.2797
- TE receiving yards: 15.6806 -> 15.5884
- WR receptions: 1.4443 -> 1.4593 (worse)
- RB receptions: 1.1412 -> 1.1650 (worse)
- RB receiving yards: 11.1075 -> 11.3091 (worse)
- WR receiving yards failed the 2025 non-worse gate

Integrity gates held:
- ML/State invariant
- QB final passing mean invariant
- RB Rush+Receiving V2 exact
- team-volume inputs invariant
- protected-rule gap = 0

This is a **scientific failure**, not a mechanical failure.

Permanent disposition for this exact candidate:
- do not rescue by dropping WR after seeing the output;
- do not create RB-only or TE-only versions post hoc;
- do not retune Bayes constants;
- do not add thresholds/subgroup routers from this result;
- branch remains research-only / unmerged.

Issue #535 result authority:
comment `5860608905`

## 8. Frozen Week-3 prospective lanes remain unchanged

### RB Vacancy Opportunity V1

DEN:
- Jonah Coleman unavailable
- frozen label: `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- lead: NONE_CLEAR

PIT:
- Rico Dowdle unavailable
- frozen label: `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
- lead: JAYLEN_WARREN
- confidence: MEDIUM

PR #642 remains open because the cohort is frozen pregame.

After outcomes:
1. grade RB Vacancy V1 independently;
2. grade frozen public-intent labels against actual carry/snap concentration;
3. never rewrite the pregame labels.

### Receiving Rule Semantics V1

Frozen Week-3 cells:
- A0B0 = current production
- A1B0 = middle-open unit repair
- A0B1 = slot-alignment repair
- A1B1 = combined repair

Confirmed deterministic defects:
- middle_open_rate is percentage-point scaled while production consumes it as 0-1;
- SWR alignment exists upstream but is dropped before production SLOT labeling.

Historical Stage 2 remains:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`

After games:
- attach outcomes to the unchanged cells;
- grade exactly as frozen;
- no postgame redesign/rescue.

### Availability -> Opportunity

Status remains:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

Frozen Week-3 authority:
- 11 definitive-unavailable skill players
- 6 WR / 3 TE / 2 RB-FB
- 0 survived eligible roles
- 0 survived PlayerForm
- 0 survived ModelContext
- 0 production-reachable injury-redistribution rows

Do not resurrect 60/30/10 and do not fit vacancy coefficients from Week-3 results.

## 9. Production-active locks that remain in force

### Discrete Count Mean Alignment V1
Production-active for:
- receptions
- rush_att

Zero/nonfinite-MC = exact no-op.
Non-count markets retain continuous semantics.

### RB Rush+Receiving Conservation V2
Production-active and locked.

Historical 2024-25 RB MAE:
27.6853 -> 25.5718

Do not reopen absent a concrete new defect or genuinely new independent evidence.

## 10. Closed / do-not-repeat families

Remain closed:
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
- RB1/RB2 role-aware Bayesian rescue
- depth-chart priority rescue
- alternate top-N / share-threshold rush allocation rescue
- generic QB mean feature hunt after M89/M90
- QB catastrophic router
- QB efficiency-volatility retry
- Opportunity Authority Priority V1 exact global source-priority candidate

Rush-att zero-MC remains:
`CLOSED_AS_HARD_SUPPORT_CONFLICT_NO_REPAIR_AUTHORIZED`

## 11. Important Week-3 betting-board interpretation

Pregame board analysis found:
- QB passing was the only clearly strong live-season market through Weeks 1-2;
- giant RB raw EVs were not automatically trustworthy;
- Week-3 RB rushing board showed an RB1-down / RB2-up flattening pattern associated with Bayesian position-group shrinkage;
- that observation did **not** authorize reopening the failed Rush Pool / Bayesian-rescue family;
- PlayerForm QB2/QB3 labels on some starters were usage-rank semantics, not a starter-authority bug.

Do not use the poor Week-3 outcome to retroactively invent a new confidence threshold. Grade first.

## 12. Immediate posture after the user's Week-3 frustration

The user is seriously considering stopping the project because the Week-3 live board looked very poor.

The correct technical response is:
- do **not** minimize the bad day;
- do **not** manufacture a rescue;
- do **not** mutate production before grading;
- preserve every pregame artifact/label;
- let Week 3 indict the model fairly if that is what the data show.

The next chat should be willing to conclude that parts of the architecture are not good enough if the evidence says so.

## 13. Exact next actions

### A. First priority once Week-3 outcomes are complete
Run a **brutally objective Week-3 postmortem** against the frozen pregame board.

Required outputs:
- full position x market record;
- hit rate, units/ROI where meaningful;
- model-vs-line gap vs realized error;
- projection MAE by market/position;
- opportunity vs efficiency vs distribution/tail attribution where supported;
- compare Week 3 against Weeks 1-2 without pooling away regime changes;
- identify whether failures are:
  - football mean
  - opportunity allocation
  - efficiency
  - distribution/calibration
  - availability/role transmission
  - identity/data/plumbing
  - bet-selection/confidence ranking
  - ordinary variance

Do not fit thresholds to the same Week-3 outcomes.

### B. Grade all frozen Week-3 prospective science
After outcomes are final:
- RB Vacancy Opportunity V1
- DEN/PIT public-intent concentration labels
- Receiving Rule Semantics A0B0/A1B0/A0B1/A1B1
- Availability -> Opportunity descriptive behavior
- attach Week-3 outcome to the 46 RB-PD2 locks

For RB-PD2, attachment is allowed; scientific PASS/FAIL remains forbidden until >=8 weeks / >=400 unique eligible player-games.

### C. Open no-outcome systems lane if useful before all outcomes are final
The only directly authorized follow-up from the confirmed specialist RNG/path drift is:
**freeze and run downstream materiality vs ordinary finite-MC resampling noise.**

No repair before materiality.

### D. Opportunity Authority Priority V1
Close/document the research branch as failed. Do not merge its candidate behavior into production. Do not rescue it.

### E. Paid data
Do not launch another OddsAPI request unless the user explicitly authorizes it in the new chat.

## 14. Current branch / PR guidance

Important open PRs:
- #642 — frozen Week-3 RB Vacancy cohort; keep open until grading
- #615 — RB-PD2 forward/shadow plan-only authority; Week 3 is now valid Observation Week #1

Old open PRs #635/#636/#619/#616/#608/#606/#601/etc may represent historical/superseded research. Do not assume an open PR means active science. Verify disposition before acting.

Research branch:
`research-opportunity-authority-priority-v1`
- authoritative run finished
- scientific disposition FAILED CLOSED
- should not be merged as production behavior

## 15. Timeout / memory-efficient resume protocol

Read only:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this file
4. Issue #535 from comment `5857262166` onward, especially:
   - `5857330835` specialist MC drift result
   - `5857337113` Bayesian current-state mismatch
   - `5857495417` Week-3 RB-PD2 lock + QB decomposition
   - `5857596405` Opportunity Authority launch
   - `5860608905` Opportunity Authority FAILED CLOSED result
5. query GitHub live for current main, PRs, branches, and Actions before acting

Do **not** recursively load older handoffs unless this handoff explicitly points to one for a specific frozen authority.

## 16. One-sentence state

**Week 3 is live and appears to have exposed serious real-world weakness; production must stay frozen until the board is objectively graded, RB-PD2 Week 3 is valid prospective Observation #1, QB remaining error is opportunity-dominant, specialist finite-MC path drift is confirmed but unproven materially, Bayes fast-state transmission is a real mismatch but the first full-stack source-priority fix failed closed, and no post-hoc rescue is authorized.**
