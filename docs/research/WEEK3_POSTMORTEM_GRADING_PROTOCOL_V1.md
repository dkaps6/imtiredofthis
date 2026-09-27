# Week-3 Frozen Postmortem Grading Protocol V1

**Repository:** `dkaps6/imtiredofthis`  
**Status:** FROZEN PLAN-ONLY / NO WEEK-3 OUTCOME SCORING AUTHORIZED YET  
**Frozen from main:** `8a965f2754b4ccfa960a2a05517fd40861f49a3d`  
**Created:** 2026-09-27 while Week 3 was still in progress

## 1. Purpose

This protocol freezes the Week-3 postmortem before the full Week-3 outcome set is available.

The purpose is diagnosis, not rescue.

Production remains frozen until the actual Week-3 board is graded. The historical board emitted pregame must never be replaced by a repaired or retuned counterfactual.

## 2. Canonical Week-3 pregame board authority

- paid Full Slate run: `36293274478` = SUCCESS
- artifact: `10923570170`
- artifact digest: `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- paid-run head: `0982b62276303403e2ca58b16e6f4fc3e041f65d`
- artifact remains the historical betting-board authority
- no OddsAPI refetch is authorized merely to grade the board

The artifact contains `outputs/props_priced_clean.csv` and the upstream Week-3 production diagnostics. It must be treated as immutable input to the postmortem.

## 3. Outcome gate

Do not begin the full Week-3 grade until all games required for the board are final and the postgame actual-stat / roster / participation sources are sufficiently complete to settle the board without knowingly mixing partial results.

Until then:
- no Week-3 threshold fitting;
- no confidence-rule tuning;
- no carveouts;
- no coefficient changes;
- no production mutation;
- no postgame redesign of frozen prospective cells;
- no scientific PASS/FAIL from incomplete Week-3 outcomes.

Allowed before the outcome gate opens:
- board-integrity verification;
- identity/lineage verification;
- grader/report-path verification;
- freezing the analysis protocol and pregame selection cohorts;
- no-outcome systems work already separately authorized by prior frozen authority.

## 4. Two distinct grading populations

### A. Production-selected betting record

Use the existing production decision semantics and canonical GSIS settlement path.

This population answers:
- what production actually selected;
- hit rate;
- units / ROI where meaningful;
- stated probability calibration;
- whether production edge / confidence ordering contained signal.

Do not redefine the selection rule from Week-3 outcomes.

### B. Full underlying football board diagnostic

Separately evaluate the football projections across the full eligible Week-3 player-market board, including PASS rows where the football mean exists.

This population answers:
- projection MAE;
- signed bias;
- football mean vs sportsbook line;
- whether useful predictions existed outside the selected betting subset;
- whether selection/ranking failed even when parts of the football projection layer were useful.

Do not count every book-side quote as an independent football forecast. Deduplicate football-mean accuracy at the player-market-event level after verifying that duplicated sportsbook offers share the same football projection. Offer-level rows may be retained only for pricing/selection diagnostics.

ATD must remain separately labeled because it is execution-capable but not dedicated-science certified.

## 5. Required Week-3 metrics

Report Week 3 independently before any cumulative Weeks 1-3 view.

For each meaningful POSITION x MARKET cell and for the aggregate:
- decided bets;
- wins / losses / pushes / voids;
- hit rate;
- units;
- ROI where captured odds make the number meaningful;
- projection MAE;
- signed projection bias;
- sportsbook-line MAE;
- model-closer-than-line rate;
- absolute model-vs-line gap;
- realized absolute projection error;
- relationship between pregame model-vs-line gap and realized error;
- fair probability vs binary result;
- fixed descriptive probability bands already used by the current report path;
- Brier score where fair probability exists;
- side balance and market coverage;
- unresolved identity/settlement rows, which must fail closed rather than disappear.

Weeks 1, 2 and 3 must also be reported separately before a cumulative live result is shown.

Do not pool away a Week-3 regime break.

## 6. Confidence / ranking audit

No new threshold may be fitted from Week 3.

Descriptively compare the ordering of the frozen Week-3 board by:
- raw EV / `edge_pct`;
- model `fair_prob`;
- absolute projection-vs-line gap;
- production selected-vs-pass decision;
- equal-sized rank buckets/quintiles only as descriptive summaries.

The central question is whether stronger pregame rank implied better realized performance.

Report monotonicity and rank association where sample size permits. Do not choose a new cutoff from whichever bucket happened to win in Week 3.

## 7. Frozen pregame confidence / ticket cohorts

These cohorts were proposed pregame and must be graded exactly as stated. They may not be rewritten after outcomes.

### SAFE — 4 legs
- Justin Herbert UNDER 228.5 passing yards
- Baker Mayfield UNDER 215.5 passing yards
- Cam Ward OVER 176.5 passing yards
- Bo Nix OVER 211.5 passing yards

### MODERATE — SAFE + 1
- SAFE four legs above
- Jared Goff UNDER 261.5 passing yards

### MEDIUM — MODERATE + 1
- MODERATE five legs above
- Deshaun Watson OVER 188.5 passing yards

### HIGH — MEDIUM + 2
- MEDIUM six legs above
- Saquon Barkley UNDER 89.5 rushing+receiving yards
- Christian McCaffrey UNDER 99.5 rushing+receiving yards

### QB-ONLY 12-LEG LOTTERY
- Justin Herbert UNDER 228.5 passing yards
- Baker Mayfield UNDER 215.5 passing yards
- Cam Ward OVER 176.5 passing yards
- Jared Goff UNDER 261.5 passing yards
- Bo Nix OVER 211.5 passing yards
- Deshaun Watson OVER 188.5 passing yards
- C.J. Stroud OVER 236.5 passing yards
- Drake Maye OVER 222.5 passing yards
- Marcus Mariota OVER 169.5 passing yards
- Malik Willis OVER 176.5 passing yards
- Jacoby Brissett UNDER 227.5 passing yards
- Tyler Shough UNDER 254.5 passing yards

### LARGE 15-LEG LOTTERY — one leg per game
- Justin Herbert UNDER 228.5 passing yards (-112)
- Baker Mayfield UNDER 215.5 passing yards (-112)
- Cam Ward OVER 176.5 passing yards (-112)
- Bo Nix OVER 211.5 passing yards (-113)
- Jared Goff UNDER 261.5 passing yards (-112)
- Malik Willis OVER 176.5 passing yards (-111)
- Derrick Henry UNDER 105.5 rushing+receiving yards (-112)
- Christian McCaffrey UNDER 99.5 rushing+receiving yards (-111)
- Chuba Hubbard UNDER 86.5 rushing+receiving yards (-111)
- Bhayshul Tuten UNDER 68.5 rushing+receiving yards (-113)
- Erick All OVER 4.5 receiving yards (-115)
- Noah Fant OVER 12.5 receiving yards (-115)
- Darius Slayton OVER 10.5 receiving yards (-107)
- Rome Odunze OVER 27.5 receiving yards (-109)
- Antonio Williams OVER 13.5 receiving yards (-117)

These tickets are descriptive evidence about confidence/selection quality. They do not authorize a new staking or confidence rule.

## 8. Failure classification

Every major failure surface must be assigned one of the following, with evidence:

- `MECHANICAL / DATA / LINEAGE BUG`
- `OPPORTUNITY / USAGE MODEL FAILURE`
- `EFFICIENCY MODEL FAILURE`
- `DISTRIBUTION / TAIL / CALIBRATION FAILURE`
- `AVAILABILITY / ROLE-TRANSMISSION FAILURE`
- `BET-SELECTION / CONFIDENCE-RANKING FAILURE`
- `ORDINARY OUTCOME VARIANCE`
- `INSUFFICIENT EVIDENCE / CANNOT DISTINGUISH YET`

Do not call a result variance merely because it lost.
Do not call a result structural merely because it lost.

## 9. Preserve the model-layer distinctions

For every important row or failure cluster preserve:
1. football-model projection;
2. simulated distribution;
3. market line;
4. model probability;
5. raw EV;
6. confidence / selection decision.

A correct mean with bad probability is a calibration failure, not necessarily a football-mean failure.
A useful full-board forecast with bad selected-bet performance is a selection/ranking failure, not automatically a projection failure.

## 10. Production-active Week-3 changes to evaluate independently

### RB Rush+Receiving Conservation V2
- grade affected Week-3 rows;
- compare behavior with prior production semantics only where frozen evidence permits;
- do not reopen V2 from a few bad outcomes absent an implementation contradiction.

### Discrete Count Mean Alignment V1
- verify count-market mathematical support and realized behavior;
- separate support correctness from predictive accuracy.

## 11. Frozen prospective Week-3 lanes

### RB-PD2
- attach Week-3 outcomes only to the 46 frozen prospective locks;
- Week 3 remains Observation Week #1;
- no scientific PASS/FAIL before >=8 distinct locked weeks AND >=400 unique eligible player-games.

### RB Vacancy Opportunity V1 / Public Intent
DEN frozen pregame state:
- unavailable: Jonah Coleman
- label: `ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION`
- lead: `NONE_CLEAR`

PIT frozen pregame state:
- unavailable: Rico Dowdle
- label: `WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT`
- lead: `JAYLEN_WARREN`
- confidence: `MEDIUM`

Grade the model candidate and public-intent labels independently against realized snap/carry concentration. Never rewrite the labels.

### Receiving Rule Semantics V1
Grade unchanged cells:
- A0B0 = current production
- A1B0 = middle-open unit repair
- A0B1 = slot-alignment repair
- A1B1 = combined repair

No postgame cell redesign.

### Availability -> Opportunity
Evaluate whether the confirmed rule-order gap materially affected actual Week-3 opportunity allocation. Do not fit redistribution coefficients from Week 3.

## 12. Other frozen research boundaries

- Specialist Non-Target MC Invariance V1: no RNG repair from Week-3 results. Only the separately frozen downstream-materiality-vs-ordinary-resampling audit is authorized.
- Bayesian Current-State Transmission finding remains valid.
- Opportunity Authority Priority V1 remains `FAILED_CLOSED`; no RB-only/TE-only rescue, Bayes retuning, threshold or subgroup router.
- QB M89/M90 remains frozen; use opportunity 42.37%, efficiency 34.01%, residual 23.62% only as diagnostic context. No generic QB mean-feature hunt, catastrophic router, efficiency-volatility retry or coefficient retune.

## 13. Mechanical defect handling

If a real implementation/data-lineage defect is discovered:
1. document it before repair;
2. prove the affected population;
3. prove the earliest corrupted stage;
4. separate mechanical repair from model science;
5. rerun only what is required to establish corrected lineage;
6. preserve the original Week-3 board as the historical betting record;
7. label any repaired replay as counterfactual and never overwrite the historical Week-3 result.

## 14. Required final model-health table

After grading, produce one row per POSITION x MARKET with:
- current production science owner;
- Week 1 result;
- Week 2 result;
- Week 3 result;
- cumulative live result;
- continuity vs regime break;
- primary observed failure mode;
- already-closed study that bears on the failure, if any;
- genuinely new research real estate, if any;
- disposition: remain frozen / continue prospective collection / new preregistered study / scientifically exhausted for now.

No ranking or model modification until this evidence table exists.

## 15. Final project-health question

Only after all above evidence is assembled answer:

> After three live weeks and everything we have built, is this model showing evidence that it can become genuinely useful, or are we mostly fitting noise and moving problems around?

Base the answer on:
- predictive accuracy;
- calibration;
- market-relative performance;
- live stability;
- position/market heterogeneity;
- prospective validation;
- architecture integrity;
- durability of out-of-sample improvement.

A mixed answer must state exactly what is working and exactly what is not.

## 16. Exact first scoring action after the outcome gate opens

1. Verify all Week-3 games and postgame actual sources are complete.
2. Re-verify the canonical paid artifact identity and board digest.
3. Run the canonical GSIS/participation-aware settlement path for **Week 3 alone** first.
4. Materialize a row-level Week-3 graded detail file without changing the original board.
5. Audit unresolved/void/identity rows before computing headline metrics.
6. Generate Week-3 market x position, MAE, units/ROI, calibration and confidence-ranking diagnostics.
7. Only then run the Weeks 1/2/3 separate-and-cumulative comparison.
8. Grade the frozen prospective lanes exactly as preregistered.
9. Do not modify production until the evidence table and failure classification are complete.
