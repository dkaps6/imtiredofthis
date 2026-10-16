# WEEK 5 FULL MODEL POSTMORTEM PROTOCOL V1

Frozen: 2026-10-13

Status: **POSTGAME DIAGNOSTIC AUTHORIZED BY USER. DIAGNOSIS ONLY. NO OUTCOME-TUNING OR PRODUCTION CHANGE FROM THIS WEEK ALONE.**

Canonical Week-5 pregame authority:
- final successful preserved-odds Full Slate run: `37860592615`
- artifact: `11585973062`
- digest: `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
- production behavior authority commit: `24704e5f1ef872ec86c36446d2c73f6d377c824c`
- original paid source run: `37852811339`
- original paid artifact: `11582178322`
- original paid digest: `sha256:4948e08003fcab0f519b35bd4b23d8c895554e4faced993238575acdc5d2e765`
- pricing status: `PRESERVED_REPLAY`
- no second OddsAPI acquisition occurred.

The preserved replay artifact is the immutable Week-5 board authority. No new sportsbook pull is authorized or needed to grade it.

## Purpose

Grade the **whole Week-5 model**, not merely the parlays.

The postmortem must separate:

1. football projection quality;
2. availability / identity / role quality;
3. opportunity-allocation quality;
4. efficiency quality;
5. distribution / probability calibration;
6. betting selection / confidence ranking;
7. alt-line / parlay construction and portfolio concentration.

A one-leg parlay miss is never evidence by itself that the football model was good or bad.

## Population A — full underlying football board

Use the preserved Week-5 `outputs/props_priced_clean.csv`.

Supported primary markets:
- pass_yards
- rush_yards
- rec_yards
- receptions
- rush_rec_yards

ATD remains separate and uncertified for dedicated science.

Deduplicate football means at:
- season
- week
- event_id
- player identity / player_clean_key
- market

Multiple books/sides are not independent football projections.

Before deduplication:
- assert duplicated offers share the same `model_proj` / `ensemble_proj` within numerical tolerance.

For line comparison only:
- use the **median unique preserved book line** for that player-market.
- do not choose the book line closest to the result.

Postgame DNP/inactive rows:
- do not silently score as zero unless participation evidence verifies zero;
- use the canonical GSIS/roster/snap settlement semantics;
- classify verified sportsbook DNP voids separately;
- unresolved identity/participation rows fail closed.

## Required full-board metrics

Overall and by POSITION x MARKET:
- settled player-market rows
- void/unresolved rows
- projection MAE
- projection RMSE
- signed projection bias = model - actual
- median preserved-book-line MAE
- median-line signed bias
- model-closer-than-line rate
- absolute model-vs-line gap
- model direction vs median line hit rate
- correlation / rank relationship between absolute model-line gap and realized absolute error
- absolute-error buckets by pregame model-line gap
- over-vs-under projection-direction performance
- worst individual misses by absolute error
- largest pregame disagreements and their realized outcomes

Do not infer a new production threshold from whichever bucket wins.

## Population B — production-selected record

Archive the immutable Week-5 priced board into an ephemeral market-track ledger and grade with the canonical GSIS-aware settlement path.

Report:
- selected settlement rows
- decided bets
- wins/losses/pushes/voids
- hit rate
- units / ROI using captured odds
- model MAE vs selected line MAE
- stated fair probability calibration
- Brier score where available
- side balance
- edge quintiles
- fair-probability quintiles
- absolute projection-vs-line-gap quintiles
- monotonicity / rank association
- largest-edge performance

No new selection cutoff may be fitted from Week 5.

## Population C — Week-5 prospective / promoted player-state lanes

Grade separately from the generic board:

### RB receiving-room share
- frozen Week-5 prospective receiving allocation;
- compare frozen candidate targets/receptions/rec-yards/rush+rec against actual;
- preserve parent baseline where the lock captured it.

### RB carry/snap allocation
- frozen Week-5 carry-share + snap-share shadow;
- compare frozen share/carry implications against realized Week-5 role where the preregistered contract permits;
- no formula retune.

### WR/TE target-share trajectory
- frozen Week-5 trajectory shadow;
- only rows eligible under the original >=4 prior same-season same-team games rule;
- compare frozen target-share allocation against actual Week-5 target shares.

### Other Week-5 locks
Any other prospective Week-5 research lock must be graded only if its original contract authorized outcome attachment. Do not invent postgame grading for a lane that did not freeze it.

## Player-level failure decomposition

For the largest meaningful misses, classify with evidence:

- `MECHANICAL / DATA / LINEAGE BUG`
- `AVAILABILITY / ROLE-TRANSMISSION FAILURE`
- `OPPORTUNITY / USAGE MODEL FAILURE`
- `EFFICIENCY MODEL FAILURE`
- `MATCHUP / CONTEXT TRANSMISSION FAILURE`
- `DISTRIBUTION / TAIL / CALIBRATION FAILURE`
- `BET-SELECTION / CONFIDENCE-RANKING FAILURE`
- `TICKET CONCENTRATION / PORTFOLIO FAILURE`
- `ORDINARY OUTCOME VARIANCE`
- `INSUFFICIENT EVIDENCE / CANNOT DISTINGUISH`

Do not call Watson or any other player a variance miss merely because the ticket lost.
Do not call every large miss structural merely because it was large.

## Ticket / parlay layer — LAST

Only after the whole-model scorecard exists:
- reconstruct the actual suggested/played Week-5 tickets available from preserved conversation/GitHub evidence;
- grade each leg;
- identify repeated legs across tickets;
- quantify portfolio concentration;
- separate full-line projection failures from alt-line survival failures;
- record Watson as a candidate concentration/projection case only after the player/model evidence is established.

## Final required outputs

1. Week-5 full-board row-level graded detail.
2. Week-5 overall / position / market / position-market scorecard.
3. Week-5 selected production settlement and calibration report.
4. Week-5 largest-miss casebook with failure classifications.
5. Week-5 prospective-lock disposition matrix.
6. Week-5 ticket construction / concentration appendix.
7. Comparison with Weeks 1-4:
   - continuity vs regime change;
   - which problems are improving;
   - which are persistent;
   - which new Week-5 evidence changes the research roadmap.

## Scientific boundary

Week-5 outcomes may diagnose existing Week-5 predictions and frozen prospective locks.

They may NOT be used to:
- choose new model inputs retroactively;
- retune thresholds on Week 5;
- change a failed study's preregistered gate;
- rescue a candidate by subgroup searching;
- rewrite the historical Week-5 prediction board.

Any new science inspired by the postmortem must be separately frozen before it is tested.
