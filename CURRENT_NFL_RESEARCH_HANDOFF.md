# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## LATEST CONTINUATION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`
2. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

---

## Exact current stop

The active research lane is the **shared QB/receiver opportunity layer**, specifically the unresolved week-specific first-down pass/run choice mechanism.

Recent frozen diagnostics established the following chain:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION -> SURVIVES FIELD POSITION -> SURVIVES SCORE STATE -> NOT EXPLAINED BY SIMPLE PASS-vs-RUN RECENT ECONOMICS -> LIKELY WEEK-SPECIFIC PREGAME GAME-PLAN / PLAY-CALL INTENT`

The same mechanism remains materially shared with receiver opportunity error.

### Most recent authoritative diagnostics

**First-down field-position decomposition**
- Run `34545401969`
- Job `103096807810`
- Artifact `10178825973`
- digest `sha256:734b1603b69f45a39f33e6d4770690bcc81b42f8a3b3dae33b74dce6c2cbeb37`
- preserved result commit `55d71b28d038d102c54b25a132c0cca79040b561`
- disposition `FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC`
- result: field-position occupancy is not the explanation; within-zone pass propensity remains dominant and shared with WR opportunity error.

**First-down score-state decomposition**
- canonical Run `34548863668`
- Job `103107307133`
- Artifact `10180050013`
- digest `sha256:969b3a3a6c435c08034d5f63dc88ca44991aa99e27b0eca54698cba174beaef9`
- preserved result commit `d819296f24f459170747040951177ae113704fbb`
- disposition `FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC`
- result: the shared first-down miss persists after score state is held constant; score-state reference level is negligible and occupancy is secondary.

**First-down choice economics D1**
- preserved result commit `ac9e7706d02d11dbcb3cfe9cef02f2bc03d95058`
- clean untouched-2023 failure; do not retune/rescue.

No recent QB opportunity diagnostic has changed production.

---

## Public pregame intent source audit — manual crawl paused

Current branch:
- `research-qb-first-down-public-intent-source-v1`

Frozen source plan:
- `4466c88a584a22f6cfebda41fd946a3cdc5bc220`

Current branch head at handoff write:
- `61f3931394c383498ebd58f6dbdb923e6feb68ba`

Manual deterministic collection reached 2023 Week 2 through Detroit with 11 timestamp-safe eligible team-weeks. Those rows are preserved as a small feasibility/gold sample only.

**Explicit user decision:** do not continue a manual team-by-team/game-by-game historical crawl. It is operationally too expensive and could consume dozens of hours.

The frozen V1 source audit has not scientifically passed or failed; it is incomplete and manually non-scalable.

### Next authorized action

If continuing this idea, freeze a **separate automation-first V1B source-retrieval/validation design** before doing more collection.

Required operating boundary:
- automatic team-week universe;
- fixed official-source-first hierarchy + predeclared local fallback;
- automated timestamp/speaker/opponent/source/semantic extraction;
- manual review only for ambiguous rows;
- validate retrieval precision/recall/timestamp correctness against a small gold sample before scaling;
- no outcomes/residuals/sportsbook/model fitting during source qualification;
- one working day / roughly 6-10 hours maximum total research budget for this hypothesis;
- if scalable retrieval and meaningful predictive promise are not demonstrated within that budget, preserve findings and close/move on.

Do not silently rewrite the original V1 gates after seeing the initial 11/11 manual coverage.

---

## Current promoted production stack — unchanged

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 + WR-R15 WR2+
- TE: TE-R5P
- RB rushing: P3
- RB receptions: R26
- RB receiving-yard mean: existing production YPT/mean path
- RB receiving-yard distribution/tails: R22 using frozen R19 assets, mean-preserving
- current player availability/current roles: promoted availability-first production plumbing
- sportsbook: downstream only
- master workbook: `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

Current operational production authority:
- `3079d8ab0512c5a1304662609e3e880d6846292f`

Protected scientific/model authority:
- `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Main immediately before this handoff update:
- `e7c952a293722082566123d8a1bf3620a592559c`

---

## Historical Vegas certification — unfinished separate lane

Do not claim that the current full model has already been broadly proven against historical Vegas player-prop lines.

- M60 historical paid Odds API benchmark was blocked by quota economics.
- M60B free Action-derived archive contained substantial prop data but failed trustworthy canonical game-ID reconciliation, yielding zero valid matches; grading was correctly skipped.
- Historical Vegas grading infrastructure exists, but broad certification of the current stack remains unfinished.
- Future no/low-cost routes: repair Action natural-key mapping, audit ParlayAPI historical closing lines/bulk CSV, and archive our own 2026 pregame boards.

Keep this market-certification work separate from upstream football research.

---

## Parked / do not reopen casually

- NE-SEA grading remains parked by explicit user instruction.
- generic QB mean feature hunt beyond M89/M90;
- M64/M65 generic pace/score-state families;
- M67 generic intent/history transforms;
- M68 playcaller/opening-script/leverage family;
- M81/FTN tactical family;
- M87/M88 funnel + short/intermediate interaction;
- directional personnel family that failed coverage;
- designed-run D1;
- schedule/rest D1;
- M89 opportunity reparameterization A1;
- fixed 0.59 pass-rate anchor promotion;
- first-down relative choice economics;
- field-position occupancy as generic correction;
- score-state occupancy/reference-level as generic correction.

Preserve all failures and mechanical repairs as anti-reinvention evidence.

---

## Resume rule for next chat/session

Read the Sep 11 detailed handoff first, then verify live GitHub main and active branch heads. Do not trust stale chat memory over newer committed evidence.

Immediate next task if nothing newer exists:

**freeze and build the automation-first public-intent retrieval/validation V1B under the hard one-day / 6-10h cap, or close the lane if scalable retrieval cannot be demonstrated quickly.**

No production change is authorized from the current public-intent source audit or any recent first-down diagnostic.
