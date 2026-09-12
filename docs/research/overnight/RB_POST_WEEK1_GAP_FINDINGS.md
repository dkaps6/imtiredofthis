# RB Post-Week-1 Gap Findings — Overnight Research Survey

**STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.**

RB has by far the largest research investment: 136 unmerged `research-rb-*` branches. I did not re-read every one individually (multi-day task) — I relied on the project's own terminal synthesis docs, which themselves already aggregate the STACK1-6/M91-96 lineage, then specifically checked everything that postdates those syntheses to confirm nothing newer/unaccounted-for exists. That check turned up a second major terminal result the earlier summary I gave you tonight didn't include (R27D, below) — worth having now.

## Current production state

- P3 (`RB_P3_SYNTHESIS_V1`) = Week 1 only for rushing yards. Weeks 2-18 use the base calibrated ensemble (no RB-specific model) since the Week-2-18 P3 formula failed its production gate — this is correct per the research below, not a shortcut.
- R22 (receiving-yard **tail shape**) and R26 (**receptions count**) — both Week 1 only, both promoted, both now fail closed outside Week 1 (fixed earlier tonight).
- Receiving-yard **mean** (the point estimate itself) has no promoted model at any week — it rides the generic ensemble always. This is the gap R23-R27D (below) tried and failed to close.

## Terminal synthesis #1 — rushing yards, ceiling compression (Sept 3, `research-rb-final-qualification`)

Already reported: P3's Weeks 2-18 formula (enriched carries × STACK1 YPC) passes 6/7 gates but fails the tail gate — in games with actual carries ≥20 or yards ≥100, P3 underprojects *worse* than the simple baseline. Formal disposition `FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED`, no waiver. Governance locks: don't reopen STACK6 team-rush-context slicing (`RB_STACK6_FINAL_STOPPING_EVIDENCE.md` authoritative); don't launch another retrospective carry-tail tuning family (the "M95T stop" is authoritative). Revisit only with genuinely new pregame information or forward evidence.

## Terminal synthesis #2 — receiving-yard mean, exhausted (Sept 9-10, R23→R27D — postdates #1, not in my earlier summary)

A second, more recent research thread (R23, R24, R27, R27B, R27C, R27C2, R27D — all Sept 9-10) attacked the **receiving-yard mean** translation problem for the vacancy-incumbent lead-back RB1 (i.e., the actual point projection, distinct from R22's tail-shape-only scope). The newest and last of these, **R27D** (`research-rb-r27d-yacoe-residual-v1`, Sept 10 — the single most recent commit in the entire 136-branch RB tree), tested a strict-prior YAC-over-expected (YACOE) residual correction. Result:

- Disposition: `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- Scientific gates: 4/13 passed (need more); aggregate MAE landed *worse* (C1 worse than B1 by 0.000058-0.006600 yards depending on slice); only 3/6 seasons improved (needed 4/6); the pre-identified 2023 stress cohort got worse, not better.
- Their own words: **"Combined with R23, R24, R27, R27B V2 and the R27C/R27C2 forensics, this materially narrows the RB receiving-yard mean frontier: generic historical YPR/YPT/YAC, raw target-shape context and now strict-prior xYAC/YACOE persistence have all failed to produce a qualifying lead-back mean correction. Future RB receiving-mean work must be based on genuinely new pregame information/mechanism rather than another transformation of these same historical efficiency families."**
- Explicit: "R26 opportunity/receptions remain production authority. R22 remains receiving-yard tail authority. No R27D integration design is authorized."

**This closes the loop cleanly: every conventional historical-efficiency angle on RB receiving-yard mean has now been tried and rejected, as of the day before this repo's most recent commits. Don't propose "another YPR/YAC transformation" — that lane is exhausted.**

## Genuinely live, unintegrated positive finding — RB-PD2 (Sept 6-7, interrupted before integration)

Separately from both threads above: **`research-rb-pd2-player-error-persistence`** found `RB_PLAYER_ERROR_PERSISTENCE_DETECTED` — all 4 frozen carry/yard persistence diagnostics passed on strictly-prior walk-forward history (last 8 games, min 4), consistent across weeks 5-12 and 13-18:

- Carry directional persistence: Spearman .154, +1.51 carry quartile gap — PASS
- Carry difficulty persistence: Spearman .241, +1.79 abs-carry quartile gap — PASS
- Yard directional persistence: Spearman .144, +12.4 yard quartile gap — PASS
- Yard difficulty persistence: Spearman .331, **+16.7 abs-yard quartile gap** — PASS (strongest of the four)

Their conclusion: this is *not* "give the depth-chart RB1 the most carries" (that's the separately-failed Role-Order Remap V1) — it's that **the current model has player-specific residual behavior that persists pregame, independent of the football model's own output.** Authorized next step (their words): a frozen full-stack calibration test using prior carry bias as a mean-calibration input, prior yard bias at the efficiency layer after carry effects, and **prior carry/yard difficulty as MC-width/uncertainty calibration rather than a blind mean offset.**

This is exactly the kind of "genuinely new pregame information" R27D says is now required, and it directly targets uncertainty/tail calibration — the same shape of fix that RB's own ceiling-compression problem (terminal synthesis #1) needs. **It was never integrated.** The follow-up (`research-rb-pd3-player-residual-calibration` → `-pd5-carry-only-residual-calibration` → `-pd6-positive-entitlement-residual-calibration`) hit a **cohort-execution discrepancy** (documented in `RB_PD5_COHORT_EXECUTION_DISCREPANCY.md`) that required remediation before PD6 could proceed as a confirmatory candidate — this reads as an interrupted/stalled integrity-repair, not a scientific rejection. I did not chase the exact current state of that repair thread further tonight.

## Explicit stop-rules (verbatim, consolidated across both threads)

- Do not reopen STACK6 team-rush-context slicing (`RB_STACK6_FINAL_STOPPING_EVIDENCE.md` authoritative).
- Do not launch another retrospective carry-tail tuning family (M95T stop authoritative).
- Do not attempt another transformation of historical YPR/YPT/YAC/xYAC efficiency for the receiving-yard mean (R23-R27D exhausted this).
- Do not relax or rewrite PD5 gates; do not launch PD6 against the same 2025 cohort as a purported independent confirmation without first dispositioning the replication.
- No waiver on the rushing-ceiling tail gate; no R27D-style integration is authorized without new signal.

## Proposed new research directions

1. **Finish the PD2 → calibration integration.** This is the strongest, most concrete lead in the entire RB backlog — a genuine positive finding, already gated, with an explicit, specific, non-duplicate integration design spelled out by the team's own diagnostic (mean-calibration for carries/yards, MC-width calibration for difficulty). It directly targets uncertainty/tail width, which is precisely what both terminal syntheses say the remaining RB problem actually is (ceiling compression on rushing, and — plausibly — some of the receiving-mean miss too, though R27D didn't test it this way). First step before building anything: resolve or confirm resolution of the PD5 cohort-execution discrepancy so PD6 isn't re-run on contaminated data.
2. **Reframe rushing ceiling-compression as a distributional-width problem, not a mean problem** — same idea I proposed for QB. P3 already computes a Monte Carlo distribution; the failure is specifically at the high tail of *actual* outcomes, meaning the *point mean* may be fine but the *predictive interval* is too narrow for high-workload games. This is different from "another carry-tail tuning family" (forbidden) if it's framed as calibrating the width of the existing simulator output for high-vacancy/high-workload pregame states, using PD2's difficulty-persistence signal as the width input, rather than retuning the residual-pool tail machinery itself (which is what M95T already tried and stopped).
3. **Do not touch rushing-yard mean or receiving-yard mean feature engineering.** Both are independently exhausted (final-qualification for rushing structure/allocation, R27D for receiving efficiency history). Any new work should be in calibration/uncertainty (proposals 1-2) or in genuinely new, non-historical-efficiency pregame signal categories neither thread tried (e.g., real-time personnel/injury-driven role certainty at the individual-back level, distinct from team-level vacancy already covered by STACK6).

## Open questions for you

- Do you want the PD2 calibration integration (proposal 1) built and tested first? It's the closest thing to a "shovel-ready, already-positive" result in this entire research program, for either rushing or receiving.
- Should I check the current state of the PD5/PD6 cohort-discrepancy repair before recommending anyone build on it, or treat PD2's original diagnostic (which passed cleanly, before the PD3/PD5 cohort issue) as sufficient grounding to design a fresh calibration test from scratch?
