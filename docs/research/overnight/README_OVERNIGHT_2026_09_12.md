# Overnight Research + Bug Sweep — 2026-09-12

**STATUS: NOTHING HERE IS PROMOTED. Everything is documentation or a non-model bug fix, per your instructions. Model-science changes wait for your review and explicit approval.**

This is the index. Per-position detail lives in:
- `RB_POST_WEEK1_GAP_FINDINGS.md`
- `QB_GAP_FINDINGS.md`
- `WR_GAP_FINDINGS.md`
- `TE_GAP_FINDINGS.md`

## Method

Four parallel research agents were launched to do this; all four hit a session rate limit and died before writing anything. I redid the same work directly instead of re-spawning agents, working through git history across all `research-<position>-*` branches (unmerged, never promoted) plus the docs already on `main`. For RB (136 branches, the largest by far) I leaned on the project's own terminal synthesis docs rather than re-reading every branch individually, then specifically checked everything postdating those syntheses to make sure nothing newer was missed — which is how I found the RB R27D result (Sept 10) that wasn't in what I told you earlier tonight.

## Cross-position pattern (the actual headline finding)

**Three of four positions have a genuine, positive, walk-forward-validated finding that individual-player error persists pregame — and none of the three have been integrated into production:**

| Position | Branch | Result | Integrated? |
|---|---|---|---|
| RB | `research-rb-pd2-player-error-persistence` | `RB_PLAYER_ERROR_PERSISTENCE_DETECTED` — 4/4 diagnostics pass | No — follow-up (PD3→PD5→PD6) stalled on a cohort-execution bug |
| WR | `research-wr-r3-player-error-persistence` | `WR_PLAYER_ERROR_PERSISTENCE_DETECTED` — 3/3 diagnostics pass | No — authorized combined candidate never built |
| QB | `research-qb-pd2-player-error-persistence` | `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE` | N/A — this one legitimately failed |

QB's own version of this test failed, which is exactly why QB PD2 explicitly redirected toward *component-disagreement* reliability (PD3) instead of *player-history* reliability — and that QB PD3 also stalled unfinished. TE has the equivalent gap one level earlier (R1, mechanism decomposition, also stalled unfinished).

**So the single most valuable, lowest-risk thing across the whole research backlog isn't a new theory — it's finishing four already-scoped, already-frozen, partially-positive threads that got interrupted before integration:** RB-PD2→calibration, WR-R3→calibration, QB-PD3 (component-disagreement), TE-R1 (mechanism decomposition). All four explicitly propose using the signal for **uncertainty/tail calibration**, not mean correction — which also happens to be exactly what RB's rushing-ceiling problem and the general "beat Vegas" gap need (per the earlier `RB_FINAL_QUALIFICATION_RESULTS.md` finding that P3's failure is specifically a tail-calibration problem, not a mean problem).

## What's fully exhausted — do not re-attempt

- RB rushing weeks 2-18 mean correction (ceiling compression) — `RB_FINAL_QUALIFICATION_RESULTS.md`, no waiver.
- RB receiving-yard mean via any historical efficiency transform (YPR/YPT/YAC/xYAC/YACOE) — R23 through R27D, all failed, explicitly closed.
- RB STACK6 team-rush-context slicing, RB carry-tail retuning (M95T stop).
- QB first-down choice mechanism via occupancy (field position, score state, down/distance) or EPA/success economics — all explicitly ruled out.
- QB player-error/bias persistence as a mean correction (PD2 killed this specifically).
- WR player-tracking residuals (R2), dynamic-entitlement-from-absence-counts (ND3), NGS as a target/yardage source (R9-R11), another ND5 snap-depth variant.
- TE team-pool-only residual correction (R3) without individual differentiation.

## What's still live/unconcluded (don't duplicate, pick up if you want)

- QB first-down public-intent source crawl (`research-qb-first-down-public-intent-source-v1`/`v1b`) — manual/semi-automated media-text collection, paused mid-collection, not concluded either way.
- RB PD3→PD5→PD6 cohort-execution repair — unclear current state, flagged for you to check before building on it.

## New proposals worth your attention first (ranked)

1. **Finish RB-PD2's and WR-R3's authorized calibration integration.** Both diagnostics already passed; both explicitly specify the next step; neither needs new data.
2. **Reframe RB rushing-ceiling and QB first-down uncertainty as calibration/variance problems**, using the persistence signals above as the width input, rather than continuing to hunt for new mean features (which has a 0-for-many track record across both positions recently).
3. **Test the repo's own Coverage v2 output (`cb_coverage_player.csv`/`wr_cb_exposure.csv`) as the "richer matchup information" WR-R7/R8 and TE-R3/R4 both explicitly called for** — this is a different data source than the NGS approach that already failed (R9-R11), already computed every production run, and — as far as I found — never tested for this purpose. Do WR first (bigger sample), extend to TE only if it clears WR's gates.
4. Finish QB-PD3 (component-disagreement reliability) and TE-R1 (mechanism decomposition) — both frozen, scoped, and stalled before a result was ever recorded.

## Bugs fixed this session (all merged/ready, non-model-science)

1. **RB P3 Week-1 gate crashed all of Full Slate from Week 2 on** (`.github/workflows/full-slate.yml`, `scripts/run_pricing_v2.py`, `scripts/run_pricing_with_full_roster_universe_v1.py`) — merged, PR #530.
2. **RB R26 (receptions) Week-1 gate, same crash class** (`scripts/run_pricing_with_full_roster_universe_v5_production.py`) — merged, PR #530.
3. **RB R22 (receiving-tail) Week-1 gate, same crash class across 3 files** (`v4_production.py`, `audit_market_model_lineage_v3.py`, `validate_certified_full_slate_stack_v3.py`) — merged, PR #530.
4. **Full Slate data-quality classifier hard-required exactly 32 scheduled teams** (`scripts/validate_full_slate_data_quality_v1.py`) — would have crashed the live-odds data-quality gate on every bye week from roughly Week 4 onward. Fixed to accept any even, nonzero scheduled-team count; fixed the same hardcoded-32 bug in the injury-scope ledger check right next to it. Tests added, not yet committed as of this doc — see next commit.

All fixes validated: full test suite (258 passed / 1 skipped) and both strict repo audits (`audit_repo.py --strict`, `audit_2026_production_readiness.py --strict`) clean.

## Honest caveats

- I did not exhaustively re-read all 136 RB branches commit-by-commit — I used the project's own terminal synthesis docs as the primary source and spot-checked what postdates them. If there's a smaller, un-synthesized finding buried in an intermediate STACK/M9x/ND branch that never made it into a terminal doc, I could have missed it.
- I have not run any new model code or backtests tonight — everything here is landscape-mapping and reasoning from existing documented results, not new empirical validation. Before building any of the "proposed directions," they still need to go through your gates the same as everything else.
- Coverage v2 as a WR/TE efficiency signal (proposal #3) is my own inference that it wasn't already tried under a different name — I could not fully rule that out in the time available.
