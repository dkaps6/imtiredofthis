# Overnight Research + Bug Sweep — 2026-09-12

**STATUS: NOTHING HERE IS PROMOTED. Everything is documentation or a non-model bug fix, per your instructions. Model-science changes wait for your review and explicit approval.**

**CRITICAL UPDATE (read first)**: GPT-5.6 found, and Claude independently confirmed, a data-integrity bug in the Vegas benchmark cohort that predates this whole research thread — 94.5% of rows have a `game_id` from the wrong season, meaning the "Vegas line" joined for grading is very often a different game's line, not the actual matchup's line. See `BENCHMARK_GAME_ID_SEASON_MISMATCH_CONFIRMED.md`. **Every ROI/win-rate/STRONG-gate finding below (situational edge hunts V1/V2, the holdout tests, the full-market scan) is downgraded from `UNVERIFIED` to `INVALID_PENDING_REBUILD`.**

**Correction (GPT-5.6 caught this, Claude's first pass had it wrong)**: the "model MAE vs Vegas MAE gap" is *not* safe either, only the model-vs-actual MAE alone is provisionally usable. The comparative "model loses to Vegas" claim depends on the Vegas line, which is exactly what the broken `game_id` corrupted. So: model-vs-actual MAE = provisionally usable; Vegas-vs-actual MAE, the model-vs-Vegas gap, and everything ROI/probability/STRONG/holdout-related = **all invalid pending rebuild.** GPT-5.6 is leading the clean rebuild (fail-closed identity assertions before any grading); Claude will independently re-verify and re-run everything fresh once that lands. Do not act on any comparative-to-Vegas number in this folder until then.

**REBUILD LANDED, RE-GRADED (read this next)**: GPT-5.6 published PR #541 with a fail-closed identity rebuild; its own audit and Claude's independent spot-check both confirm zero season/week/team/opponent identity failures across 17,715 graded rows. Claude then independently re-ran the full four-part re-grade directly off the clean artifact. See `CLEAN_BENCHMARK_INDEPENDENT_REGRADE_V1_RESULT.md` for full detail. Headline results, now trustworthy: **Vegas still beats the model's raw MAE in every one of 10 season/market cells and every one of 8 season/position cells — zero exceptions.** New finding this cleanly isolated for the first time: the model carries a large, systematic *under-projection* bias in every market (e.g. 2024 rush_rec_yards model bias -19.0 vs Vegas bias -4.6), much bigger than Vegas's own bias. The STRONG-gate overconfidence pattern (worst where component models agree most) replicates unchanged on the clean cohort — confirms it was never caused by the identity bug. Of the 6 situational candidates found pre-fix, only 4 survive on the clean cohort; 2 (`rec_yards` prob_edge OVER, `receptions` prob_edge OVER) vanished entirely — direct proof some of what looked like an edge before was the corrupted join itself, not football signal. Every surviving candidate still carries the disclosed probability-fidelity caveat (Normal-CDF approximation, not production's real simulated distribution) and is not promotable.

This is the index. Per-position detail lives in:
- `RB_POST_WEEK1_GAP_FINDINGS.md`, `RB_PD_CHAIN_STATUS.md`
- `QB_GAP_FINDINGS.md`, `QB_PD3_AND_TE_R1_RECOVERED_RESULTS.md`
- `WR_GAP_FINDINGS.md`, `WR_R3_COMBINED_CANDIDATE_STATUS.md`
- `TE_GAP_FINDINGS.md`, `QB_PD3_AND_TE_R1_RECOVERED_RESULTS.md`
- `COVERAGE_V2_EFFICIENCY_SIGNAL_NOVELTY_CHECK.md`
- **`SITUATIONAL_EDGE_HUNT_V1_RESULT.md`, `SITUATIONAL_EDGE_HUNT_V2_WR_R15_TE_R5P_APPLIED.md`, `RECEPTIONS_UNDER_HOLDOUT_TEST_V1_RESULT.md`, `FULL_MARKET_HOLDOUT_SCAN_V1_ANALYSIS.md` — read in order. The candidate downgrades under a real holdout test, and the full-market scan surfaced more candidates with a live suspicion they're a measurement artifact, not signal. See below.**
- **Cross-audited jointly with GPT-5.6 in [GitHub Issue #535](https://github.com/dkaps6/imtiredofthis/issues/535) — that thread is now the live source of truth for this candidate's status, more current than this file.**

## Situational edge hunt (new, most actionable finding of the whole sweep)

Instead of more feature-hunting, re-sliced the *existing* real 2024-2025 full-stack Vegas benchmark (`data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv`, 16,973 already-graded bets) by situational context instead of just the tier cut it already reported. 146 slices tested; one real candidate survived a both-seasons-independently-positive bar:

**Receptions market, model's highest-confidence quartile, UNDER side only: n=1,258, win rate 52.5%, ROI +2.66%/unit, positive independently in both 2024 (+2.43%) and 2025 (+2.90%).** Full writeup, including the statistical-honesty caveats (146 tests run, multiple-comparisons exposure) and why this one looks real rather than noise, in `SITUATIONAL_EDGE_HUNT_V1_RESULT.md`.

**V2 update**: that V1 result graded receptions on the base engine only (disclosed gap — WR-R15/TE-R5P not yet applied). You located and provided the two canonical GH Actions validation artifacts (`wr_r15_confirmation_predictions.csv`, `te_r5_oos_player_casebook.csv` — direct download was blocked by this session's network egress policy, an org-level block on Azure blob storage, not a repo issue) and I applied them to the same cohort and re-graded with the unmodified grading script. **The edge survives**: softens from +2.66% to +1.50% pooled ROI but stays positive independently in both seasons (2024 +2.07%, 2025 +0.92%). The subset of bets the real promoted models actually touched shows a stronger edge (+2.94%) than the untouched subset (+0.70%) — encouraging, but the 2025-specific slice of that (n=21, almost all TE-R5P since WR-R15 has zero 2025 coverage by its own frozen contract) is too small to lean on. Full writeup in `SITUATIONAL_EDGE_HUNT_V2_WR_R15_TE_R5P_APPLIED.md`.

**Holdout test (self-attack, per the #535 cross-audit protocol)**: V1/V2's "positive both seasons" check is weaker than it looks — the prob_edge quartile threshold that *defines* the candidate was fit on the pooled 2024+2025 sample, so 2025 data influenced the rule used to grade 2025. Refit the threshold on ONE season only, froze it, and graded the OTHER season blind, both directions: **fit-2024→test-2025: n=577, ROI +0.61%; fit-2025→test-2024: n=543, ROI +1.59%.** Both still positive and win rate stays consistent (~52.7-52.9%) either direction, but this is materially weaker than the pooled +1.50%, and at this n a ~52.7% win rate carries a ~2.1pp binomial standard error against a 50% null — not distinguishable from noise on hit rate alone. **Downgraded from "validated, modest edge" to "directionally plausible, unconfirmed at this sample size."** Full writeup in `RECEPTIONS_UNDER_HOLDOUT_TEST_V1_RESULT.md`. Not promotable, not disproven — needs more data (a real third season, or 2026-forward tracking) to move either direction.

## Method

This ran in two passes. Pass 1: four parallel research agents to survey each position; all four hit a session rate limit and died before writing anything, so I redid that survey directly — working through git history across all `research-<position>-*` branches (unmerged, never promoted) plus the docs already on `main`. Pass 2 (after capacity reset, at your request to use agents): three more parallel agents, each chasing a specific loose end pass 1 surfaced (the RB PD-chain's true status, whether WR-R3's authorized follow-up was ever built, whether Coverage v2 had already been tried as a WR/TE efficiency signal). Those three completed cleanly and each found something pass 1 didn't have.

A pattern showed up twice and is now the standard first move for any "this thread looks stalled" claim in this repo: **check GitHub Actions run history before assuming a thread never finished.** QB-PD3, TE-R1, RB-PD3, RB-PD4, RB-PD5, and a WR player-bias-shrink attempt had *all* actually executed successfully on GitHub's runners — every one of them just never got a result doc committed afterward. All six were recovered from real job logs (cited run/job/artifact IDs, nothing re-run) rather than re-executed.

## Cross-position pattern (the actual headline finding, now fully resolved)

Three of four positions have (or had) a genuine, positive, walk-forward-validated finding that individual-player error persists pregame:

| Position | Diagnostic | Result | Follow-up status |
|---|---|---|---|
| RB | PD2 | `RB_PLAYER_ERROR_PERSISTENCE_DETECTED` — 4/4 pass | PD3/PD4/PD5 all **recovered and FAILED** (close misses, mostly on yard p90). PD6 **genuinely blocked** — cohort-execution discrepancy unresolved, PD6 itself never implemented (byte-identical to PD5, zero CI runs). |
| WR | R3 (`research-wr-r3-player-error-persistence`) | `WR_PLAYER_ERROR_PERSISTENCE_DETECTED` — 3/3 pass | R3's own *combined* candidate was never built. A related, differently-branched R3 (`research-wr-r3-player-bias-persistence`) ran a first-pass bias-shrink and it **recovered and FAILED**. A correctly-scoped implementation-ready design spec now exists for the real combined candidate (never attempted). |
| QB | PD2 | `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE` | Legitimately failed originally. Its redirect, PD3 (component-disagreement reliability), **recovered**: `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE` — also null, but reassuring (no state where the synthesis correction breaks down). |
| TE | R1 (mechanism decomposition, one level earlier than the other three) | — | **Recovered**: `TE_MECHANISM_DECOMPOSITION_ACTIONABLE`, all 5 gates pass. Confirms targets/entitlement dominate TE error (45.2%), validating the R2→R5 strategy that became TE-R5P, and quantifies the remaining efficiency gap at ~55% of error mass. |

**Net effect of the follow-up pass: every thread that looked "stalled, might be shovel-ready" in the first pass is now either a documented failure, a documented pass, or a documented hard block with a named cause.** Nothing is ambiguous anymore. The one live opportunity that survived scrutiny is the WR-R3 combined candidate's design spec — real, unbuilt, correctly scoped, but still unvalidated until actually run.

## What's fully exhausted — do not re-attempt

- RB rushing weeks 2-18 mean correction (ceiling compression) — `RB_FINAL_QUALIFICATION_RESULTS.md`, no waiver.
- RB receiving-yard mean via any historical efficiency transform (YPR/YPT/YAC/xYAC/YACOE) — R23 through R27D, all failed, explicitly closed.
- RB STACK6 team-rush-context slicing, RB carry-tail retuning (M95T stop).
- RB residual/MC-width calibration via PD3's, PD4's, and PD5's specific designs — all three now confirmed failed (close, but failed; yard p90 is the recurring blocker). A materially different design would be needed, not a retry of these.
- QB first-down choice mechanism via occupancy (field position, score state, down/distance) or EPA/success economics — all explicitly ruled out.
- QB player-error/bias persistence as a mean correction (PD2 killed this specifically). QB internal-disagreement/correction-magnitude as an unreliability flag (PD3, now recovered — null).
- WR bias-shrink applied as a direct patch to M38's MC output (the one build that exists) — confirmed failed. WR player-tracking residuals (R2), dynamic-entitlement-from-absence-counts (ND3), NGS as a target/yardage source (R9-R11), another ND5 snap-depth variant.
- TE team-pool-only residual correction (R3) without individual differentiation.
- Coverage v2 team-level rates (`coverage_man_rate`/`coverage_zone_rate`) as a WR receiving-efficiency feature — already backtested via feature ablation (run `32316784561`, Aug 2026, recovered from CI logs): near-neutral, essentially no effect (rec_yards MAE delta -0.04, receptions -0.002). Don't re-test this exact feature the same way.

## What's still live/unconcluded

- **WR-R3 combined candidate** — real, correctly-scoped, implementation-ready design spec exists (`WR_R3_COMBINED_CANDIDATE_STATUS.md`), never built or run. The most concrete "actually go build this" candidate to come out of tonight, if you want one.
- QB first-down public-intent source crawl (`research-qb-first-down-public-intent-source-v1`/`v1b`) — manual/semi-automated media-text collection, paused mid-collection, not concluded either way. Unchanged from the first pass.
- Coverage v2 **player-level** WR-CB matchup data (`wr_cb_exposure.csv`) as an efficiency signal — genuinely never tried, but for a structural reason: `audit_wr_cb_source.py` already found nflverse has no reconstructable ground truth for who covered whom historically (`NO_GO_TRUE_ASSIGNMENT`). This data only exists live/current-slate via scrapers, so a frozen walk-forward diagnostic can't be built on it without new historical data acquisition. Not shovel-ready — flagging as a real gap, not a proposal.
- Worth knowing, not necessarily worth acting on: the repo's coverage-based `coverage_penalty()` WR efficiency multiplier (0.94x/1.04x YPT, in `scripts/modeling/rules_v2.py`) has been live since the repo's earliest commits and, per the finding above, has never been empirically validated — the one time this exact signal family was tested (the team-level ablation above) it came back null. Not urgent, but it's a piece of production logic resting on an untested assumption.

## Ranked next steps if you want to actually build one

1. **Build and run the WR-R3 combined candidate** per `WR_R3_COMBINED_CANDIDATE_STATUS.md`'s spec. Only genuinely live, unvalidated, ready-to-build candidate from the entire sweep.
2. **Decide on RB PD6**: someone needs to assemble genuine multi-season (not 2025-only) RB residual-calibration evidence and freeze a real replication plan before this lane can move at all. This is a data/scoping decision, not a modeling one — your call on priority.
3. Reconsider whether `coverage_penalty()` in `rules_v2.py` is worth empirically validating or retiring, given the one relevant test came back null.
4. Everything else in "fully exhausted" is closed; don't relitigate without genuinely new signal or new forward-season evidence, per each thread's own stop-rules.

## Bugs fixed this session (merged, non-model-science)

1. **RB P3 Week-1 gate crashed all of Full Slate from Week 2 on** (`.github/workflows/full-slate.yml`, `scripts/run_pricing_v2.py`, `scripts/run_pricing_with_full_roster_universe_v1.py`) — merged, PR #530.
2. **RB R26 (receptions) Week-1 gate, same crash class** (`scripts/run_pricing_with_full_roster_universe_v5_production.py`) — merged, PR #530.
3. **RB R22 (receiving-tail) Week-1 gate, same crash class across 3 files** (`v4_production.py`, `audit_market_model_lineage_v3.py`, `validate_certified_full_slate_stack_v3.py`) — merged, PR #530.
4. **Full Slate data-quality classifier hard-required exactly 32 scheduled teams** (`scripts/validate_full_slate_data_quality_v1.py`) — would have crashed the live-odds data-quality gate on every bye week from roughly Week 4 onward. Fixed to accept any even, nonzero scheduled-team count; fixed the same hardcoded-32 bug in the injury-scope ledger check right next to it. Merged, PR #531.

All fixes validated: full test suite (258 passed / 1 skipped) and both strict repo audits (`audit_repo.py --strict`, `audit_2026_production_readiness.py --strict`) clean at time of merge.

## Honest caveats

- I did not exhaustively re-read all 136 RB branches commit-by-commit in pass 1 — I used the project's own terminal synthesis docs as the primary source and spot-checked what postdates them. Pass 2 closed several of the resulting gaps but was itself targeted (RB PD-chain, WR-R3, Coverage v2), not exhaustive either.
- No new model code was run tonight anywhere. Every "recovered" result is real, already-executed CI output pulled from job logs — not new compute, and not re-verified independently by me beyond confirming run/job/artifact IDs and reading the printed JSON directly.
- The WR-R3 design spec (the one live proposal left standing) still needs to actually be built and pass its own frozen gates before it means anything. Nothing here is validated until it's run.
