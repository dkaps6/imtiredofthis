# M95T — Constrained Dual-Layer Stable-Workhorse Tail Candidate

## Purpose

M95T is the final retrospective stable-workhorse carry-tail candidate in M95. It implements the M95S decomposition rather than another broad feature search:

1. **Population mass** must react quickly and conservatively to the current pregame league workload regime.
2. **Player ranking** may use persistent feed/carry-ceiling history only as a bounded, conditional, mass-preserving reranker.

M95T cannot declare the RB rushing-yard problem solved. Carries are the opportunity engine, not the final prop target. Regardless of M95T pass/fail, the next RB phase is **M96 — Rushing-Yard Synthesis / Opportunity-to-Yardage Translation**, which must validate the selected workload architecture against rushing-yard point accuracy and yardage tails before RB is closed.

## Frozen candidate

### Baseline

- M94C central carries remain unchanged.
- M95F `p20` / `p25` remain the player-game tail backbone.
- Vacancy/transition remains the separate M95I regime; M95T changes stable workhorses only.
- Primary evaluation scope is stable workhorses, Weeks 13-18, seasons 2020-2025.

### Layer 1 — fast population-mass anchor

For each season-week, use only completed prior weeks from the broad league lead-RB workload census:

- `league_prior4_20`: mean lead-RB 20+ rate over the prior four completed NFL weeks;
- `league_std_20`: season-to-date lead-RB 20+ rate through the prior completed week.

Compute a relative current-regime ratio:

`regime_ratio = clip(league_prior4_20 / league_std_20, 0.70, 1.30)`

Then strongly shrink it toward no adjustment:

`mass_factor = 1 + 0.50 * (regime_ratio - 1)`

Thus the population layer can move M95F's weekly mean by at most +/-15% **relative**, not by an unrestricted cross-season log-odds residual. No team recent-workload variable may act as a blanket positive mass booster.

The target stable-workhorse 20+ mass is:

`weekly_target20 = clip(M95F_weekly_mean20 * mass_factor, 0.05, 0.70)`

The same mass factor is used only diagnostically for 25+.

### Layer 2 — conditional feed/carry-ceiling reranker

Reuse the exact leakage-safe M95K feed-feature constructor with the frozen `k=4` shrinkage semantics. No new feed feature is searched.

For stable workhorses within each season-week:

- `base_rank` = percentile rank of M95F p20;
- `feed_rank` = mean percentile rank of frozen `feed20_rate` and `carry_ceiling95`;
- an observation is **aligned** only when base and feed ranks are on the same side of the within-week median (both >=0.50 or both <0.50);
- discordant observations receive **zero** feed reranking;
- aligned observations receive bounded log-odds delta:

`rank_delta = 0.50 * (feed_rank - base_rank)`

Because aligned ranks occupy the same half of the distribution, the raw delta is naturally bounded; it is also hard-clipped to `[-0.25, +0.25]` log-odds.

First, reranked probabilities are mean-anchored back to the original M95F weekly stable-workhorse mean. This makes the player layer exactly mass-preserving. Only after that does Layer 1 move the weekly aggregate to the frozen population target.

No outcomes are used to choose thresholds, coefficients, windows, or caps.

### 25+ diagnostic

25+ is not an advancement target because event counts are sparse. The final p20/base-p20 relative shift is propagated conservatively to M95F p25, capped to `[0.70, 1.30]`, constrained `p25 <= p20`, and mean-anchored to the M95F weekly p25 mean times the same population mass factor.

## Historical inputs

- Exact stable-workhorse panel from authoritative M95Q/M95P/M95R traces.
- Durable nflverse player-weekly cache: `data/research_cache/nflverse_player_weekly/player_weekly_2018.parquet` through `player_weekly_2025.parquet`.
- Exact M95K leakage-safe feed constructor from `scripts/backtest/evaluate_rb_feed_tendency_carry_ceiling.py`.
- No historical source re-download is required.

The cached weekly source is normalized through the existing PlayerForm v2 canonical player/team semantics before feed features and rushing-yard truth are attached.

## Precommitted advancement gates — stable 20+

M95T advances only if **all** are true on the comparable W13-18 2020-2025 panel:

1. pooled Brier improves vs M95F;
2. pooled logloss improves vs M95F;
3. pooled AUC is no worse than `-0.01`;
4. at least 4 of 6 seasons have non-negative Brier gain (ties allowed within `1e-6`);
5. at least 4 of 6 seasons have non-negative logloss gain;
6. no season's Brier worsens by more than `0.0075`;
7. no season's logloss worsens by more than `0.020`;
8. no season's absolute calibration gap worsens by more than `0.025`;
9. 2023 and 2025 each pass the same material-regression guards;
10. player reranking is exactly mass-preserving before the population layer (numerical tolerance `1e-9`).

No gate may be changed after the run exposes results.

## Rushing-yard translation guard

M95T still does **not** alter M94C central carries or construct a new yardage point estimate, so rushing-yard MAE cannot be claimed improved here. However, the candidate must pass a downstream sanity guard before it can be considered a viable workload input to M96:

- attach actual rushing yards from the frozen cached weekly source;
- audit correlation between carries and rushing yards by season and pooled;
- compare M95F p20 vs M95T p20 as discriminators of `75+` and `100+` rushing-yard outcomes;
- pooled 75+ AUC and pooled 100+ AUC may not each regress by more than `0.01`;
- output season-level yardage-tail metrics and event counts for M96.

This guard only verifies that a carry-tail improvement is not obviously anti-informative for rushing-yard upside. **M96 remains mandatory** for point rushing-yard MAE/RMSE/correlation, efficiency/YPC synthesis, 75+/100+ calibration, and ordinary-game guards.

## Non-negotiable safeguards

- No sportsbook inputs.
- No production changes.
- No target-week/postgame inputs.
- No feature selection.
- No coefficient search.
- No hyperparameter search.
- No retuning M95K or M95R on exposed 2023/2025 outcomes.
- Preserve failed M95K/L/O/R evidence.

## Stopping rule

- **If M95T fails:** stop new retrospective RB carry-tail candidate development. Retain M94C + M95F for stable workhorses (with M95I separate vacancy diagnostics) and proceed to M96 rushing-yard synthesis using the conservative workload architecture.
- **If M95T passes:** freeze it immediately with no further retrospective tuning, proceed to M96 rushing-yard synthesis, and run M95T/M96 prospectively in 2026 shadow confirmation before any production promotion.
