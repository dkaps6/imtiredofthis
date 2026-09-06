# WR-ND3 — Dynamic Target Entitlement Diagnostic

## Status

Frozen research protocol. Diagnostic only. No production change is authorized by this document.

## Lineage

- Exact M38 merge parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- M38 WR hierarchy multipliers remain frozen at `(1.40, 1.14, 0.91, 0.78)`.
- WR-ND1 result: overall post-M38 receiving-yard residual was YPT-dominant, while within-WR allocation dominated the false-low / 10+ target tail.
- WR-ND2 result: `MIXED_WR_EFFICIENCY_MECHANICS`; yards per reception led catch rate but failed the frozen 60% dominance gate.
- Strict-prior four-way post-M38 result: run `34044780465`, disposition `OPPORTUNITY_DOMINANT`; opportunity explained ~41.8% of total WR yardage headroom while explosive yardage dominated large underprediction / 100+ yard tails.
- Mechanical anomaly carried forward for comparability: Isaiah Bond, CLE, 2025 W16 (0 recorded targets, +21 receiving yards) is excluded from the target-result evaluation population only when matching the ND1/strict-prior casebook. Historical logs remain untouched for all prior construction.

## Research question

After M38 fixed the static WR hierarchy, can strictly pregame dynamic information identify when a WR deserves materially more or fewer targets than M38 currently allocates?

This is specifically a **within-WR entitlement** question. It is not a new generic target-pool calibration and it is not permission to retune the M38 rank multipliers.

## Why this is next

The completed diagnostics now agree on the following structure:

1. Broad WR mean error still contains more opportunity error than any single efficiency subcomponent.
2. ND1 localized the most important opportunity weakness to within-WR allocation, especially false-low and 10+ target games.
3. Explosive yardage is a separate ceiling/tail lane and must not be forced into the mean target model.

Therefore the next test is whether **dynamic entitlement state** explains residual target allocation beyond M38's static pregame hierarchy.

## Exact parent reconstruction

Every canonical run must separately check out exact M38 commit `b98518d97b3038f471aee9ae3201009b2c70bb29`, rebuild 2025 Weeks 1-18 historical inputs using 2024 as the prior season, and verify the canonical M38 receiving-yard reference:

- `n = 4647`
- rec-yards MC MAE = `17.099904733366`
- RMSE = `25.196099510686`
- bias = `-5.238640833495`
- correlation = `0.567945850835`

Target-opportunity reconstruction must also remain consistent with the known post-M38 deterministic target MAE of approximately `2.076010` on the ND1-comparable WR casebook. A deviation greater than `0.01` is an integrity failure, not a scientific result.

## Leakage boundary

For target week `W`, every dynamic signal may use only:

- player/team game logs from weeks strictly before `W` (or prior season for Week 1),
- the frozen target-week **pregame universe** for current roster membership,
- target-week pregame role/depth metadata only where it already exists in the historical universe,
- existing target-week pregame injury state already admitted by the canonical historical pipeline.

Target-week targets, receptions, receiving yards, or any future row may be used only after prediction for evaluation.

Sportsbook/prop/market information is prohibited upstream.

## Source constraint discovered before launch

Historical weekly route participation is not populated reliably in the canonical player logs (`routes`, `route_rate_game`, and `yprr_game` are generally unavailable). Therefore route participation / routes-run trajectory is **source-blocked for WR-ND3** and may not be silently imputed or reconstructed from target-week outcomes.

If a timestamp-safe historical route source is added later, that constitutes genuinely new information and may justify a separate future branch.

## Frozen candidate signal families

### A. `RECENCY_ACCEL_2V8`

Player-specific recent target-share acceleration:

`mean target share over last 2 prior same-team games - mean target share over last 8 prior same-team games`

- Same-team history only.
- Crosses the season boundary naturally for Week 1.
- This tests trajectory/role acceleration rather than repackaging the current-season mean already used by the baseline.

### B. `HIGHER_WR_ABSENT_COUNT`

For each current WR, count WR teammates from the team's prior four games who:

1. had a larger prior-four target share than the current WR, and
2. are absent from the target-week pregame universe.

This measures hierarchy uplift caused by a previously higher-usage WR disappearing.

### C. `VACATED_SHARE_ABOVE_PLAYER`

For each current WR, sum prior-four team target share belonging to absent eligible pass-catchers (WR/TE/RB/FB) whose prior-four share exceeded the current WR's prior-four share.

This is a magnitude-sensitive version of dynamic opportunity opening. It is player-specific; raw team vacated share alone is only a context/control variable and cannot establish within-WR entitlement.

## Frozen controls / audits

The diagnostic must also report, but may not route on:

- total absent pass-catcher target share from the prior four team games,
- absent WR target share from the prior four team games,
- recent raw target-count acceleration (last 2 vs last 8),
- existing canonical `rules_injury_redistribution` flag,
- target-week pregame `role` field coverage,
- Week 1 vs Weeks 2-18 vs Weeks 13-18,
- WR1 / WR2 / WR3 / WR4+ under M38.

These controls exist to distinguish a real dynamic entitlement signal from team-volume changes or the already-existing legacy alpha-vacancy rule.

## Evaluation outcomes

Primary outcome:

`allocation_residual = actual within-WR target share - M38 predicted within-WR target share`

Secondary outcome:

`raw_target_error = actual targets - M38 predicted targets`

Frozen high-entitlement miss tail:

- actual targets >= 10, and
- raw target underprediction >= 3 targets.

This tail was frozen from the already-established ND1 finding that 10+ target games remain the unresolved dynamic entitlement/ceiling lane.

## Signal scoring

For each candidate signal:

1. coverage rate,
2. Spearman correlation with `allocation_residual`,
3. high-vs-low allocation residual gap,
4. enrichment of the frozen 10+ / under-by-3 target tail,
5. directional gap in W2-18,
6. directional gap in W13-18,
7. directional gap separately for WR1, WR2, WR3.

High/low definitions are frozen by signal family and do not use outcomes:

- `RECENCY_ACCEL_2V8`: top quartile vs bottom quartile of available signal values.
- `HIGHER_WR_ABSENT_COUNT`: value > 0 vs value = 0.
- `VACATED_SHARE_ABOVE_PLAYER`: value > 0 vs value = 0.

## Frozen actionable gate

A candidate advances only if **all** are true:

- coverage >= 70%,
- overall Spearman >= +0.08,
- high-vs-low allocation-residual gap >= +0.025,
- W2-18 gap > 0,
- W13-18 gap > 0,
- positive gap in at least 2 of WR1/WR2/WR3,
- frozen 10+ / under-by-3 tail enrichment >= 1.20x.

No threshold may be lowered after results are visible.

## Frozen disposition

- exactly one candidate passes: `<SIGNAL>_ACTIONABLE`
- two or more candidates pass: `MULTIPLE_DYNAMIC_ENTITLEMENT_SIGNALS`
- no candidate passes: `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`

A diagnostic pass does **not** authorize production. It only authorizes a subsequent frozen predictive branch using the winning information family.

## Anti-duplication / prohibitions

Do not:

- retune M38's `1.40 / 1.14 / 0.91 / 0.78` multipliers,
- rerun M31-M32 generic target-pool pruning,
- treat current-season mean target share as a new recency signal,
- repackage the existing alpha-vacancy rule as new evidence,
- invent route participation where historical routes are unavailable,
- infer fake WR-CB assignments from participation,
- rerun M75 NGS separation/cushion/aDOT/YACOE + PFR aggregate interactions under another algorithm,
- use sportsbook/player-prop data upstream,
- fit a supervised model in this diagnostic,
- combine candidate signals after seeing which ones look favorable.

## Separate QB/explosive research note

The user hypothesis that improved WR ceiling characterization may explain QB explosive passing games is preserved as a **downstream research lane**, not part of WR-ND3.

After WR mean entitlement and WR explosive-tail signals are independently validated, a future QB diagnostic may test whether pregame aggregate receiver ceiling strength / explosive-play probability explains QB upper-tail misses. That future test must be frozen separately and cannot use actual target-week WR explosions as QB inputs.
