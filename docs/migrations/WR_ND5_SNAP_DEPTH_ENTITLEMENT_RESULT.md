# WR-ND5 — Snap / Depth Target Entitlement Result

## Final disposition

**`NO_ACTIONABLE_SNAP_DEPTH_ENTITLEMENT_SIGNAL`**

No candidate cleared the frozen actionable gate. No production change is authorized.

## Canonical run

- Branch: `research-wr-nd5-snap-depth-entitlement`
- Canonical successful run: `34055534002`
- Job: `101546628738`
- Tested SHA: `d5f8d9c6dddfb1e0ad7bd76404c10ec4f05e4b1c`
- Exact M38 parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`
- Artifact: `9995958638`
- Artifact SHA256: `1025bf8027264c6e06cb9f7aeb12625d3f91c8d9a9b5071655008ae068b9b729`

Mechanical pre-result failures were implementation-only and did not change the frozen scientific protocol:

1. Original ND5 attempt `34054460026` failed on mixed datetime/NaN comparison during prior-depth leakage validation.
2. Corrected datetime wrapper attempt `34055114893` failed before evaluation because Python imported the exact-M38 parent `scripts.backtest` package instead of the research evaluator.
3. Canonical run `34055534002` explicitly loaded the local frozen evaluator and completed successfully. No signal definitions, thresholds, slices, gates, or outcomes were changed between attempts.

## Integrity / leakage checks

- M38 receiving-yard rows: `4647`
- M38 receiving-yard MC MAE: `17.099904733366`
- M38 RMSE: `25.196099510685915`
- M38 bias: `-5.238640833494836`
- M38 correlation: `0.5679458508349823`
- WR entitlement evaluation rows: `2130`
- Factorization anomalies excluded: `1` (known Isaiah Bond anomaly)
- Target reconstruction MAE: `2.076010432545868`
- Current depth timestamp violations: `0`
- Previous depth timestamp violations: `0`
- Postseason participation used: `false`
- Sportsbook inputs used: `false`
- Model fitting used: `false`
- Production changed: `false`

## Frozen gate

A candidate required ALL of:

- coverage >= `0.85`
- Spearman >= `+0.08`
- high-vs-low allocation-residual gap >= `+0.025`
- W2-18 gap > `0`
- W13-18 gap > `0`
- positive gap in at least 2 of WR1/WR2/WR3
- 10+ target / under-by-3 tail enrichment >= `1.20x`

No threshold was changed after results.

## Candidate results

### `SNAP_ACCEL_1V4`

- coverage: `0.8525821596244132`
- Spearman: `0.07454640697222922`
- allocation-residual gap: `0.020577690860850983`
- raw target-error gap: `0.31863532256926297`
- tail enrichment: `0.822964276928256x`
- W2-18 gap: `0.022496506965589753`
- W13-18 gap: `0.020032317768094672`
- WR1 gap: `0.017296556708429417`
- WR2 gap: `0.005154672051250315`
- WR3 gap: `-0.0030238332227070404`
- positive WR1/WR2/WR3 count: `2`
- frozen quartiles: low `-0.0675`, high `0.075`
- gate: **FAIL**

Interpretation: coherent but not actionable. It missed the Spearman and allocation-gap gates and did not enrich the frozen 10+/under-by-3 tail. Do not search adjacent acceleration windows after seeing this result.

### `SNAP_LEVEL_PRIOR1`

- coverage: `0.9605633802816902`
- Spearman: `0.07352987444942609`
- allocation-residual gap: `0.02134191499775379`
- raw target-error gap: `1.3963313764189815`
- tail enrichment: `2.1660325147007953x`
- W2-18 gap: `0.01892970212739541`
- W13-18 gap: `0.011874097119910047`
- WR1 gap: `0.08623416804525323`
- WR2 gap: `0.15248277338791122`
- WR3 gap: `0.06755287294125219`
- positive WR1/WR2/WR3 count: `3`
- frozen quartiles: low `0.36`, high `0.82`
- gate: **FAIL**

Interpretation: this is the strongest descriptive result in ND5. High prior-game snap participation strongly enriched the extreme false-low target tail and was directionally positive in every requested stability slice, especially WR1/WR2/WR3. However, it still missed both the frozen Spearman (`0.07353 < 0.08`) and allocation-gap (`0.02134 < 0.025`) gates. It therefore cannot advance or be promoted under the preregistered protocol. Do not lower the gates because the result is close.

### `DEPTH_TOP2_STATE`

- coverage: `0.9788732394366197`
- Spearman: `-0.06429318947538074`
- allocation-residual gap: `-0.021675262592151075`
- raw target-error gap: `0.2549146532019003`
- tail enrichment: `1.4284735543984652x`
- W2-18 gap: `-0.023883502954726014`
- W13-18 gap: `-0.04297408902363673`
- WR1 gap: `0.07953303853826618`
- WR2 gap: `0.0681470636254591`
- WR3 gap: `0.03785314837367889`
- positive WR1/WR2/WR3 count: `3`
- gate: **FAIL**

Interpretation: current top-two depth rank is not a globally valid residual-correction signal. The all-WR and late-season direction is opposite the preregistered hypothesis despite positive within-role slices. Do not use it as a generic M38 entitlement adjustment.

### `DEPTH_RANK_PROMOTION`

- coverage: `0.9061032863849765`
- Spearman: `0.029816690859347964`
- allocation-residual gap: `0.014785972625507262`
- raw target-error gap: `-0.16561465732937997`
- tail enrichment: `0.31025197926295056x`
- W2-18 gap: `0.014785972625507262`
- W13-18 gap: `0.023764808933688503`
- WR1 gap: `0.016594208009409986`
- WR2 gap: `-0.04568300827423176`
- WR3 gap: `0.0018684753332864207`
- positive WR1/WR2/WR3 count: `2`
- gate: **FAIL**

Interpretation: simple pregame depth promotion does not identify the unresolved false-low opportunity problem.

## Durable conclusion

ND4 successfully recovered two source-time-safe information families, but ND5 shows that **simple prior-game snap state and simple pregame depth rank are not strong enough, by themselves, to justify changing M38 target allocation**.

The most useful evidence is that high prior-game snap share contains a real descriptive relationship with severe target underprojection (`2.166x` tail enrichment; positive WR1/WR2/WR3 gaps), but the global monotonic strength and average allocation-residual separation remain below the preregistered standard.

Therefore:

- do not retune M38 hierarchy multipliers;
- do not lower ND5 gates;
- do not search nearby snap windows merely because `SNAP_LEVEL_PRIOR1` was close;
- do not combine snap and depth after seeing results; ND5 explicitly prohibited post-result combinations unless component families independently established evidence;
- do not use postseason-released participation upstream;
- do not change production from ND5.

The unresolved WR mean error remains opportunity-dominant, but the tested simple target-history, vacancy, snap, and depth entitlement families have now failed to produce a frozen actionable correction. A next branch must therefore introduce materially new information or move to the next independently supported error family rather than repackaging these signals.
