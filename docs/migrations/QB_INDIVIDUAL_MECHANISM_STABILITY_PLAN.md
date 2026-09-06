# QB Individual Mechanism Stability Audit

## Status

`PREREGISTERED / DIAGNOSTIC ONLY`

## Why this exists

The QB individual-mechanism decomposition mapped 884 canonical 2024-2025 QB games into attempt-volume, YPA-efficiency, stack-adjustment, and synthesis-adjustment contributions. Many high-error QBs are attempt-dominant while others are YPA-dominant.

Before using those individual profiles to guide future football research, we need to know whether a QB's mechanism profile is reasonably persistent or whether it changes so much season-to-season that game context must dominate the next research layer.

This audit does not search for new features and does not reopen broad QB research closed after M89/M90.

## Frozen input

Canonical M89 QB evidence from run `33331073376`.

Reconstruct the identical 884-game component casebook used by the prior individual-mechanism decomposition.

No sportsbook data. No model fitting. No production changes.

## Frozen per-season profile

For each QB-season with at least `6` canonical games, calculate:

- mean absolute attempt component
- mean absolute YPA component
- mean absolute stack adjustment
- mean absolute synthesis adjustment
- component shares of their summed mean-absolute contribution
- passing-yard MAE and bias
- attempt residual mean and MAE
- YPA residual mean and MAE
- dominant component = component with largest mean-absolute contribution

## Frozen cross-season cohort

A QB qualifies for stability analysis only if:

- at least `6` games in 2024;
- at least `6` games in 2025;
- at least `16` games combined across the two seasons.

## Frozen stability metrics

Report:

1. fraction with the same dominant component in 2024 and 2025;
2. Pearson/Spearman cross-season correlation of attempt-component share;
3. Pearson/Spearman cross-season correlation of YPA-component share;
4. Pearson/Spearman cross-season correlation of passing-yard MAE;
5. sign persistence of attempt residual bias;
6. sign persistence of YPA residual bias.

## Frozen disposition gate

Call `QB_PLAYER_MECHANISMS_SHOW_USEFUL_STABILITY` only if all are true:

- qualifying QBs >= `18`;
- same dominant-component rate >= `0.55`;
- attempt-share Spearman >= `0.30`;
- YPA-share Spearman >= `0.30`;
- at least one of attempt-bias sign persistence or YPA-bias sign persistence >= `0.60`.

Otherwise call `QB_PLAYER_MECHANISMS_REQUIRE_CONTEXT_REGIMES`.

Thresholds may not be changed after results.

## Interpretation

A pass does not authorize player-specific constants. It means player identity/mechanism archetype contains enough stable structure that future pregame research may condition candidate football mechanisms on the QB's dominant error family.

A fail means the decomposition is still useful diagnostically, but the next research layer must emphasize game-level context/regime state rather than treating a QB as having one persistent correction type.

## Relationship to cross-position work

The QB-WR shared pass-volume bridge is a separate audit. It cannot determine this result and cannot change the QB model directly.

## Production rule

`production_changed = false`

`sportsbook_inputs_used = false`