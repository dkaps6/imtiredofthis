# WR-ND2 — Yards-Per-Target Mechanics Decomposition Plan

**Branch:** `research-wr-nd2-ypt-mechanics`

**Parent result:** `WR_ND1_POST_M38_RESIDUAL_DECOMPOSITION_RESULTS.md`

**Parent result commit:** `8380b831a8a211ff018388a1da190f2a0340df2c`

**Status:** frozen diagnostic plan; no production change.

## Why this study exists

WR-ND1 established, after the promoted M38 target-hierarchy transform, that `YARDS_PER_TARGET` is the largest receiving-yard MAE component overall and in WR1, WR2, and WR3. The frozen dominance gate passed:

- YPT Shapley MAE attribution: `9.566137`
- positive-attribution share: `43.44%`
- same top component in WR1/WR2/WR3: `3 / 3`
- disposition: `YARDS_PER_TARGET_DOMINANT`

WR-ND2 does not fit a model. It decomposes the current YPT representation into the two multiplicative football mechanics hidden inside it:

`YARDS_PER_TARGET = CATCH_RATE × YARDS_PER_RECEPTION`

The purpose is to determine whether the next predictive architecture should focus on conversion/catch probability, yards generated on completed receptions, or a joint efficiency process.

## Frozen population and information contract

- Evaluation season: 2025.
- Weeks: 1-18.
- Prior season: 2024.
- Same strict-prior historical bundle as WR-ND1.
- Same WR/LWR/RWR/SWR evaluation population.
- Same M38 hierarchy transform, frozen at `(1.40, 1.14, 0.91, 0.78)`.
- Same target-result anomaly handling as canonical WR-ND1: mathematically non-factorizable rows are excluded only from the target-game evaluation view; original historical logs remain intact for pregame priors.
- No sportsbook information.
- No fitting, tuning, threshold search, or production mutation.

## Frozen factorization

Receiving yards are represented as:

`TEAM_TARGET_VOLUME × WR_TARGET_MASS × WITHIN_WR_ALLOCATION × CATCH_RATE × YARDS_PER_RECEPTION`

Pregame/base factors:

- `TEAM_TARGET_VOLUME`, `WR_TARGET_MASS`, `WITHIN_WR_ALLOCATION`, and `CATCH_RATE` are exactly the WR-ND1 frozen pregame factors.
- base `YARDS_PER_RECEPTION` is derived mechanically as:

  `predicted YPT / predicted catch rate`

  and must reproduce the WR-ND1 predicted YPT within floating-point tolerance.

Target-game truth factors:

- actual team targets;
- actual WR target mass;
- actual within-WR player allocation;
- actual catch rate = receptions / targets when targets > 0;
- actual yards per reception = receiving yards / receptions when receptions > 0.

For rows with zero receptions, target-game YPR is set equal to the frozen pregame YPR because the corrected catch-rate term already drives the full prediction to zero. This avoids undefined 0/0 values without creating artificial YPR attribution.

## Exact Shapley decomposition

Calculate all `2^5 = 32` subsets for receiving yards across:

1. `TEAM_TARGET_VOLUME`
2. `WR_TARGET_MASS`
3. `WITHIN_WR_ALLOCATION`
4. `CATCH_RATE`
5. `YARDS_PER_RECEPTION`

Required identities:

- empty subset receiving-yard MAE must equal WR-ND1 post-M38 deterministic MAE within `1e-9` after the same anomaly filter;
- base catch rate × base YPR must reproduce base YPT row-by-row within `1e-10`;
- full subset must reproduce actual receiving yards row-by-row within `1e-8`;
- Shapley sum must equal empty-to-full MAE recovery within `1e-8`.

Canonical MC remains reference-only.

## Required slices

Report exact receiving-yard Shapley attribution for:

- ALL_WR;
- WR1, WR2, WR3, WR4_PLUS;
- base yard overprojection and underprojection;
- W1-4, W5-9, W10-13, W14-18;
- actual target tiers 0-3, 4-6, 7-9, 10+.

Also report descriptive base-vs-actual distributions for:

- catch rate;
- yards per reception;
- YPT;

by ALL_WR and WR role.

## Frozen efficiency routing gate

The routing decision is intentionally made **within the efficiency pair** identified by WR-ND1.

Let positive efficiency attribution be:

`max(CATCH_RATE, 0) + max(YARDS_PER_RECEPTION, 0)`.

Allowed dispositions:

- `YARDS_PER_RECEPTION_DOMINANT`
- `CATCH_RATE_DOMINANT`
- `MIXED_WR_EFFICIENCY_MECHANICS`

A component can be called dominant only if:

1. it is the larger of `CATCH_RATE` and `YARDS_PER_RECEPTION` overall;
2. it accounts for at least `60%` of positive efficiency Shapley attribution overall;
3. it is the larger efficiency component in at least two of WR1, WR2, and WR3.

Otherwise disposition is `MIXED_WR_EFFICIENCY_MECHANICS`.

No waivers.

## What each disposition means next

### `YARDS_PER_RECEPTION_DOMINANT`

Next investigate the completed-catch yardage process. First run a source-safe decomposition/audit of:

- completed-target air-yard contribution;
- yards after catch;
- explosive completed receptions;
- route/depth role persistence.

Do not immediately reuse M75's separation/cushion/aDOT/YACOE + PFR-secondary feature family. New information or a materially different representation is required.

### `CATCH_RATE_DOMINANT`

Next investigate target conversion mechanics, including target depth, QB accuracy/placement proxies, receiver conversion history, and coverage context only where timestamp-safe and historically deployable.

### `MIXED_WR_EFFICIENCY_MECHANICS`

Build a joint target-conversion + completed-catch yardage architecture rather than a one-factor patch.

## Preserved secondary lane

WR-ND1 also showed a separate opportunity-tail issue:

- 10+ target games: within-WR allocation attribution `22.597776`, larger than YPT `13.221916`;
- false-low receiving-yard games: within-WR allocation is the largest component.

WR-ND2 does not discard that finding. It remains a separate future dynamic-entitlement lane. This diagnostic only resolves the broad YPT/efficiency branch first because it passed the structural dominance gate overall and across WR1/WR2/WR3.

## Explicit anti-duplication rules

Do not use WR-ND2 to reopen:

- M31-M32 generic target-pool pruning;
- M36-M38 universal WR hierarchy multiplier tuning;
- M72 aggregate explosive-weapon × defense interactions;
- M75 NGS separation/cushion/aDOT/YACOE + PFR secondary aggregate interactions under another algorithm;
- fake WR-CB assignments from participation data;
- source-blocked M84 current-only WR-CB pages as historical truth;
- sportsbook inputs upstream.

## Required artifacts

- `wr_nd2_player_factors.csv`
- `wr_nd2_shapley_summary.csv`
- `wr_nd2_slice_shapley.csv`
- `wr_nd2_efficiency_descriptives.csv`
- `wr_nd2_factorization_anomalies.csv`
- `wr_nd2_summary.json`

No production file may be changed.
