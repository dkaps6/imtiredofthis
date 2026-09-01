# M96B — RB Modular Joint Workload × Efficiency Synthesis

## Why this migration exists

M96A showed that remaining rushing-yard error is joint: perfect opportunity recovered 7.68 MAE yards, perfect efficiency recovered 6.73, and the dominant source changes by workload regime. M95 also produced several real but narrow positive signals that failed when treated as universal replacements.

M96B therefore changes the research question from **"which one model replaces M94C?"** to:

> Which already-demonstrated capabilities can coexist as modules, each owning only the job it proved it can improve, without damaging the jobs owned by other modules?

This modular / puzzle principle is now part of the research contract. A component is not discarded merely because it fails as a universal replacement. It may survive if its positive capability can be isolated, added incrementally, and shown not to create unacceptable collateral regression.

## Capability ledger — frozen before M96B results

| Module | Source | Proven / retained capability | Explicitly NOT allowed to do |
|---|---|---|---|
| **C — central opportunity** | M94C | Strongest conservative central carry anchor / ordinary workload behavior | Must not be globally inflated to chase 20+/25+ outcomes |
| **W — workload tail** | M95F | Calibrated 20+/25+ workload-state probability and upper workload distribution | Must not replace C with `m95f_mix_mean` as the universal point workload |
| **V — vacancy / transition** | M95I | Strong vacancy/role-transition tail ranking evidence | Must not be applied to stable incumbents; deterministic M95I carry transform is not globally promoted |
| **E — mean efficiency / environment** | M95C exact environment control carried through M95D | Blocking/offensive-environment signal improved mean rushing-yard prediction in both 2024 and 2025 relative to its role baseline | Must not change carries; must not inherit M95D tail/context changes into the point mean |
| **X — explosive/upside context** | M95D full environment/matchup layer | Improved 100+ rushing-yard discrimination in both 2024 and 2025 while mean projection was not better | Must not universally raise point YPC/yards; tail/ranking use only |

## Frozen source artifacts

- M94C run `33353485070`, artifact `migration-94c-rb-game-environment`
- M95F run `33389924330`, artifact `migration-95f-rb-workload-regime-calibration`
- M95I run `33402566592`, artifact `migration-95i-rb-deep-concentration-tail`
- M95D run `33359898917`, artifact `migration-95d-rb-rushing-environment-scheme`
  - `role_plus_m95c_environment` is the exact M95C environment control.
  - `full_environment_matchup` supplies X-tail context.

No sportsbook inputs. No production changes.

## Evaluation universes

### 2024 temporal calibration / module support
Use the exact 2024 W13-18 intersection of:
- M95D out-of-sample prediction trace, and
- M95F 2024 holdout trace.

This window is used only to fit the one-dimensional probability calibrators described below and to document the already-frozen E/X module behavior. It is not pristine confirmation.

### 2025 primary evaluation
Use the exact intersection of:
- M94C 2025 RB trace,
- M95D 2025 out-of-sample prediction trace, and
- M95F 2025 trace.

Fail mechanically if coverage is <97% of the M95D 2025 OOS universe. M95I is joined separately and may be missing only for rows outside its exact trace semantics.

Report both full-2025 and W13-18 comparable-window results.

## Point-yard synthesis — C and E only

The point center is protected. W, V and X are prohibited from changing the deterministic point prediction in M96B.

### C
Exact frozen M94C point rushing yards:

`C = candidate_rush_yards`

### E residual module
M95D contains an exact frozen M95C environment arm and its role-only baseline on the same OOS rows.

Define the already-frozen environment-only residual:

`E_delta = pred(role_plus_m95c_environment, rush_yards) - pred(role_baseline, rush_yards)`

Then:

`CE = C + E_delta`

There is **no shrink search, coefficient search, clipping search or fitted weight**. The direct residual is the single precommitted E test.

E earns retention as a point module only if, on 2025:
- all-RB MAE improves versus C;
- absolute bias does not worsen by >1.0 yard;
- none of the ordinary workload slices 0-5, 6-10, 11-14, 15-19 worsens MAE by >1.0 yard;
- 20+/25+ slices are reported but are not allowed to override ordinary-game damage.

If E fails, C remains the point center. Tail modules are still evaluated independently.

## Tail-module synthesis — modular ablation, no target-driven feature search

The tail question is 75+ and 100+ rushing yards.

### Base score B
Use the percentile rank of the chosen frozen point anchor within each season:
- `CE` if E passes its precommitted point gate;
- otherwise `C`.

This conditional fallback is predeclared here; it is not a post-result invention.

### W score
Workload-tail score is the equal-weight percentile-rank average of M95F `m95f_p90` and `m95f_p95`.

W is distribution information only. It does not modify the point prediction.

### X score
X is the **incremental** M95D upside context, isolated from the mean environment module:

`X_delta = pred(full_environment_matchup, rush_yards) - pred(role_plus_m95c_environment, rush_yards)`

Use the percentile rank of `X_delta`. This prevents the M95C environment component from being counted twice.

### V score
V is 2025 vacancy-only diagnostic evidence because there is no like-for-like frozen 2024 M95I trace for temporal calibration.

For rows where `prior_top1_unavailable == 1`:
- 75+ diagnostic V score uses `p20_joint`;
- 100+ diagnostic V score uses `p25_joint`.

V never alters incumbent rows and cannot be promoted from M96B alone; it can only be retained as a prospective/shadow candidate.

## Frozen ablation arms

For 75+ and 100+ separately:

1. `B` — point-anchor rank only.
2. `B+W` — equal mean of B and W percentile ranks.
3. `B+X` — equal mean of B and X percentile ranks.
4. `B+W+X` — equal mean of B, W and X percentile ranks.
5. `B+W+X+V` — 2025 vacancy diagnostic only; for vacancy rows replace the W component with equal mean(W, V), leaving incumbent rows identical to arm 4.

No other feature combinations are permitted in M96B. Equal weighting is fixed. There is no search over weights.

## Probability calibration

For arms 1-4 only:
- fit a one-dimensional Platt logistic calibrator on the 2024 W13-18 common universe;
- fit separate calibrators for 75+ and 100+;
- apply frozen calibrators to 2025 full season and 2025 W13-18;
- no C search, feature search or alternate calibration family search.

Arm 5/V is ranking-only diagnostic in 2025 because no predeclared temporally prior V calibration set exists.

## Tail metrics

For every eligible arm / target:
- event count and base rate
- AUC
- Brier score
- log loss
- mean predicted probability and calibration gap

Also report point-anchor 75+/100+ AUC for continuity with M96A.

### Incremental-retention gates
A module is not retained merely because one number rises.

For W or X to earn modular retention:
- full-2025 AUC must improve by at least 0.005 **or** Brier must improve by at least 0.001 versus its parent arm;
- neither AUC may regress by >0.005 nor Brier worsen by >0.001 on the other primary tail target;
- 2025 W13-18 may not show material reversal: AUC regression >0.02 or Brier regression >0.003.

For the combined `B+W+X` arm to be the preferred tail composition:
- it must be non-inferior to the best single-module arm within the same regression tolerances on both 75+ and 100+;
- and it must materially improve at least one primary tail metric versus B.

If W and X each help different targets but their combination is destructive, retain them as target-specific modules rather than forcing a single universal fusion.

V can only be labeled `RETAIN_VACANCY_DIAGNOSTIC_FOR_PROSPECTIVE_CONFIRMATION` if its 2025 vacancy AUC improves versus the non-V arm for its mapped target; it cannot be production-promoted here.

## Modular non-degradation principle

M96B is explicitly allowed to keep multiple modules if they improve different jobs without damaging one another. Examples:
- E can own the point mean while W/X own tail probabilities.
- W may own workload-driven high-yardage probability while X owns explosive-efficiency upside.
- V may replace only the vacancy workload-tail submodule while incumbents continue using W.

Conversely, a positive standalone module is rejected from a combination if its addition destroys another validated capability beyond the frozen tolerances.

## Output / documentation requirements

Export:
- source/join audit
- module capability ledger
- 2024 temporal calibration metrics
- 2025 point ablation by workload slice
- 2025 tail ablation full and W13-18
- V vacancy diagnostic
- casebook of largest module disagreements
- disposition table showing each module as RETAIN / REJECT / DIAGNOSTIC

The canonical handoff must record:
- exact run/job/SHA/artifact/digest;
- every retained and rejected module;
- why it was retained/rejected;
- the surviving modular architecture;
- exact next migration and confirmation status.

## Integrity rules

- Research only.
- Sportsbook inputs = 0.
- No production change.
- No random split.
- No manual player/game choice.
- No weight search.
- No feature-combination search outside the five frozen arms.
- 2024/2025 are already inspected research data; no M96B result is pristine confirmation.
- Any survivor must still face genuinely prospective/untouched 2026 confirmation before production promotion.
