# Receiving Ecosystem + Pass-Yard Conservation Audit — Frozen Plan

## Purpose

Audit the entire passing/receiving ecosystem before adding another receiver-specific model candidate.

The current engine already gives RB/FB/TE receiving markets and allocates team targets jointly across WR/TE/RB/FB, but the research program has been WR-heavy. M38 changes only within-WR target shares, while TE and RB/FB target mass and within-position hierarchy have not received equivalent multi-season research. In addition, `simulation_v2` currently generates receiver receiving yards and QB passing yards through separate yardage draws; they share pass-volume/pass-efficiency context but are not forced to satisfy the football accounting identity that completed-pass yardage becomes receiver receiving yardage.

This audit is diagnostic only. It does not alter M89/M90 QB mean production, M38 WR production, RB P3 production, or any sportsbook-facing output.

## Lineage

- Branch: `audit-receiving-ecosystem-conservation`
- Parent branch: `audit-qb-wr-shared-pass-volume-mechanism`
- Parent SHA: `ba23da103e6e0e828496793c8c930363c2b50f28`
- Prior QB-WR residual coupling disposition: `STRONG_QB_WR_ERROR_COUPLING`
- Prior shared-pass-volume disposition: `STRONG_SHARED_PASS_VOLUME_MECHANISM`
- Sportsbook inputs: **0**
- Model fitting: **0**
- Production change: **0**

## Frozen questions

1. How much of team target, reception, and receiving-yard mass belongs pregame to WRs, TEs, and RB/FBs?
2. Is the current model systematically moving receiving opportunity between position groups incorrectly even when total pass volume is reasonable?
3. What is the current leakage-safe baseline accuracy for RB receiving yards/receptions and RB rush+receiving yards?
4. What is the current leakage-safe baseline accuracy for TE targets/receptions/receiving yards?
5. How large is the current model's QB-passing-yards versus aggregate-receiver-yards conservation gap?
6. Do QB residuals remain coupled to aggregate receiving residuals after WR, TE, and RB/FB are accounted for together rather than treating WRs as the receiving game?

## Frozen scope

### Receiving ecosystem
- Seasons: **2020-2025**, regular season only.
- 2020: Weeks 1-17.
- 2021-2025: Weeks 1-18.
- Walk-forward pregame context only.
- Position groups:
  - WR = `WR/LWR/RWR/SWR`
  - TE = `TE`
  - RB receiving = `RB`
  - FB = `FB`; combined with RB for team position-mass diagnostics but reported separately where sample permits.
- Any actual receiver outside those modeled groups is reported as `OTHER`; it may not be silently discarded from accounting checks.

### QB/receiver projection alignment
- Primary projected-QB comparison: **2024-2025**, the exact common era already established by the M89/M38 QB-WR coupling work.
- Earlier seasons may be reported only if the exact comparable QB projection path is available without changing model semantics.
- No convenient-era substitution after results.

## Frozen current architecture to audit

Do not retune it.

- Team pass opportunity is generated from pregame plays/pass-rate context.
- Targets are jointly allocated across the modeled receiving pool plus residual target mass.
- M38 WR sharpening changes only WR shares and preserves total WR target mass.
- TE shares remain structurally present but are not M38-sharpened.
- RB/FB receiving shares remain structurally present but are not M38-sharpened.
- `rules_v2` may apply pre-existing TE and RB receiving matchup multipliers; these are audited as-is.
- QB passing yards are generated separately from aggregate receiver receiving-yard draws in the current simulation and therefore are not mechanically conserved.

## Frozen outputs

### A. Position-group opportunity ledger
For every team-game and season, report actual and projected:
- targets / target share,
- receptions / reception share,
- receiving yards / receiving-yard share,
for WR, TE, RB, FB, RB+FB, OTHER, and all modeled receivers combined.

Report pooled and season-level bias, MAE, RMSE, correlation, and share error.

### B. Cross-position displacement diagnostics
Within team-game, calculate residual coupling for:
- WR target-mass error vs TE target-mass error,
- WR target-mass error vs RB+FB target-mass error,
- TE target-mass error vs RB+FB target-mass error,
with Pearson, Spearman, same-sign rate, and quartile residual separation.

Repeat for receptions and receiving yards.

### C. RB receiving baseline
RB-only, pooled and by season:
- target error,
- receptions MAE/RMSE/bias/correlation,
- receiving-yards MAE/RMSE/bias/correlation,
- rushing-yards baseline where available,
- **rush + receiving yards** MAE/RMSE/bias/correlation.

This establishes the baseline for a dedicated RB receiving lane; it is not a promotion test.

### D. TE baseline
TE-only, pooled and by season:
- target error,
- receptions MAE/RMSE/bias/correlation,
- receiving-yards MAE/RMSE/bias/correlation,
- target/reception/yard share error by team-game.

This establishes the baseline for a dedicated TE lane using the WR research process but TE-specific priors, roles, coverage mechanisms, and gates.

### E. Football accounting / conservation
First validate the historical source identity using **all passers and all receivers**:

`actual team passing yards ≈ sum(actual receiving yards)`

Then audit the current model at the projection/sample level:

`projected primary-QB passing yards` vs `sum(projected receiving yards across modeled receivers)`

Report mean signed gap, MAE, median absolute gap, p90/p95 absolute gap, correlation, season slices, and the percentage of team-games exceeding 10/25/50 yards.

Where per-iteration receiver samples are recoverable from the canonical MC, calculate the conservation gap iteration-by-iteration rather than only comparing means.

### F. Full receiving ecosystem versus QB error
For aligned 2024-2025 team-games, compare QB passing-yard residual against:
- total modeled receiving-yard residual,
- WR receiving-yard residual,
- TE receiving-yard residual,
- RB+FB receiving-yard residual.

This is diagnostic only; realized receiver outcomes remain forbidden as QB inputs.

## Frozen integrity gates

1. Historical accounting check must have all-receiver receiving-yards vs all-passer passing-yards **MAE <= 1.0 yard** on aligned team-games. If not, stop scientific interpretation and diagnose source/key/stat semantics.
2. No target-game outcomes may enter any pregame projection or feature.
3. No sportsbook data may enter this audit.
4. Exact target-season position labels must come from the leakage-safe pregame universe where required; any fallback must be documented and cannot use future depth/roster state.

## Frozen scientific gates / dispositions

### Position-group mass misallocation
`POSITION_GROUP_MISALLOCATION_SUPPORTED` if either condition is met:
- TE or RB+FB pooled target-share bias has absolute magnitude **>= 1.5 percentage points** and the same bias sign in at least **4 of 6 seasons**; or
- WR target-mass residual versus TE or RB+FB target-mass residual has Pearson **<= -0.15** with the same negative direction in at least **4 of 6 seasons**.

Otherwise: `NO_MATERIAL_POSITION_GROUP_MASS_MISALLOCATION`.

### QB/receiver conservation
`MATERIAL_PASS_RECEIVING_CONSERVATION_GAP` if, over the frozen 2024-2025 aligned projection sample, current-model absolute QB-versus-receiver-sum gap has either:
- median **>= 10 yards**, or
- p90 **>= 25 yards**.

Otherwise: `CONSERVATION_GAP_SMALL_AT_CURRENT_RESOLUTION`.

These thresholds are frozen before the audit is run and may not be relaxed after results.

## Authorized follow-up logic

- RB receiving and TE baseline research are authorized as distinct lanes regardless of whether position-group mass bias passes, because both are player statistics we intend to project independently. This audit determines **where** their corrections should enter, not whether those markets matter.
- If position-group mass misallocation passes, the next candidate must calibrate WR/TE/RB receiving **group mass before within-position hierarchy**. It may not repair WRs by stealing arbitrary mass from TE/RB after the fact.
- If position-group mass misallocation fails, retain existing group mass and research TE/RB within-position allocation/efficiency independently.
- If conservation gap is material, the next joint-MC architecture must test a single completed-pass/yardage process whose receiving-yard allocation sums to team passing yardage within each simulation iteration. M89/M90 remains the frozen QB mean anchor unless a separately frozen full-stack test proves a replacement is better.
- A TE model may reuse the WR research **process**, not WR coefficients. TE role, target hierarchy, catch conversion, YPR/YAC, coverage, and tail behavior must be independently calibrated.
- RB receiving must remain separate from RB rushing mechanics until the final joint player distribution; final RB outputs must include rushing yards, receiving yards, receptions, and rush+receiving yards.

## Stopping rule

This audit is not a feature hunt. It produces the frozen ledgers and dispositions above. No new coefficients, interaction search, threshold search, or market comparison is permitted inside this migration.