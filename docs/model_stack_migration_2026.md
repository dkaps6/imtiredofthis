# 2026 Model Stack Migration

## Goal

Preserve the empirically tested football logic built during the 2025 season while consolidating the repo around one production projection architecture.

The production target is:

1. PlayerForm / TeamForm establish pregame baseline information.
2. Bayesian shrinkage and future ML components estimate stable player expectations.
3. A canonical football rules layer applies tested matchup, role, injury, pressure, coverage, box-count, game-script and weather context.
4. `simulation_v2.py` remains the final joint game Monte Carlo distribution engine.
5. Sportsbook prices are compared only after the football projection distribution exists.
6. Model components earn their weight through leakage-free walk-forward backtesting.

## Current production path

`Full Slate -> metrics_v2 -> simulation_v2 -> run_pricing_v2`

This is the only production path until later migration phases deliberately wire additional components.

## Legacy component disposition

| Component | Disposition | Reason |
| --- | --- | --- |
| `simulation_v2.py` | KEEP / PRODUCTION | True joint game Monte Carlo and current pricing foundation. |
| `run_pricing_v2.py` | KEEP / PRODUCTION | Converts simulation outcomes into projections/probabilities. |
| `scripts/models/monte_carlo.py` | SALVAGE RULES | Parametric normal model, not a true Monte Carlo. Preserve script/funnel ideas, then retire duplicate engine. |
| `scripts/models/bayes_hier.py` | REWRITE | Current implementation is a normal-CDF adapter, not hierarchical Bayesian inference. |
| `scripts/models/markov.py` | PARK / REEVALUATE | Current implementation is not a Markov chain and can fall back to 0.5. |
| `scripts/models/ml_ensemble.py` | REWRITE | Adapter for `p_ml`; no trained ML model exists in the module. |
| `scripts/models/ensemble.py` | REBUILD | Equal 25% voting and 65/35 model-market blending must be learned/validated, not assumed. |
| `scripts/models/elite_rules.py` | PRESERVE / MERGE | Contains empirically developed 2025 football rules. |
| `scripts/models/agent_based.py` | MERGE | Small set of matchup heuristics overlaps the rulebook; no actual agent system. |
| `scripts/model/rules_engine.py` | PRESERVE / REWORK | Strong rulebook scaffold; never wired to production and contains schema/pressure semantics drift. |
| `scripts/models/drl_allocator.py` | PARK | Placeholder; bankroll allocation is downstream of projection quality. |
| `scripts/models/run_predictors.py` | LEGACY REFERENCE | Valuable logic/SGP ideas but tied to old artifacts and old ensemble. |
| `scripts/models/run_full_model.py` | RETIRE AFTER PARITY | Incomplete bridge into stale legacy stack. |
| `scripts/models/model_stack_patch.py` | RETIRE AFTER PARITY | Imports/functions and metrics path do not match current modules. |
| `scripts/models/team_script_features.py` | SALVAGE | Useful concept; migrate to canonical `TeamContext`. |
| `shared_types.py`, `types.py`, `result.py` | CONSOLIDATE | Multiple overlapping model contracts create schema drift. |

## Migration phases

### Phase 1 — Foundation (this PR)

- Add one canonical modeling package and model contracts.
- Consolidate the tested football heuristics into `rules_v2.py` without changing Full Slate behavior.
- Correct pressure matchup semantics: opponent defensive pressure generation is compared with offensive pressure allowed.
- Preserve the tested 2025 rule thresholds/multipliers explicitly until backtesting recalibrates them.
- Add regression tests.
- Delete nothing yet. Preservation/parity comes before removal.

### Phase 2 — Team/player context adapters

- Build canonical `TeamContext` directly from current `team_form.csv`.
- Build canonical player role/context from PlayerForm + Ourlads + Coverage v2.
- Remove duplicate TeamScript dataclasses after callers migrate.

### Phase 3 — Rules integration

- Apply rules to football assumptions before simulation rather than treating rules as a separate voting model.
- Feed projected plays/pass/rush split, coverage/role multipliers, box-count effects, injury redistribution and volatility modifiers into `simulation_v2`.
- Add provenance columns so each projection records which adjustments were active.

### Phase 4 — 2025 walk-forward backtest

- Reconstruct every regular-season week using only information available before kickoff.
- Compare baseline vs each rule family vs final simulation.
- Keep the tested 2025 values as the starting parameters, then measure whether recalibration improves out-of-sample performance.

### Phase 5 — Bayesian layer

- Replace `bayes_hier.py` with real shrinkage/posterior logic for player usage and efficiency.
- Weight prior season, career/role priors and current-season observations based on sample size.
- Validate especially Weeks 1-4 where shrinkage should matter most.

### Phase 6 — ML layer

- Train only after a leakage-safe historical feature table exists.
- Keep ML predictions independent of sportsbook price for projection-quality evaluation.
- Compare ML against baseline/Bayesian/simulation before including it in production.

### Phase 7 — Evidence-weighted ensemble and cleanup

- If multiple independent models add value, learn weights by market/position from walk-forward results.
- Market probability remains a separate benchmark/decision signal rather than silently becoming part of the football projection.
- Remove superseded legacy files only after tests demonstrate parity or documented replacement.

## Non-negotiable backtest rule

A Week N prediction may use only data that would have existed before the Week N game. No season-final aggregates, future weeks, later injury knowledge or later depth/usage outcomes may enter the feature set.
