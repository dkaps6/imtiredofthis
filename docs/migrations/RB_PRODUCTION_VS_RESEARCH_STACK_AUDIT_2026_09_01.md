# RB Production vs Research Stack Audit — 2026-09-01

## Why this audit exists

The canonical production `full-slate.yml` contains a richer model/rule/context stack than the frozen M94C RB research point that has been used as the conservative central reference. This distinction must remain explicit so future RB work does not accidentally treat M94C as equivalent to the complete production projection engine.

## Canonical production full-slate path

`main` remains at `7532a2c29dde78a5c3758eb1427561cfed801d67`. The only canonical production orchestrator is `.github/workflows/full-slate.yml`.

Before pricing, the production slate currently builds or validates:

1. current Ourlads roles/depth data (`roles_ourlads.csv`),
2. authoritative team-week mapping,
3. TeamForm and preseason-prior bridge,
4. promoted QB context,
5. weather and injuries,
6. Coverage v2 matchup intelligence,
7. optional current-season PBP enrichment,
8. PlayerForm v2 with stable identities,
9. canonical Team Context v3/provider provenance,
10. empirical Bayesian baseline,
11. supervised ML v2,
12. Markov State v2,
13. empirical football rule layer,
14. evidence-weighted ensemble framework,
15. joint Monte Carlo simulation,
16. promoted QB M89/M90 synthesis for passing yards,
17. sportsbook comparison only after the independent football projection is formed.

Production pricing explicitly applies ML, State, Bayesian, and rules, simulates Monte Carlo outcomes, and then uses the calibrated MC/ML/State ensemble where weights exist. Sportsbook lines do not construct the projection.

## Canonical empirical rule layer

The production rule layer is real football logic, not a placeholder. It includes game-script and play-volume mechanics, offense-vs-defense success, pressure mismatch, coverage/box tendencies, injury limitation/redistribution, RB receiving matchup effects, and RB rushing-efficiency multipliers for light/heavy boxes. Rules modify simulation assumptions before Monte Carlo rather than blindly multiplying final projections.

## Historical walk-forward baseline

The historical `walk_forward.py` / `component_predictions.py` path also constructs the canonical components at each historical cutoff:

- empirical Bayesian baseline feeds the rule layer and Monte Carlo,
- canonical rules are applied before simulation,
- MC projection is produced,
- supervised ML v2 is trained/predicted leakage-safely,
- State v2 is trained/predicted leakage-safely.

Therefore the M91 artifact contained MC, ML, and State component predictions rather than being a one-model-only backtest.

## Critical M94/M94C distinction

M94/M94B/M94C intentionally isolated the **team rushing-volume/game-state problem**. In doing so, their player baseline and team aggregation use the frozen M91 **`ml_proj`** directly:

- `baseline_team_rush_att` = sum of M91 `ml_proj` for `rush_att`;
- the M94C team-strength features aggregate M91 `ml_proj` by market;
- after M94C predicts improved team rushing volume, individual carries are redistributed using the player's existing ML share of the baseline team carry pool;
- rushing yards are translated with baseline ML implied YPC.

Thus M94C is **not the same thing as the final production MC + ML + State + Bayesian/rules ensemble**. It is a research architecture built around the ML component plus M94C's explicit team-volume/game-environment correction.

This was scientifically valid for isolating team opportunity, but it creates a major integration obligation now: the project has accumulated production rule/context machinery and M95/M96 RB research capabilities that are not all represented in the frozen M94C point.

## What this means for RB research now

Do not throw away M94C. It remains useful evidence and a strong central opportunity component. But future RB work should no longer frame the problem as simply finding a standalone model that replaces M94C.

The next architecture program must be **integration/additive**:

1. establish a production-equivalent historical RB baseline using the canonical MC/ML/State/Bayesian/rule machinery;
2. explicitly audit which production rules/context are already contributing to RB rush attempts/yards and whether their historical coverage is valid;
3. add the newly validated timestamp-safe current backfield/depth/snap state as an allocation correction;
4. preserve M94C's validated team-opportunity capability where it adds incremental value;
5. evaluate retained M95/M96 capabilities by job: M95F workload-tail distribution, M95I vacancy/transition, M95C environment, M96C opponent-defense efficiency, etc.;
6. use precommitted ablations/non-degradation gates to determine which capabilities improve the current system rather than searching for a monolithic replacement;
7. keep sportsbook downstream only.

## Immediate implication

Before treating the forthcoming allocation model as the new RB model, run a **production-equivalent RB stack audit/backtest** and establish the correct baseline. Then test `baseline + allocation`, `baseline + M94C opportunity`, and compatible retained modules against that same historical panel.

This is now a continuity requirement.