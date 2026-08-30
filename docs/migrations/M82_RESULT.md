# Migration 82 — Authoritative Result

## Disposition

`FULL_STACK_RECONCILIATION_COMPLETE`

Migration 82 completed a clean 2024-2025 production-style walk-forward reconciliation across the exact canonical-v3 QB identity set. The result establishes the current authoritative football-only full-stack QB passing benchmark and freezes the M40-M81 integration-eligibility ledger.

## Authoritative run

- GitHub Actions workflow: `Migration 82 QB Full-Stack Clean Reconciliation`
- Run: `33320763343` (Run #1)
- Artifact: `m82-full-stack-clean-reconciliation`
- Artifact ID: `9734973786`
- Artifact SHA256: `1c62eccd4803d870676d0fce35ab26fe8338558eb49a3ae760688e7a7b0d8459`
- Canonical snapshot SHA256: `c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742`
- Canonical QB games: `884` (`444` in 2024, `440` in 2025)
- Monte Carlo iterations: `2000`
- Sportsbook features used: `False`
- Production actionable: `False`

## Authoritative QB passing scoreboard

| Season | Model | N | Coverage | MAE | RMSE | Bias | Correlation | 100+ misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | canonical_v3 | 444 | 1.0000 | 60.151290 | 75.757791 | -18.649312 | 0.118452 | 74 |
| 2024 | current_mc | 444 | 1.0000 | 60.151290 | 75.757791 | -18.649312 | 0.118452 | 74 |
| 2024 | current_ml | 444 | 1.0000 | 58.673821 | 74.684949 | -10.204457 | 0.163270 | 70 |
| 2024 | current_state | 441 | 0.9932 | 58.601052 | 74.953003 | -23.082350 | 0.115788 | 76 |
| 2024 | oos_ensemble | 444 | 1.0000 | **57.241861** | **72.976406** | -17.779733 | **0.164181** | **67** |
| 2025 | canonical_v3 | 440 | 1.0000 | 56.843832 | 72.161124 | -8.112639 | 0.126202 | 68 |
| 2025 | current_mc | 440 | 1.0000 | 56.843832 | 72.161124 | -8.112639 | 0.126202 | 68 |
| 2025 | current_ml | 440 | 1.0000 | 59.547053 | 76.203117 | -7.299505 | 0.060823 | 78 |
| 2025 | current_state | 440 | 1.0000 | 57.189777 | 72.894982 | -18.102102 | 0.117448 | 62 |
| 2025 | oos_ensemble | 440 | 1.0000 | **56.252696** | **71.618883** | -13.233423 | **0.127425** | **56** |
| COMBINED | canonical_v3 | 884 | 1.0000 | 58.505044 | 73.989452 | -13.404814 | 0.118114 | 142 |
| COMBINED | current_mc | 884 | 1.0000 | 58.505044 | 73.989452 | -13.404814 | 0.118114 | 142 |
| COMBINED | current_ml | 884 | 1.0000 | 59.108462 | 75.444417 | -8.758553 | 0.116758 | 148 |
| COMBINED | current_state | 881 | 0.9966 | 57.896216 | 73.932322 | -20.595052 | 0.119500 | 138 |
| COMBINED | oos_ensemble | 884 | 1.0000 | **56.749517** | **72.303902** | -15.516864 | **0.149475** | **123** |

## Primary conclusion

The authoritative clean full-stack benchmark is now:

- **QB passing MAE: `56.749517`**
- RMSE: `72.303902`
- correlation: `0.149475`
- 100+ yard misses: `123`

The OOS ensemble improves canonical-v3 by `1.755527` yards of MAE and reduces 100+ misses from `142` to `123` on the exact same 884-game population.

The current Monte Carlo/Bayesian/rules pass-yard output is identical to canonical-v3 on this cohort. The improvement over the canonical research baseline therefore comes from the independent ML/State layer and its OOS combination, not from an unaccounted-for Monte Carlo rules layer.

## Ensemble leakage boundary

- 2024: expanding earlier-week OOS component rows only; explicit MC fallback until at least 40 complete historical rows exist.
- 2025: weights fit only on 2024 OOS component predictions and frozen for the entire 2025 season.

No current-season future outcomes are used to select the target-game ensemble weights.

## Model diversity diagnostic

Representative component residual correlations remain high:

- canonical/current MC vs ML: `0.895563`
- canonical/current MC vs State: `0.919370`
- ML vs State: `0.893964`
- median absolute MC/ML/State residual correlation: `0.895563`

Therefore another same-information ensemble/model-zoo search is not the next research frontier.

## Hindsight library oracle

If one could choose after each game whichever of canonical/current full-stack model outputs was closest, the 884-game MAE would be:

- hindsight oracle MAE: **`41.103131`**
- best deployable single model in M82: `56.749517`
- hindsight model-selection headroom: **`15.646386` yards**

This oracle is not deployable. Its purpose is diagnostic: the existing model library sometimes contains a much better representation of a game, but the current pregame information does not reliably identify which model/regime will be correct.

## M40-M81 integration ledger

M82 freezes 27 research families/areas into:

- `PROMOTED_FOUNDATION`: 5
- `FULL_STACK_TESTED_CLOSED`: 7
- `SIGNAL_SCREEN_FAILED`: 11
- `SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED`: 4

No prior family is classified as a generic same-data `PARTIAL_SIGNAL_INTEGRATION_CANDIDATE`. A failed family may not be reopened merely by swapping Ridge/HGB/XGB/neural-network/ensemble architecture over the same information.

The four surviving new-information/source frontiers are:

1. route × coverage-shell interaction with a trustworthy historical + deployable pregame route contract;
2. true blocker × true rusher assignment;
3. `TOP_WEAPON_ESCAPE_HATCH`: a materially new pregame single-weapon route/responsibility matchup signal, not the rejected M72/M75 proxies;
4. `DEFENSIVE_ADAPTIVE_GAMEPLAN`: a pregame prediction of how a defense changes blitz/shell/man-zone/box/pressure behavior conditional on the opponent offense/QB archetype, using strictly prior comparable-opponent evidence.

## Next research boundary

The next migration should not tune the current model library. The highest-priority next step is a source/feasibility audit for `DEFENSIVE_ADAPTIVE_GAMEPLAN`.

That audit must establish, before predictive testing:

- whether defensive tactical-response variables can be reconstructed reliably across historical seasons;
- whether comparable-offense/QB archetype history is sufficiently dense;
- whether every target-game value is computable strictly pre-kickoff;
- whether the resulting observable is materially distinct from M56 static defensive context, M67-M69 offensive intent/opening tendencies, and M80 static coverage/FTN history.

Only if that source/novelty contract qualifies should a subsequent migration test its incremental predictive value against the `56.749517` full-stack benchmark and its attempt/YPA/tail failure modes.
