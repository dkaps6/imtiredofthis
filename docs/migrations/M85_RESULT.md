# Migration 85 — Authoritative Result

## Disposition

`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

Migration 85 completed the frozen `TRUE_BLOCKER_X_TRUE_RUSHER_ASSIGNMENT` source/feasibility audit successfully. Exact who-blocked-who information is scientifically available in NFL tracking/competition systems, but no free source satisfied the complete multi-season historical + in-season + machine-readable contract required for a legitimate QB predictive test.

## Authoritative run

- GitHub Actions workflow: `Migration 85 QB True Blocker-Rusher Source Audit`
- Run: `33323202581` (Run #1)
- Conclusion: `success`
- Artifact: `m85-blocker-rusher-source-audit`
- Artifact ID: `9735478496`
- Artifact SHA256: `cc1c8f88041fed10efd913e0ab473273561609fb5d317f17fc5a13d5b6c8d290`
- Source candidates: `4`
- Qualifying sources: `0`
- QB outcomes read: `False`
- Sportsbook features used: `False`
- Production actionable: `False`
- M86 predictive development allowed: `False`

## Candidate source results

| Source | Exact assignment? | Historical contract | In-season contract | Free phase | Disposition |
|---|---|---|---|---|---|
| NFL Next Gen Stats blocking matchups | Yes | Internal history since 2018, no public machine-readable contract | No public feed contract | No | `IDEAL_BUT_PROPRIETARY` |
| Big Data Bowl / PFF blocker-rusher assignments | Yes | Competition slice only | No | Yes | `COMPETITION_SLICE_ONLY` |
| nflverse participation/pass-rush | No | Multi-season | No for 2023+ participation | Yes | `AGGREGATE_PLAY_CONTEXT_ONLY` |
| Public advanced pass-rush / OL tables | No | Multi-season | Yes | Yes | `AGGREGATE_PLAYER_TABLES_NOT_ASSIGNMENTS` |

## Interpretation

M85 does **not** reject the football mechanism that a specific blocker-rusher mismatch can create unexpected QB pressure and passing outcomes. It rejects pretending that current free public aggregate data are equivalent to exact assignment information.

NFL Next Gen Stats describes automated blocker-rusher identification and pressure probability with historical tracking coverage. Big Data Bowl scouting data expose exact fields such as `blockedPlayerNFLId1/2/3` and blocker pressure attribution. Those sources demonstrate scientific feasibility, but the exact information is not available to this project as a stable free 2023-2026 historical + live feed.

Public nflverse/PFR-style sources remain useful for aggregate pressure and OL/pass-rush context, but those mechanisms are not materially new relative to prior M56/M69/M80-M81 work and therefore cannot substitute for the M85 assignment hypothesis.

## Anti-loop consequence

Do not open M86 by:

- substituting aggregate pressure rate;
- substituting sacks/hurries/hits;
- substituting OL continuity;
- substituting player pass-rush win-rate tables;
- using a Big Data Bowl competition slice as if it covered complete 2024/2025/live seasons;
- changing algorithms over the same aggregate pressure information.

A predictive revisit requires a genuinely new source contract containing exact blocker-rusher assignment/exposure historically and in-season.

## Strategic frontier after M85

M82 established the authoritative clean full-stack QB passing benchmark at `56.749517` MAE with `123` 100+ yard misses and a nondeployable hindsight model-library oracle at `41.103131` MAE.

M83 then rejected the comparable-offense defensive-adaptation mechanism scientifically. M84 and M85 show that two of the strongest remaining player-level matchup mechanisms — top-weapon coverage responsibility and exact blocker-rusher assignment — are currently blocked by data access rather than by model architecture.

The next migration should therefore **not** create another transformation of already-tested public aggregate information. The recommended next step is an error-floor / recoverability audit of the M82 full-stack trace:

- classify the 123 100+ yard ensemble misses into attempt-volume, YPA-efficiency, or mixed failure;
- attribute, where historical play-by-play permits, how much of each miss is associated with target-game events not knowable pre-kickoff (explosive YAC/long completions, turnovers and short fields, sacks/scrambles, early injuries/benching, overtime, unusual fourth-down/drive outcomes, etc.);
- separate plausibly recoverable pregame regime errors from event-driven stochastic misses;
- estimate a practical pregame forecast-error floor before opening another information frontier.

This diagnostic must not claim causality from postgame variables or feed them into a pregame model. Its purpose is to quantify where the remaining error actually comes from and determine whether continued free-source pregame feature research has a realistic path to materially lower MAE.
