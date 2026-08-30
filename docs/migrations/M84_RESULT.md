# Migration 84 — Authoritative Result

## Disposition

`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

Migration 84 completed the frozen `TOP_WEAPON_ESCAPE_HATCH` source/feasibility audit successfully. The football hypothesis remains plausible, but no free source candidate satisfied the complete historical + in-season + machine-readable + receiver/defender-responsibility contract required for a legitimate predictive test.

## Authoritative run

- GitHub Actions workflow: `Migration 84 QB Top Weapon Source Audit`
- Run: `33323056814` (Run #1)
- Conclusion: `success`
- Artifact: `m84-top-weapon-source-audit`
- Artifact ID: `9735439868`
- Artifact SHA256: `a4d0f3e1ecfe6cdded18c7ac435cd07a32d8a57a80b383e1a7a7805dcd9baef7`
- Source candidates: `5`
- Qualifying sources: `0`
- QB outcomes read: `False`
- Sportsbook features used: `False`
- Production actionable: `False`
- M82 full-stack benchmark retained: `56.749517` MAE

## Candidate source results

| Source | Novel exact matchup? | Historical contract | In-season contract | Free phase | Disposition |
|---|---|---|---|---|---|
| NFL Next Gen Stats Coverage Responsibility | Yes | No public machine-readable contract | No public feed contract | No | `IDEAL_BUT_PROPRIETARY` |
| Big Data Bowl / PFF coverage assignments | Yes | Competition slice only | No | Yes | `COMPETITION_SLICE_ONLY` |
| nflverse participation route data | No direct receiver-defender responsibility | Historical multi-season | No (2023+ released after season) | Yes | `HISTORICAL_ROUTE_ONLY_NONDEPLOYABLE` |
| Fantasy Points / VSiN WR-CB report | Yes/current matchup concept | No stable machine-readable multi-season archive established | Yes | Yes | `CURRENT_REPORT_NO_STABLE_HISTORY_CONTRACT` |
| PFF WR-CB matchup chart | Yes | No free historical machine-readable contract | Yes | No | `PAID_NO_FREE_RESEARCH_CONTRACT` |

## Interpretation

M84 does **not** reject the football idea that one extreme weapon matchup can allow a QB to outperform an otherwise poor macro matchup. It rejects opening a predictive migration with the sources currently available for free.

The exact information we want is real: modern NFL tracking systems can identify receiver-defender coverage responsibility and matchup assignment. The blocker is acquisition, not conceptual validity.

The public Big Data Bowl slices prove that exact matchup IDs and route labels are scientifically tractable, but those competition datasets are not an extensible 2023-2025 + live source.

nflverse participation has useful route information but lacks direct receiver-defender responsibility and, from 2023 onward, is not released in-season. Current WR/CB tools can be useful for present-day handicapping, but without a stable historical archive they cannot support a leakage-safe backtest.

## Anti-loop consequence

Do not open M85 by:

- reusing M72 aggregate explosive-weapon matchup;
- reusing M75 NGS/PFR receiver-secondary aggregates;
- correlating realized receiver big games with QB big games;
- treating a current-only WR/CB webpage as historical training data;
- using the Big Data Bowl competition slice as though it covered 2024/2025/live seasons;
- switching algorithms over the same rejected receiver/secondary proxies.

A future revisit requires a genuinely new source contract: historical + live receiver-to-defender/responsibility matchup data (or an equivalent pregame single-weapon exposure observable).

## Next migration boundary

M85 predictive development is **not allowed** from M84.

The next migration should move to another surviving M82 new-information frontier rather than model-zoo rescue. The strongest remaining frontier is exact `TRUE_BLOCKER_X_TRUE_RUSHER_ASSIGNMENT`, provided a broader historical/live source can be found beyond limited competition data. If that source contract also fails, the research program should explicitly reassess whether the remaining pregame headroom is primarily inaccessible/proprietary or intrinsically game-day stochastic rather than continuing to manufacture new transformations of existing information.
