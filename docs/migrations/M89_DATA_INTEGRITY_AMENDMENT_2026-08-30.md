# M89 Data-Integrity Amendment — 2026-08-30

## Why this amendment exists

M89 preregistered an independent truth gate that attempted to reconstruct official QB pass attempts from parsed nflverse play-by-play and require >=99% exact agreement with nflverse weekly player stats.

Run #9 demonstrated that assumption was invalid as a source contract:

- raw parsed-PBP `pass_attempt` includes sacks;
- excluding sacks improved agreement substantially, but parsed PBP still matched official weekly attempts exactly in only ~87.84% of matched QB-games;
- official passing yards matched at 100%;
- the remaining parsed-PBP attempt differences were small and systematically positive, consistent with parsed-play semantics rather than a corrupt weekly-stat source.

The important scientific conclusion is that parsed PBP should not be forced to serve as the official box-score ledger. Official weekly player stats and parsed PBP have different jobs.

## Amended source-of-truth contract

Effective before the next authoritative M89 rerun:

### Official statistical outcomes

Use nflverse weekly player stats loaded through:

`nflreadpy.load_player_stats(..., summary_level="week")`

as the source of truth for:

- official pass attempts;
- official passing yards;
- team-week official pass attempts/yards used in dropback-to-attempt conversion.

The hard integrity gate remains >=99% exact agreement, but it now tests the frozen canonical QB actuals against this official weekly-stat source rather than pretending parsed PBP is an independent official-stat ledger.

### Parsed play-by-play

Use parsed nflverse PBP for football mechanics that require play sequence/state:

- QB/team dropbacks;
- sacks and QB hits;
- scrambles;
- score/game state;
- drives and possession shape;
- xpass / expected pass probability;
- EPA and success;
- air yards, YAC and explosive-play forensics;
- quarter/drive descriptive attempt approximations in the casebook.

Parsed-PBP pass-attempt agreement with official weekly stats is still emitted as a diagnostic, but it is not a promotion gate.

### Dropback-to-official-attempt conversion

For each completed team-week observation:

`pass_attempts_per_dropback = official_team_pass_attempts / parsed_PBP_team_dropbacks`

The numerator therefore matches the quantity the QB passing-yards model is trying to project, while the denominator retains the football opportunity process represented by PBP.

## No model tuning is changed

This amendment changes no predictive result threshold and does not use 2024-2025 model performance to tune a candidate.

The following remain frozen exactly as preregistered:

- 2023-only synthesis training;
- locked 2024-2025 evaluation cohort;
- Ridge only;
- `alpha=20`;
- residual correction cap `+/-45 yards`;
- football-only feature universe;
- separately labeled market-assisted feature universe;
- MAE/RMSE/correlation/tail/bootstrap promotion gates;
- no Phase-1 postgame variable in Phase 2;
- no sportsbook variable in the football-only model.

Because official team attempts affect the historical MC opportunity conversion, 2023, 2024 and 2025 must all be rebuilt and rerun after this amendment. Results from Run #9 remain diagnostic/provisional only.

`model_thresholds_changed = false`

`evaluation_rows_changed = false`

`official_attempt_source = nflverse_weekly_player_stats`

`parsed_pbp_attempts_are_official_source = false`
