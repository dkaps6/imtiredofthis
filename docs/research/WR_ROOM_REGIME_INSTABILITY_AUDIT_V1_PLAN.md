# WR Room Regime-Instability Audit V1 — Diagnostic Plan

Status: **FROZEN DIAGNOSTIC-ONLY PLAN**

Parent scientific disposition:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

This audit is not a rescue of Receiver Room Targets-Per-Play V1. The failed formula stays closed.

## Purpose

Explain why the unchanged WR room targets-per-play candidate improved WR room MAE in 2022, 2023 and 2025 but worsened in 2024, while TE and RB/FB were materially more directionally stable.

The audit is hypothesis discovery only.

- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs: **0**
- production mutation: **0**
- no player/full-stack integration

## Frozen outcome authority

The audit consumes the exact already-scored room-detail artifacts rather than recomputing or altering the failed candidate.

Discovery authority:
- run `36172644864`
- artifact `10880948137`
- file `room_detail_2022_2023.csv`
- disposition `RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED`

Confirmation authority:
- run `36174077739`
- artifact `10881825507`
- file `room_detail_2024_2025.csv`
- disposition `RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

The audit may compute descriptive error deltas from those frozen rows but may not alter predictions, gates, seasons, specialists, windows, rates, weights, or baselines.

## Questions

At the team-WR-room game level, determine:

1. Is 2024 harm broad across the league or concentrated in a limited set of teams?
2. Is harm associated with large prior-season -> current-season WR room rate movement?
3. Is harm associated with WR personnel turnover visible before kickoff?
4. Is harm associated with QB transition visible before kickoff?
5. Is harm associated with verified play-caller transition where source coverage exists?
6. Does the cumulative prior-season + strict-current-history rate adapt too slowly after an identifiable transition?
7. Are the same structural relationships directionally present in 2022, 2023 and 2025, or is any apparent explanation merely a 2024-only retrospective story?

## Allowed explanatory descriptors

### A. Leakage-safe pregame WR room state

For each team/week:
- prior-season WR targets per offensive play;
- current-season strict-prior WR targets per offensive play;
- strict-prior current-vs-prior WR rate gap;
- number of current-season games already observed;
- fraction of the failed candidate's historical game mass contributed by current-season games.

These use only prior-season and games strictly before the target week.

### B. Leakage-safe WR roster continuity

Use the existing historical pregame universe built from nflverse weekly rosters.

For each team/week:
- fraction of prior-season WR target mass belonging to WRs still present on the target-week pregame roster;
- whether the prior-season leading WR target earner is still on the target-week pregame roster;
- whether the strict-prior current-season leading WR target earner differs from the prior-season leader, when at least one current-season game exists.

No target-week participation/results may define the roster.

### C. Leakage-safe QB transition

Using historical player logs and the same pregame roster authority:
- prior-season primary QB by pass attempts;
- whether that QB is present on the target-week pregame roster;
- whether the strict-prior current-season primary QB differs from the prior-season primary QB, when current-season prior games exist.

This is descriptive transition context only.

### D. Verified play-caller transition

Use only the already-frozen verified play-caller mapping in
`scripts/backtest/build_qb_playcaller_opening_leverage.py`.

Allowed:
- season-opening caller change versus the prior season where both seasons have verified coverage;
- documented midseason caller change already effective by the target week.

Do not fetch or infer a new coaching source inside this audit.

Coverage is expected to be incomplete for earlier seasons and must be reported explicitly.

### E. Outcome-only season drift descriptor

For mechanism description only, the audit may compute realized full-season WR targets-per-play drift from prior season to current season.

This field is **not pregame deployable** and may not be used to justify a deployable candidate by itself. It exists only to answer whether 2024 was an unusually large realized regime shift.

## Frozen analyses

The audit will report:

- WR candidate absolute-error delta:
  `abs(candidate - actual) - abs(baseline - actual)`
  where positive means the failed candidate harmed that row;
- by-season WR MAE reproduction from frozen room-detail files;
- number/share of teams with positive vs negative season-aggregate harm;
- concentration of 2024 excess harm, including top-8-team share and ranked team table;
- Spearman associations between row-level harm and continuous pregame descriptors;
- grouped descriptive means for natural binary transition states;
- week-by-week harm and adaptation-state summaries;
- team-season structural table for 2022-2025;
- source/provenance/coverage audit.

These are diagnostics, not candidate scorecards.

No optimized threshold, subgroup cutoff, window, weight, blend, multiplier, shrinkage, bias offset or season route may be searched.

## Interpretation rule

The audit must end in one of two states:

1. **STRUCTURAL HYPOTHESIS WARRANTED** — a genuinely pregame structural mechanism is coherent enough across the evidence to justify freezing a separate new hypothesis before any scoring; or
2. **NO STRUCTURAL EXPLANATION / CLOSE FAMILY** — evidence is diffuse, retrospective, season-specific, or not leakage-safe enough, so the room-history family closes and research moves to another architecture frontier.

The audit itself cannot promote or integrate anything.

## Explicit prohibitions

Do not:
- exclude 2024;
- route WR separately;
- change the room-rate history window;
- add recency/shrinkage;
- blend fixed57;
- add a bias correction;
- special-case WR1/Q4;
- change TE-R5P/WR-R15 order;
- fit 2026 outcomes;
- use sportsbook information;
- score any new candidate;
- reopen C1/C3, Room Targetable-Rate V1, Active-Roster Room State V1, WR1-only current-state, One-Pass V1, hierarchical reconciliation, official-attempt pool, uniform targetable-dropback player thinning, Rush Pool rescues, or M96E-closed RB routing.

## Required artifact

The run must emit:
- `wr_room_regime_detail.csv`
- `team_season_regime_summary.csv`
- `season_summary.csv`
- `association_summary.csv`
- `transition_group_summary.csv`
- `source_coverage.csv`
- `summary.json`
- `RESULT.md`

The payload must state:
`candidate_variants_scored = 0`.
