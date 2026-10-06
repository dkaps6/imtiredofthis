# Public Position YPT Allowed Parity V1 — Frozen Source Contract

**STATUS: FROZEN BEFORE SOURCE-PARITY RESULT. RESEARCH ONLY. NO CANDIDATE SCORE.**

## Purpose

Resolve one specific source block left by Football Matchup Transmission V1.

Phase B/C found that:

`TE_REC x def_te_ypt_allowed`

replicated in the expected direction in both 2024 and 2025, but was not allowed to become an integration candidate because the historical diagnostic was built from public official weekly player statistics while the live production field is Sharp-derived.

This audit asks a narrower question:

> Can we build the historical diagnostic definition itself from the same free public nflverse/nflreadpy weekly-stat source in 2024, 2025, and live 2026, so it becomes a separate same-semantic public football input rather than pretending it is identical to the Sharp field?

This does **not** alter or replace the existing Sharp field. It establishes whether a new public field can have exact historical/live source parity.

## Frozen semantic definition

Source:
- `nflreadpy.load_player_stats(seasons=[season], summary_level="week")`
- normalized through the same `scripts.player_form_v2._normalize_weekly` path used by canonical historical player logs;
- opponent attached from `nflreadpy.load_schedules` regular-season schedule.

For each completed player-game:
- classify position group:
  - WR = WR
  - TE = TE
  - RB = RB/HB/FB
- receiving opportunities = `targets`
- receiving production = `rec_yards`
- defense = opponent.

For target `(season, week, defense)`:

1. use only rows from the same season with source week strictly less than target week;
2. retain only the latest 8 distinct completed source weeks for that defense;
3. for each position group:
   `position_ypt_allowed_public = sum(rec_yards) / sum(targets)`;
4. denominator must be > 0 or field is missing;
5. no target-week or future row may enter;
6. no sportsbook or market input may enter.

This is intentionally the exact definition used by
`build_position_ypt_pregame()` in the frozen Football Matchup Transmission Phase B/C audit.

## Historical reproduction gate

Authority:
- Phase B/C run `37514137803`
- artifact `11436786668`
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- file `football_matchup_phase_bc_team_features.csv`

Rebuild the public position-YPT fields independently from nflreadpy for 2024 and 2025.

Join on:
- season
- week
- offense team
- opponent defense

Compare:
- rebuilt opponent WR YPT allowed vs `def_wr_ypt_allowed`
- rebuilt opponent TE YPT allowed vs `def_te_ypt_allowed`
- rebuilt opponent RB YPT allowed vs `def_rb_ypt_allowed`

Historical parity requires:
- exact identity coverage for all authority rows where either side is finite;
- finite/missingness agreement >= 99.5%;
- max absolute numerical gap <= `1e-10` on mutually finite rows.

If this fails, do not reinterpret the old diagnostic.

## Live 2026 deployment-readiness gate

Build the same field for target Week 5 using only completed 2026 Weeks 1-4.

The source is deployment-ready only if:
1. 2026 weekly player stats are nonzero;
2. regular-season schedule covers all Week-5 teams;
3. every Week-5 defense has a public TE YPT value with positive target denominator;
4. chronology violations = 0;
5. all values come from source weeks < 5;
6. same code path can emit WR/RB values too, although TE is the only historically replicated family currently eligible for possible follow-up;
7. sportsbook inputs = 0;
8. target Week-5 outcomes = 0;
9. production change = false.

## Dispositions

`PUBLIC_POSITION_YPT_EXACT_PARITY_READY`
requires both historical reproduction and live-2026 readiness.

`PUBLIC_POSITION_YPT_HISTORICAL_PARITY_ONLY`
if 2024/2025 reproduce exactly but live Week-5 coverage is incomplete.

`PUBLIC_POSITION_YPT_PARITY_FAILED`
if the historical reproduction fails.

No predictive candidate is scored by this audit.

If exact parity is ready, only the already-replicated
`TE_REC x public_def_te_ypt_allowed`
may advance to a separately frozen integration-candidate contract. WR/RB YPT may not be promoted merely because the source is available; their Phase B/C residual signal did not replicate.

## Explicit boundaries

- Do not compare public YPT numerically to Sharp and demand equality.
- Do not overwrite the Sharp field.
- Do not use outside/slot YPT.
- Do not score Week 4 or Week 5 outcomes.
- Do not fit a coefficient.
- Do not create WR/RB candidates.
- Do not use sportsbook data.
- Do not mutate production.

Candidate variants scored: **0**  
Parameters fit: **0**  
Sportsbook inputs: **0**  
Target Week-5 outcomes: **0**  
Production mutations: **0**
