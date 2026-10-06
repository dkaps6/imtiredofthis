# Public Position YPT Allowed Parity V1 — Result

**STATUS: COMPLETE / EXACT HISTORICAL-LIVE PARITY READY**

Frozen plan:
`docs/research/PUBLIC_POSITION_YPT_ALLOWED_PARITY_V1_PLAN.md`

Authority:
- branch: `research-public-position-ypt-parity-v1`
- successful run: `37545711420`
- source SHA: `8c41021f7c3904d449897d33ddb82d01eadd5398`
- artifact: `11451030931`
- digest: `sha256:7a52455276e99efa18b4b6b9e22fd248d50038e2da9f3cae4e4fddf3e4c329d3`

Final disposition:

`PUBLIC_POSITION_YPT_EXACT_PARITY_READY`

## Exact source semantic

The new public field uses:

- `nflreadpy.load_player_stats(..., summary_level="week")`;
- canonical `scripts.player_form_v2._normalize_weekly`;
- regular-season opponent identity from `nflreadpy.load_schedules`;
- same-season source weeks strictly before the target week;
- latest eight distinct prior source weeks;
- `sum(rec_yards) / sum(targets)` by opponent defense and position group.

This is the exact public-stat definition used by Football Matchup Transmission Phase B/C. It is a new public input surface; it is **not** a claim that public YPT equals the existing Sharp field.

## Historical reproduction

Compared against frozen Phase B/C authority:
- run `37514137803`
- artifact `11436786668`
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- 1,024 target team-week rows

Identity coverage:
- `1.0`

WR:
- either-finite rows: 1,024
- both-finite rows: 1,024
- missingness agreement: 1.0
- max absolute gap: `1.7763568394002505e-15`

TE:
- either-finite rows: 1,024
- both-finite rows: 1,024
- missingness agreement: 1.0
- max absolute gap: `1.7763568394002505e-15`

RB:
- either-finite rows: 1,023
- both-finite rows: 1,023
- missingness agreement: 1.0
- max absolute gap: `1.7763568394002505e-15`

All three source-parity gates pass.

## Live 2026 Week-5 readiness

No Week-5 outcome was read.

- completed-prior 2026 player-stat rows: 1,263
- maximum source week used: 4
- Week-5 scheduled teams: 30
- Week-5 defenses with positive-denominator TE YPT: 30
- missing Week-5 TE defenses: none
- chronology violations: 0

Research boundary:
- predictive candidates scored: 0
- parameters fit: 0
- sportsbook inputs used: 0
- target Week-5 outcomes read: 0
- production changed: false

## Interpretation

The old source block on `TE_REC x def_te_ypt_allowed` is resolved **by defining a separate public same-semantic feature**, not by asserting provider equality with Sharp.

The Phase B/C TE YPT relationship had already replicated across both 2024 and 2025 but was marked:

`REPLICATED_DIAGNOSTIC_SOURCE_PARITY_BLOCKED`

because the then-current candidate gate required the live Sharp field to match the historical official-stat field.

That limitation no longer applies to a new public-field candidate because the same free source and formula are now available historically and live.

This does **not** authorize WR or RB YPT candidates. Their Phase B/C signals did not replicate.

This also does not authorize production. The next permitted step is a separately frozen TE-only candidate contract using out-of-selection 2022/2023 evidence.

No paid OddsAPI or paid external source is required.
