# RB R26J 2020 Week-1 Comparability Source Audit V1 — Implementation Lock

Date: 2026-09-09
Status: LOCKED BEFORE EXECUTION

Frozen plan: `docs/research/RB_R26J_2020_WEEK1_COMPARABILITY_SOURCE_AUDIT_V1_FROZEN_PLAN.md`

Implementation details fixed before source results:

- R26 prediction CSVs are read with an explicit pregame-only `usecols` whitelist. No `actual_*` column is loaded.
- Week-1 vacancy team-week keys are the unit of source comparison.
- R26C source disposition must prove ACT/INA-only roster state, no target-game outcomes/participation, no same-week historical depth, and zero sportsbook inputs.
- R26D meaningful-exit definition remains exactly: `prior_targets_pg > 1 OR prior_rb_room_share >= 0.25`.
- `entrants_n` is taken from frozen R26 room state; entrant share is `entrants_n / current_room_n`.
- returning continuity count/share uses `continuing_same_team` from R26.
- baseline room HHI is `sum(baseline_room_share^2)` within the current RB room.
- R26 allocation-shift L1 is `sum(abs(candidate_room_share - baseline_room_share))` within the current RB room.
- for the plan's “non-near-zero positive metric” relative-shift rule, the fixed implementation floor is `abs(mean_2021_2025) >= 0.05`.
- rate dimensions are predeclared in code and use the frozen >=0.15 absolute rate-shift rule.
- range separation is strict outside the 2021-2025 season-level min-max range, with floating tolerance `1e-12`.
- a source dimension is counted at most once toward the audit-level distinction even if it satisfies multiple rules.
- audit-level `DISTINCT` still requires at least 3 dimensions across at least 2 of sections A-D.
- section E source-integrity differences are reported but do not count toward the A-D distinction requirement.
- no 2020 outcome/error data is loaded or scored.

R26J can authorize only a later frozen mechanism/comparability diagnostic. It cannot authorize shadow, production, or exclusion of 2020.
