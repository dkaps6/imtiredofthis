# RB STACK6 / ND3 — Secondary-Back Role & Substitution Source Audit

## Why this exists

STACK5 localized the remaining 2025 Vegas MAE advantage primarily to M95F-nonrisk, depth-rank 2/3 backs and especially to large false-high disagreements where projected carries exceeded actual carries. The next justified information family is therefore player-specific situational participation and substitution role, not another generic global rushing model.

## Scientific question

Can strictly prior-game, player-level participation reconstruct materially different RB roles that are hidden by aggregate depth rank and total snap percentage?

Examples: early-down runner, third-down/pass-down specialist, two-minute back, short-yardage/goal-line specialist, red-zone back, rotating-series back, and multi-RB personnel specialist.

## Source contract

Primary audit source: nflverse participation joined to nflverse PBP at game/play grain.

- Target-game participation is POSTGAME truth and may never be used as a pregame feature.
- Only completed games strictly before the target game/week may create projection features.
- nflverse participation 2023+ is treated as HISTORICAL-ONLY because the public release is postseason. A winning feature family still requires a separate live-capable 2026 source before production use.
- PBP-derived situation labels are historical truth used only to classify prior-game participation.
- No sportsbook data enter this audit or any future football-model fit.

## Audit outputs

1. Source schema and play-key join coverage for 2024-2025.
2. Player/position list parse coverage.
3. RB/FB player-game situational participation table:
   - all offensive plays
   - early downs
   - third downs
   - third-and-long
   - two-minute offense
   - short yardage
   - red zone
   - inside 10
   - inside 5
   - shotgun / under-center where available
   - 11/12/21/22 personnel where available
4. Rushing-attempt ownership where rusher IDs can be matched.
5. Strictly lagged prior-game/pre-target features and coverage.
6. Team backfield rotation descriptors: number of RBs used, concentration/HHI, and role specialization.
7. Explicit source/live-capability manifest.

## Gate to proceed to STACK6 modeling

Proceed only if:
- participation/PBP play-key match >= 95%;
- aligned offense player/position parse coverage >= 95% of eligible joined plays;
- player identity/name coverage is sufficient to join >= 90% of STACK5 market/player rows after canonical-name normalization OR a stable player-id bridge exists;
- at least early-down, third-down, two-minute, short-yardage, and red-zone role features have usable prior-game coverage;
- leakage audit confirms target-game participation is never used as a feature.

Failure is a source/mechanical finding, not a model failure. Do not weaken these gates.

## Mechanical audit corrections

The first source run exposed a chronology-check bug only: a valid prior observation from 2024 Week 18 was compared numerically to 2025 Week 1 as `18 < 1`. The first repair compares a season-week ordinal instead.

The second source run exposed a separate boolean-check bug: Pandas evaluates `NaN < target` as `False` before `fillna`, so every player's first historical observation was incorrectly labeled leakage. The corrected audit explicitly treats `no prior observation` as safe missing history, and otherwise requires `prior season-week ordinal < target season-week ordinal`.

Neither correction changes a source definition, feature family, sportsbook rule, or scientific threshold. The player-level on-field proxy is treated only as RB-presence allocation and will not be interpreted as true offensive snap share; STACK2's qualified snap data remains the snap-share source in downstream modeling.
