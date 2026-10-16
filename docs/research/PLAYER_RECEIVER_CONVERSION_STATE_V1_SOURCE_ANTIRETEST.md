# PLAYER RECEIVER CONVERSION STATE V1 — SOURCE / ANTI-RETEST AUDIT

Date: 2026-10-13

Status: **FROZEN SOURCE AUDIT ONLY — NO PREDICTIVE CANDIDATE AUTHORIZED YET**

## Why this audit exists

The Week-5 postmortem still contains a meaningful efficiency-dominant miss
family, but `PLAYER_EFFICIENCY_TAIL_STABILITY_V1` closed the simple hypothesis
that recent player YPC/YPT volatility itself reliably predicts next-game
efficiency-tail risk.

A next experiment is legal only if it uses materially different football
information.

The candidate information family under source audit is **receiver-level
conversion attribution**:

- charted receiver drops (`is_drop`);
- charted throw catchability (`is_catchable_ball`);
- stable receiver GSIS identity;
- receiver position from weekly roster authority.

The football question, if the source survives, would be whether the *specific
receiver's conversion environment* contains pregame information beyond the
canonical catch-rate mean. This source audit does **not** answer that question.

## Anti-retest inventory

This is not permission to reopen any of the following.

### M34 receiving decomposition

Migration 34 already decomposed receiving error into:
- opportunity / targets;
- catch conversion;
- yards-per-target efficiency.

It diagnosed where error lives but did not use charted receiver drop blame or
receiver-level catchability history as a pregame mechanism.

### QB M70 / M71

QB completion/CPOE/YPA mechanism decomposition and generic QB efficiency
volatility are closed. No QB mean or QB distribution candidate is being opened.

### M81 FTN QB development

M81 tested FTN charting at the **QB/offense** level, including:
- QB catchable-ball history;
- offense/QB drop-rate history.

Disposition:
`NO_FTN_DEVELOPMENT_SURVIVOR`.

Do not rerun M81, change its model, or claim its failed QB/offense aggregation
was positive.

The narrow novelty being audited here is different:
**receiver-attributed player history** joined to the named WR/TE, not a QB or
offense aggregate.

### WR / TE historical efficiency work

Remain closed/protected:
- WR-R3 combined calibration:
  `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`;
- WR R7 persistent explosive/YAC/air-yard traits for the YPR-dominant subgroup;
- vendor NGS R9-R11 target/yardage source lane;
- generic team Coverage-v2 man/zone efficiency use;
- fabricated historical WR-CB assignment;
- raw QB-receiver pair YPT;
- universal symmetric target-depth distribution transform;
- direct WR/TE depth-YPT transform;
- current-only early-season raw catch-rate/YPT state as a mean replacement.

The confirmed target-depth-dispersion signal remains real, but its universal
distribution transform failed. This audit may not use that same transform under
another name.

### RB

RB receiving-yard mean rescue from generic historical YPR/YPT/YAC/xYAC
transformations remains closed. RB R22 tail authority is protected.

Therefore this source audit is WR/TE-only for any eventual predictive follow-up.

## Source contract to audit

Historical seasons:
- 2023
- 2024
- 2025

Regular-season weeks:
- 1-18 only

Raw sources:
1. nflverse FTN charting release:
   `ftn_charting_<season>.parquet`
2. nflverse play-by-play:
   `play_by_play_<season>.parquet`
3. nflverse weekly rosters through `nflreadpy.load_rosters_weekly`

Required FTN fields:
- `nflverse_game_id`
- `nflverse_play_id`
- `season`
- `week`
- `is_drop`
- `is_catchable_ball`

Required PBP fields:
- `game_id`
- `play_id`
- `season`
- `week`
- `season_type`
- `posteam`
- `defteam`
- `pass_attempt`
- `complete_pass`
- `receiver_player_id`
- `receiver_player_name`

Required roster identity:
- season/week;
- GSIS/player ID;
- position.

## Frozen source gates

Every season must independently pass:

1. FTN regular weeks 1-18 complete.
2. PBP regular weeks 1-18 complete.
3. FTN -> PBP play join rate >= 95%.
4. Stable receiver GSIS ID present on >= 95% of receiver-named pass attempts.
5. Weekly-roster position mapping >= 95% of stable receiver-ID rows.
6. `is_drop` observed on >= 80% of stable receiver-ID pass attempts.
7. `is_catchable_ball` observed on >= 80% of stable receiver-ID pass attempts.
8. At least 100 distinct WR/TE receivers have >=4 receiver-attributed games in
   the season.
9. At least 75 distinct WR/TE receivers have >=8 receiver-attributed games in
   the season.

These thresholds are frozen before the source is scored.

## Output / decision

If all gates pass in all three seasons:

`PLAYER_RECEIVER_CONVERSION_SOURCE_READY`

This authorizes only a separately frozen historical mechanism experiment.

If any season fails:

`PLAYER_RECEIVER_CONVERSION_SOURCE_BLOCKED`

No predictive candidate may be fit from this source contract and no threshold,
identity, support, or season rescue is allowed after seeing the result.

## Hard integrity boundaries

This audit:
- uses no sportsbook information;
- makes no OddsAPI call;
- uses no Week-5 or Week-6 betting outcome to select features;
- fits no predictive model;
- changes no canonical mean;
- changes no production code;
- cannot promote anything.

If source-ready, the next contract must still freeze the exact player-level
mechanism, target, controls, chronology, seasons, support, inference, and
promotion gates before predictive scoring.
