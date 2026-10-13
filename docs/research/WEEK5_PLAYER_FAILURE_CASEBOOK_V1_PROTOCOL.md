# WEEK 5 PLAYER FAILURE CASEBOOK V1 — PROTOCOL

Frozen: 2026-10-13

Status: **POSTGAME DIAGNOSTIC ONLY. NO WEEK-5 OUTCOME TUNING.**

Parents:
- `docs/research/WEEK5_FULL_MODEL_POSTMORTEM_PROTOCOL_V1.md`
- `docs/research/WEEK5_FULL_MODEL_POSTMORTEM_RESULT_V1.md`
- `docs/research/WEEK5_PROSPECTIVE_OPPORTUNITY_LOCK_POSTMORTEM_RESULT_V1.md`

## Purpose

For the largest Week-5 football projection misses, determine whether the miss was
primarily caused by:
- opportunity / usage;
- efficiency;
- both;
- or insufficient evidence.

This is a diagnostic of the frozen Week-5 model. It does not authorize a new
formula.

## Immutable authorities

Pregame:
- full-slate run `37860592615`
- artifact `11585973062`
- production behavior authority `24704e5f1ef872ec86c36446d2c73f6d377c824c`

Postgame full-board:
- run `38395384228`
- artifact `11812200405`
- digest `sha256:5888cab39506ae0bf70034dfb60f297defd28dd3219ea8118870738706f55ac3`

Actuals:
- current nflverse/nflreadpy 2026 weekly player stats;
- Week 5 only;
- stable GSIS identity where available.

## Case universe

Take the top **40 settled player-market rows by absolute final-model projection
error** from the canonical full-board postmortem.

No outcome-derived subgroup selection beyond that fixed top-40 ranking.

## Pregame component reconstruction

Use the preserved `model_rule_simulation_inputs.csv` and
`props_priced_clean.csv`.

### Team volume

For each team:
- predicted plays = frozen `rules_plays_est`;
- predicted pass rate = frozen `rules_pass_rate`;
- predicted pass attempts = plays * pass rate;
- predicted team rush attempts = plays * (1-pass rate).

For QB pass yards, when frozen `qb_pred_attempts` / `qb_pred_ypa` exist,
prefer those exact promoted QB component values over the generic team-volume
product.

### Individual receiving opportunity

Recreate the exact simulation target-share preparation:
- frozen `rules_tgt_share`;
- M38 WR hierarchy sharpening;
- team share cap / residual semantics identical to canonical simulation.

Expected player targets:
`pred_team_pass_att * final_player_target_probability`

### Individual rushing opportunity

Recreate the exact simulation rushing-share preparation:
- frozen `rules_rush_share`;
- top-5 rush-share filter;
- team share cap / residual semantics identical to canonical simulation.

Expected carries:
`pred_team_rush_att * final_player_rush_probability`

### Efficiency

Pregame:
- pass yards: `qb_pred_ypa` when available, otherwise frozen `rules_ypa`;
- rush yards: frozen `rules_ypc`;
- receiving yards: frozen `rules_ypt`;
- receptions: frozen `rules_catch_rate`.

Actual:
- QB actual YPA = passing yards / passing attempts;
- RB actual YPC = rushing yards / rushing attempts;
- receiving actual YPT = receiving yards / targets;
- actual catch rate = receptions / targets.

Zero-opportunity actuals remain valid zeros; undefined efficiency is reported
as unavailable rather than manufactured.

## Component attribution

For one-dimensional markets, compute descriptive contributions.

Passing:
- opportunity contribution:
  `(actual_attempts - predicted_attempts) * predicted_ypa`
- efficiency contribution:
  `actual_attempts * (actual_ypa - predicted_ypa)`

Rushing:
- opportunity contribution:
  `(actual_carries - predicted_carries) * predicted_ypc`
- efficiency contribution:
  `actual_carries * (actual_ypc - predicted_ypc)`

Receiving yards:
- opportunity contribution:
  `(actual_targets - predicted_targets) * predicted_ypt`
- efficiency contribution:
  `actual_targets * (actual_ypt - predicted_ypt)`

Receptions:
- opportunity contribution:
  `(actual_targets - predicted_targets) * predicted_catch_rate`
- efficiency contribution:
  `actual_targets * (actual_catch_rate - predicted_catch_rate)`

These decompose the frozen rules component, not the final ensemble mean. The
final-model error is retained separately.

For rush+rec:
- report rushing and receiving opportunity/efficiency contributions separately;
- do not force one scalar attribution if both materially contribute.

## Diagnostic labels

For pass/rush/rec/receptions:
- `OPPORTUNITY_DOMINANT` if |opportunity contribution| >= 1.5 *
  |efficiency contribution|;
- `EFFICIENCY_DOMINANT` if |efficiency contribution| >= 1.5 *
  |opportunity contribution|;
- `MIXED_OPPORTUNITY_EFFICIENCY` otherwise;
- `INSUFFICIENT_COMPONENT_EVIDENCE` if required components cannot be
  reconstructed.

For rush+rec:
- label from the combined absolute component contributions, while preserving the
  four separate contribution fields.

These are descriptive postmortem labels only.

## Additional flags

Report:
- actual inactive/DNP;
- exact identity route;
- whether the market line was closer than model;
- whether the player was selected by production;
- whether the row appeared in known user ticket construction where preserved
  evidence exists.

Do not infer a new betting rule from these flags.

## Required outputs

1. top-40 casebook row table;
2. attribution counts by position / market;
3. top opportunity-dominant misses;
4. top efficiency-dominant misses;
5. Watson case explicitly;
6. summary of what Week 5 changes in the research roadmap.

## Scientific boundary

No coefficient, threshold, source, feature or production change may be selected
because it explains Week 5.

Any new hypothesis must be separately frozen and tested on independent
historical/prospective data.
