# WR Player Efficiency-Difficulty Trust — Week 6 Pregame Lock V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE WEEK-6 EFFICIENCY-TRUST LABEL CONSTRUCTION / RESEARCH ONLY**

Version:
`WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1`

Historical parent:
`WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Historical authority:
- run `38826418315` SUCCESS;
- exact scored head `c15f70f3bacf9d0d7f301d334419841ad6d9d492`;
- artifact `12015303274`;
- digest `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`.

The historical V1 proved that WR same-player efficiency difficulty adds final
receiving-yard reliability information after strictly-prior opportunity
difficulty is controlled.

TE did not qualify and receives no Week-6 efficiency-trust lock from this V1.

## Objective

Freeze a **separate WR efficiency-trust label** for Week 6 before any Week-6
outcomes are read.

This lock does not change:
- WR-R15 mean;
- WR distribution width or shape;
- target entitlement;
- opportunity trust;
- sportsbook comparison;
- production.

Opportunity trust and efficiency trust remain separate axes.

## Current Week-6 cohort authority

Use the already-frozen Week-6 WR opportunity-trust artifact:
- run `38823520954`;
- artifact `12012884262`;
- digest `sha256:e5bc113e507ded651ecc29080c7fdbe4fd55182d1d039d0dd14057ada4b7bb38`.

Required files:
- `wr_week6_trust_lock.csv` — current 152-WR cohort and already-frozen
  opportunity-trust labels;
- `wr_week6_reconstructed_2026_opportunity_errors.csv` — exact strict-prior
  W1-W5 WR-R15 pregame reconstruction.

The existing opportunity-trust lock must report:
- status `WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN`;
- Week-6 outcomes read = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- production unchanged.

## Settled W1-W5 realized receiving yards

Use only:
- no-odds Week-6 Full Slate run `38790425676`;
- artifact `11999618082`;
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`;
- file `data/player_game_logs.csv`.

This source must contain:
- 2026 Weeks 1-5 only for settled current-season rows;
- no Week-6-or-later 2026 player outcomes;
- player_clean_key;
- team;
- targets;
- receiving_yards.

No sportsbook data is used.

## Frozen efficiency-error reconstruction

For each row in the already-frozen W1-W5 WR reconstruction:

- `pred_ypt` comes from that target week's strict-prior WR-R15 projection;
- `actual_targets` is already frozen in the parent reconstruction;
- `actual_rec_yards` is joined from the no-odds player-game log.

Define:

`efficiency_error = actual_targets * pred_ypt - actual_rec_yards`

This matches the historical WR mechanism decomposition sign convention:

`total_error = opportunity_error + efficiency_error`.

Do not recompute or modify the original W1-W5 pregame projection.

## Frozen player state

For each Week-6 WR:
- same `player_clean_key` only;
- use only completed 2026 Weeks 1-5;
- trailing maximum 8 qualifying games;
- minimum 4 prior games;
- `player_efficiency_mae = mean(abs(efficiency_error))`.

No 2025 backfill.

Players with fewer than four qualifying 2026 games are:
`UNSUPPORTED`.

Among supported Week-6 WRs:
- `week_pool_efficiency_mae = median(player_efficiency_mae)`;
- `efficiency_fragility_score = rank(player_efficiency_mae, method="average", pct=True)`;
- HIGH_EFFICIENCY_TRUST <= 0.25;
- LOW_EFFICIENCY_TRUST >= 0.75;
- otherwise MIDDLE.

These are reliability labels only.

## Opportunity-trust separation

Carry the already-frozen Week-6 opportunity-trust score/bucket into the output
for auditability, but do not combine the two scores and do not construct a
composite bet selector.

No 2D combination rule is authorized by this lock.

## Frozen integrity gates

Fail closed unless:
1. historical parent disposition is WR-qualified;
2. historical parent TE disposition remains non-qualified;
3. Week-6 opportunity-trust parent status is frozen;
4. current WR cohort = 152;
5. W1-W5 pregame reconstruction source max week = 5;
6. no Week-6-or-later 2026 outcomes exist in player-game logs;
7. every history join is one-to-one on season/week/team/player_clean_key;
8. historical W1-W5 actual-target value agrees with player-game logs within
   numerical tolerance;
9. efficiency-error values are finite;
10. supported players have >=4 qualifying games;
11. sportsbook inputs = 0;
12. OddsAPI calls = 0;
13. Week-6 outcomes read = 0;
14. models fit = 0;
15. projection/distribution values changed = 0;
16. production unchanged.

Report:
- current WR cohort;
- supported/unsupported count;
- efficiency-trust bucket counts;
- reconstructed W1-W5 rows;
- pool efficiency MAE;
- each player's prior-game count and efficiency MAE;
- existing opportunity-trust label;
- row CSV SHA256.

## Status on success

`WEEK6_WR_EFFICIENCY_TRUST_PREGAME_LOCK_FROZEN`

## Boundaries

This lock does not:
- change WR-R15;
- change opportunity trust;
- combine trust scores;
- choose an over/under;
- choose a bet;
- use live odds;
- use Week-6 outcomes.

Prospective scoring may occur only after Week 6 settles.

No paid OddsAPI.
No production mutation.
