# PRISM WR/TE Opportunity-Bias Conservation Bridge V1 — Frozen Source Audit

Date: 2026-10-18

Status: **FROZEN SOURCE / TOPOLOGY AUDIT — NO BRIDGE OUTCOME SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_AUDIT`

## 1. Why this audit exists

The signed component-bias signal qualified only:
- WR opportunity;
- TE opportunity.

WR efficiency and TE efficiency are CLOSED and excluded.

Before converting the qualified opportunity bias into any player projection
action, verify that the exact promoted historical authorities expose complete
team/position rooms that can absorb a named-player entitlement change while
preserving finite opportunity.

No target, receiving-yard, MAE, CRPS, or outcome improvement is scored here.

## 2. Qualified signal authority

Run:
- `39109423266` — SUCCESS;
- exact head `ac31b82907c416f76ffb078e9b2af8b6100a9ea6`;
- artifact `12128983909`;
- digest `sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`.

Qualified cells:
- `WR_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`;
- `TE_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`.

Closed cells:
- WR efficiency;
- TE efficiency.

The bridge audit must not inspect or revive the efficiency cells.

## 3. Exact full-room parent authorities

### WR

Promoted WR authority:
- run `34238301577`;
- artifact `10061328722`;
- artifact name `wr-r15-wr1-anchor-participation-v1`;
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`;
- file `wr_r15_confirmation_predictions.csv`.

Required projection fields:
- season;
- week;
- team;
- player_clean_key;
- `pred_targets`;
- `mc_rec_yards`.

Frozen seasons:
- 2023;
- 2024.

### TE

Promoted TE-R5P OOS authority:
- run `34152797603`;
- artifact `10029942404`;
- artifact name `te-r5p-production-contract-refit-v1`;
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`;
- file `te_r5p_oos_player_casebook.csv`.

Required projection fields:
- season;
- week;
- team;
- player_clean_key;
- `candidate_targets_r5p`;
- `candidate_rec_yards_r5p`.

Frozen seasons:
- 2024;
- 2025.

## 4. Exact signal-row join

Qualified opportunity signal rows come from:
`wr_te_player_signed_component_bias_signal_rows.csv`.

Join exactly on:
- position;
- season;
- week;
- team;
- player_clean_key.

Every source-eligible WR/TE opportunity signal row in the frozen seasons must
match exactly one full-authority row.

No fuzzy identity mapping.

## 5. Room definition

A room is:

`position + season + week + team`.

WR room:
all rows from the exact WR promoted authority for that team/week.

TE room:
all rows from the exact TE-R5P authority for that team/week.

This V1 does not move opportunity across WR and TE rooms and does not touch RB.

A later bridge, if authorized, must preserve the original room target sum
exactly to numerical tolerance.

## 6. Source-eligible bridge topology

A qualified signal row is mechanically bridge-eligible only when:
- exact authority join succeeds;
- predicted targets >0;
- predicted receiving yards >0;
- implied predicted YPT = predicted receiving yards / predicted targets is
  finite and >0;
- room has at least 2 authority players.

Single-player rooms are retained as diagnostics but cannot claim a zero-sum
within-room personalization effect.

## 7. Frozen support gates

Per position / scored season require:
- >=250 bridge-eligible supported target rows;
- >=50 distinct supported named players;
- exact signal-to-authority join coverage = 100%;
- duplicate authority identity rows = 0;
- duplicate signal identity rows = 0;
- nonfinite / nonpositive predicted target rows among matched signal rows = 0;
- nonfinite / nonpositive implied predicted YPT rows = 0.

These are topology/source gates only.

No gate may be lowered after this audit.

## 8. Report only source/topology facts

For each position/season report:
- full authority rows;
- room count;
- room-size p10 / median / p90 / max;
- signal-supported rows;
- exact matched rows;
- multi-player-room eligible rows;
- eligible distinct players;
- single-player-room supported rows;
- predicted target/YPT integrity counts.

Also report the qualified signal's deterministic multiplicative factor inventory
for later bridge design:

`bias_factor = exp(-hierarchical_signed_bias / predicted_receiving_yards)`.

This is **mechanical feasibility only**:
- min / p01 / p10 / median / p90 / p99 / max;
- finite count.

Do not apply the factor to targets.
Do not renormalize rooms.
Do not score outcomes.

The exponential form is inspected only because it is positive by construction
and is the exact no-tuning candidate for a later separately frozen bridge.

## 9. Anti-retest / exclusions

Do not:
- score receiving-yard or target outcomes;
- inspect current actual targets or receiving yards for improvement;
- tune a multiplier;
- clip or winsorize based on observed outcomes;
- revive efficiency;
- split by WR rank or TE role;
- use sportsbook or 2026 outcomes;
- mutate production.

## 10. If source-ready

A separate bridge freeze may test exactly one action:

1. multiply each supported player's predicted target entitlement by the frozen
   positive `bias_factor`;
2. leave unsupported room members at factor 1;
3. renormalize the entire same-position room back to its exact original target
   sum;
4. propagate through the parent predicted YPT to receiving yards;
5. compare against the unchanged promoted parent with player-aware OOS gates.

That action is not authorized by this source audit.

## 11. Integrity at freeze

- projection actions scored: **0**;
- outcomes scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 / Week-6 outcomes: **0**;
- sportsbook / OddsAPI: **0**;
- production changed: **false**.

Disposition:
`PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_AUDIT_FROZEN`.
