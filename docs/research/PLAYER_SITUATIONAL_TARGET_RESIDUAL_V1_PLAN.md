# Player Situational Target Residual V1 — Frozen Historical Diagnostic

**STATUS: FROZEN BEFORE RESULT. NO FITTED CANDIDATE. NO PRODUCTION CHANGE.**

## Parent evidence

Player-level opportunity residuals persist after promoted entitlement:

- WR: `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
  - run `37634044900`
  - pooled opportunity signed rho `0.1699`
  - pooled opportunity-difficulty rho `0.3183`

- TE: `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`
  - run `37631011196`
  - pooled opportunity signed rho `0.2500`
  - pooled opportunity-difficulty rho `0.3936`

The live/source audit then found three player-specific target-earning contexts that are materially nonredundant with ordinary overall target share:

- THIRD_DOWN
- RED_ZONE
- TWO_MINUTE

Authority:
- `PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_READY`
- run `37635210417`
- artifact `11489645374`
- digest `sha256:6c83ba1973cc7c5961e8f277402cb50a143db1c73ec6305c6d56a5c652ad6da5`

EARLY_DOWN is explicitly excluded because it was redundant with ordinary target share.

## Question

For the **same individual player**, does strictly-prior situational target earning explain signed target/opportunity error left behind by the exact promoted WR/TE entitlement model?

This is not a position-level test.

The player is the unit of state:
- his own target earning;
- on his current target team;
- in specific football situations;
- measured strictly before the target game.

## Exact promoted authorities

### WR

Run `34238301577`
Artifact `10061328722`
Digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

File:
`backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv`

Variant:
`WR_R15_WR1_ANCHORED_PARTICIPATION`

Target seasons:
- 2023
- 2024

Opportunity error:

`WR opportunity_error = (pred_targets - actual_targets) * (mc_rec_yards / pred_targets)`

### TE

Run `34152797603`
Artifact `10029942404`
Digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`

File:
`data/backtests/te_r5p_production_contract_refit_v1/te_r5p_oos_player_casebook.csv`

Target seasons:
- 2024
- 2025

Opportunity error:

`TE opportunity_error = (candidate_targets_r5p - targets) * (candidate_rec_yards_r5p / candidate_targets_r5p)`

Negative error means the promoted model underpredicted the player's target opportunity.

## Identity bridge

Use free nflreadpy weekly player stats only to bridge stable GSIS ID to the canonical `player_clean_key` used by the exact authority.

For seasons 2022-2025:
- normalize weekly stats with the canonical `player_form_v2._normalize_weekly`;
- build `(season, player_clean_key) -> player_id`;
- a target authority player is eligible only when the target-season key maps to exactly one non-empty stable player ID.

Fail/omit ambiguous mappings. Do not use fuzzy PBP display-name matching as scientific authority.

Report mapping coverage separately for WR and TE.

## Historical situational state

Source:
- free nflreadpy regular-season PBP from 2022-2025.

For each target authority player-game:

1. use only target-team offensive PBP strictly before the target season/week;
2. identify the latest up to **8 completed target-team games**;
3. require at least **4** of those team games contain at least one target to the target player;
4. require at least **5 total target events** to the player across the history window.

For those same team games compute:

`overall_share = player ALL targets / team ALL targets`

and for each frozen context:

`context_share = player context targets / team context targets`

`context_delta = context_share - overall_share`

Context definitions are inherited unchanged from the source audit:

- THIRD_DOWN: down == 3
- RED_ZONE: yardline_100 <= 20
- TWO_MINUTE: half_seconds_remaining <= 120

A context row is scoreable only when:
- team context denominator >= **5** targets within the history window.

No target-game PBP is used in the feature.

## Frozen directional hypothesis

If a player earns a disproportionately large share of high-leverage situational targets relative to his ordinary overall target share, a participation/room-entitlement model may understate his next-game target opportunity.

Therefore the predeclared sign is:

`context_delta ↑ -> opportunity_error more negative`

Expected Spearman sign:
- **negative**

No opposite-sign rescue is permitted.

## Diagnostics

For each context report separately:

### WR
- 2023 rho(context_delta, opportunity_error)
- 2024 rho
- pooled WR rho
- rows / players
- player-cluster bootstrap P(rho < 0)

### TE
- 2024 rho
- 2025 rho
- pooled TE rho
- rows / players
- player-cluster bootstrap P(rho < 0)

### Combined
- pooled WR+TE rho
- player-position cluster bootstrap P(rho < 0)

Bootstrap:
- 5,000 replicates
- seed `20261007`
- cluster by `position + player_clean_key`
- preserve all target rows for each sampled player cluster.

## Support floors

For a context to be scientifically eligible:

WR:
- >=500 scoreable rows in 2023;
- >=500 scoreable rows in 2024;
- >=100 distinct WRs pooled.

TE:
- >=250 scoreable rows in 2024;
- >=250 scoreable rows in 2025;
- >=50 distinct TEs pooled.

Identity mapping coverage on the exact parent authority must be >=95% for each position.

## Context confirmation gate

A context is `REPLICATED_PLAYER_ROLE_SIGNAL` only if all are true:

1. support floors pass;
2. WR 2023 rho < 0;
3. WR 2024 rho < 0;
4. TE 2024 rho < 0;
5. TE 2025 rho < 0;
6. pooled WR rho <= -0.05;
7. pooled TE rho <= -0.05;
8. WR player-cluster bootstrap P(rho < 0) >= 0.95;
9. TE player-cluster bootstrap P(rho < 0) >= 0.95;
10. combined player-position cluster bootstrap P(rho < 0) >= 0.99.

This deliberately requires cross-position replication. A WR-only or TE-only subgroup cannot rescue V1.

## Final disposition

`PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL_CONFIRMED`
if at least one of THIRD_DOWN / RED_ZONE / TWO_MINUTE passes the full context confirmation gate.

`NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL`
if identity/support pass but zero contexts confirm.

`PLAYER_SITUATIONAL_TARGET_ROLE_SOURCE_LIMITED`
if identity or support prevents all three contexts from being judged.

## Anti-rescue

After results are exposed, do not search:
- alternate 4/6/10-game windows;
- different context denominator floors;
- different target-history floors;
- early-down state;
- fourth down;
- goal-to-go;
- score differential buckets;
- WR-only rescue;
- TE-only rescue;
- WR1/WR2+ rescue;
- player residuals as features;
- fitted weights/blends;
- sportsbook-conditioned variants.

A future candidate is authorized only from a context that clears the frozen cross-position replication gate.

Models fit: **0**
Sportsbook inputs: **0**
2026 outcomes: **0**
Production mutations: **0**
