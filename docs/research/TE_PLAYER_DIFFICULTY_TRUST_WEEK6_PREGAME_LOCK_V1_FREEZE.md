# TE Player Difficulty Trust — Week 6 Pregame Lock V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE WEEK-6 TRUST LABEL CONSTRUCTION / RESEARCH ONLY**

Version:
`TE_PLAYER_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1`

Parent:
`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Parent authority:
- run `38821176218`
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`

## Objective

Freeze 2026 Week-6 TE trust labels **before Week-6 outcomes** using the exact
same named-player allocation-difficulty semantics that qualified historically.

This is a prospective shadow only.

It may not change:
- TE-R5P mean;
- TE-R5P distribution;
- target entitlement;
- sportsbook price;
- bet selection;
- production.

## Current Week-6 player authority

Use the already-frozen football-only Week-6 opportunity lock:
- run `38644002109` SUCCESS;
- artifact `11930748974`;
- digest `sha256:8aae43d895599bcb516ae79ecc0045dd4fad7a15aea48a8fe24c407f0b62a0f9`;
- status `WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_PACK_FROZEN`;
- exact roster source week = 6;
- prior Week-5 stats/snaps complete;
- target-week outcomes read = 0;
- sportsbook inputs = 0.

The Week-6 TE cohort is the exact TE-R5P parent trace in that artifact:
`wrte/te_r5p_week6_parent_trace.csv`.

Do not widen to reserve-only TE identities outside that TE-R5P modeled cohort.

## Historical allocation-error authority

Use two chronological sources.

### 1. Exact historical parent through 2025

Use the exact allocation-only result artifact:
- run `38656442803`;
- artifact `11936407975`;
- digest `sha256:7ded306cd0a049902130ccc5cab995231b192120c02679c3852757bc8df29cc0`.

Its row file contains the exact 2024/2025 TE allocation errors and stable
player identities already used to confirm:
`TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

Only TE rows may enter this Week-6 history.

### 2. 2026 Weeks 1–5 strict-prior replay

Reconstruct each completed 2026 Week 1–5 **pregame** TE-R5P target entitlement
using the current frozen football pipeline:
1. build that week's historical context using only information strictly prior to
   the target week;
2. materialize canonical target entitlement;
3. apply the frozen `TE_R5P_PRODUCTION_MODEL_V1`;
4. do not use sportsbook data;
5. do not use any Week-6 outcomes.

For each TE player-game define:

`pred_team_targets = clip(rules_plays_est, 50, 80) * clip(rules_pass_rate, 0.35, 0.75)`

`promoted_targets = pred_team_targets * te_r5p_entitlement_tgt_share`

`actual_share = actual_player_targets / actual_team_targets`

`allocation_actual_targets = pred_team_targets * actual_share`

`allocation_error = promoted_targets - allocation_actual_targets`

Target-game actual targets are labels only and may enter history **after** that
week's pregame projection has been built. Week 6 itself is never read.

## Frozen semantic parity gate

Before any Week-6 labels may be emitted, prove the team-target formula against
the exact recovered 2025 TE-R5P OOS authority.

Authority:
- TE fold recovery run `37782189538`, artifact `11552447166`;
- historical benchmark rebuild run `38657125777`, artifact `11937579170`.

For every matched 2025 TE authority row compare:

`b0_expected_targets / b0_target_probability`

against:

`clip(rules_plays_est,50,80) * clip(rules_pass_rate,.35,.75)`.

Frozen gate:
- >=1,100 matched 2025 TE rows;
- max absolute gap <= `1e-5`;
- zero ambiguous identities.

The pre-freeze audit observed 1,113 matched rows and max gap about
`7.55e-7`; the workflow must reproduce the gate independently.

## Frozen player history state

Combine exact 2025 parent errors with reconstructed 2026 W1–W5 errors.

For each Week-6 TE:
- same stable named-player identity only;
- sort chronologically;
- keep the trailing maximum 8 strictly-prior allocation errors;
- require at least 4 prior player-games;
- `player_mae = mean(abs(last <=8 allocation errors))`.

Do not:
- use target-share SD as a substitute;
- use pooled TE error in place of player history;
- use same-team-only history;
- impute unsupported players;
- fit a coefficient.

Unsupported current TEs remain explicitly `UNSUPPORTED` and receive no trust
bucket.

## Frozen Week-6 trust score

Among **supported Week-6 TE-R5P players only**:
- average percentile rank of `player_mae`;
- lower = more trustworthy;
- `HIGH_TRUST <= 0.25`;
- `LOW_TRUST >= 0.75`;
- otherwise `MIDDLE`.

Also report the contemporaneous supported-cohort median `player_mae` and
`player_mae / cohort_median` descriptively.

No alternate ranking or threshold.

## Integrity gates

The lock must fail closed unless:
1. parent trust disposition is `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`;
2. Week-6 opportunity pack is frozen and target-week outcomes read = 0;
3. exact Week-6 roster source week = 6;
4. exact Week-6 TE-R5P cohort is nonempty;
5. 2025 team-target semantic parity passes;
6. no reconstructed pregame feature for 2026 week W uses source week >= W;
7. reconstructed history stops at Week 5;
8. Week-6 outcomes read = 0;
9. sportsbook inputs = 0;
10. OddsAPI calls = 0;
11. fitted models = 0;
12. projection/distribution changes = 0;
13. production changed = false.

Report:
- current TE cohort rows;
- supported/unsupported count;
- HIGH/MIDDLE/LOW count;
- player-history rows by season;
- source max week;
- parity max gap;
- per-player history count and MAE;
- immutable CSV SHA256.

## Interpretation

This lock does **not** prove Week-6 predictive value.

It prospectively freezes the already-qualified historical trust rule so Week-6
outcomes can later provide a truly forward score.

No production promotion is authorized by creating the lock.

No paid OddsAPI.
No production mutation.
