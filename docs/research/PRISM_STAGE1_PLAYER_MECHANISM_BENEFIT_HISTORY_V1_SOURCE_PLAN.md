# PRISM Stage-1 Player Mechanism Benefit History V1 — Frozen Source Audit Plan

Date: 2026-10-18

Status: **FROZEN SOURCE-ONLY AUDIT / NO STAGE-1 PERSONALIZATION SCORE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT`

## 1. Frozen question

Stage-1 Active-Role Participation is already a qualified player-aware mechanism:
it shrinks a named player's prior participation history toward the family prior.

The new question is one level higher:

> Do we have enough strictly-prior same-player OOS history of **Stage-1's own
> predictive benefit** to test whether the named player should receive different
> authority from this otherwise-qualified Stage-1 player-identity mechanism?

This is source readiness only.

No player-specific Stage-1 authority rule is scored here.

## 2. Canonical parent authority

Stage-1 mechanism:
- run `38907459996` — SUCCESS;
- job `116994537005` — SUCCESS;
- artifact `12049120604`;
- artifact name
  `player-active-role-participation-mechanism-v1-38907459996`;
- digest
  `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`;
- disposition `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Immutable row authority:

`player_active_role_mechanism_scored_rows.csv`

with 11,166 OOS player rows.

## 3. Frozen mechanism-benefit identity

For each already-OOS parent row:

`baseline_brier = (y - p_baseline)^2`

`candidate_brier = (y - p_candidate)^2`

`delta_brier = baseline_brier - candidate_brier`.

Positive:
the Stage-1 named-player participation mechanism helped that OOS row.

Negative:
it hurt.

Also reconstruct binary log loss with probabilities clipped only for numerical
logging at `1e-12`; this is secondary source state, not a tuning target.

No parent formula is changed.

## 4. Families

Audit separately:
- QB;
- RB;
- WR;
- TE.

No family pooling for readiness.

## 5. Strict chronology

For a target historical row, same-player mechanism-benefit history may use only:
- same family;
- same stable player ID;
- an earlier season, or same season with strictly smaller week.

Same-week rows are excluded.

The target row never enters its own pregame state.

Chronology violations must equal 0.

## 6. Frozen source outputs

Per target row record:
- season / week / game / team / opponent;
- family;
- stable player ID;
- `p_baseline`;
- `p_candidate`;
- parent player-history support;
- prior Stage-1 mechanism-row count;
- prior mean / median `delta_brier`;
- prior share of mechanism rows with `delta_brier > 0`;
- prior mean `delta_logloss`.

Support reporting bins:
- Tier 0: 0;
- Tier 1: 1–2;
- Tier 2: 3–5;
- Tier 3: 6–9;
- Tier 4: 10+.

These bins are descriptive only.

## 7. Frozen source-readiness gates

All identity / duplicate / chronology gates are mandatory:
- stable player identity >=99%;
- duplicate family/player/week/team identities = 0;
- chronology violations = 0;
- >=28 teams in both target seasons.

### QB

2024:
- >=550 target rows with >=3 prior same-player mechanism rows;
- >=65 players with >=3.

2025:
- >=700 rows;
- >=70 players.

### RB

2024:
- >=900 rows;
- >=105 players.

2025:
- >=1,100 rows;
- >=120 players.

### WR

2024:
- >=1,300 rows;
- >=175 players.

2025:
- >=1,700 rows;
- >=190 players.

### TE

2024:
- >=850 rows;
- >=105 players.

2025:
- >=1,100 rows;
- >=110 players.

These gates were frozen from source inventory only.
They may not be lowered after any predictive signal is scored.

## 8. Later role/fallback object

If source-ready, the first later meta-signal should treat the existing
Stage-1 family prior as the broader fallback.

Potential clean role context already present before the target:
- family;
- current parent `p_baseline`.

Do not invent outcome-derived archetypes in the source audit.

The first later experiment should ask whether same-player prior Stage-1 utility
adds information beyond the broader family/baseline state.

## 9. Explicit prohibitions

This source audit must not:
- change `p_candidate`;
- select a Stage-1 authority threshold;
- fit a per-player participation model;
- retune Stage-1 prior strength/window;
- score 2026;
- read/grade Week-6 outcomes;
- use sportsbook/OddsAPI;
- reopen Stage-2 closed/source-blocked families;
- change production.

## 10. Possible dispositions

Per family:
- `<FAMILY>_STAGE1_PLAYER_BENEFIT_HISTORY_SOURCE_READY`;
or
- `<FAMILY>_STAGE1_PLAYER_BENEFIT_HISTORY_SOURCE_BLOCKED`.

Overall:
- READY_ALL;
- READY_PARTIAL;
- SOURCE_BLOCKED.

## 11. If source-ready

Freeze a separate hierarchical **future Stage-1 mechanism-utility signal**
before scoring.

Do not assume that a qualified utility signal implies the same hard authority
integration that worked for RB receiving Stage-2.

The RB-rushing Stage-2 result already proves those are distinct questions.

## 12. Integrity at freeze

- predictive Stage-1 personalization scores: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT_FROZEN`.
