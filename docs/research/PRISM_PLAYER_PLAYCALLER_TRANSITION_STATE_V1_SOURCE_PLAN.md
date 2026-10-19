# PRISM Named-Player Offensive Playcaller Transition State V1 — Frozen Source Audit

Date: 2026-10-18

Status: **FROZEN SOURCE / PARITY AUDIT ONLY / NO PLAYER OUTCOME SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM is an individual-player projection model.

The broader Current Role-State Arbitration frontier asks:

> What source-valid football evidence tells PRISM that this named player's
> current role/state is genuinely different from his longer identity now,
> before the target game?

This V1 audits one genuinely distinct pregame regime object:

> Has this named player's team entered a different primary offensive
> playcalling regime before the target game, and can that regime transition be
> reconstructed with exact historical/live semantic parity at
> named-player + target-game grain?

A primary playcaller change can alter how a team chooses to distribute and
sequence opportunity even when the player, QB, teammates and recent usage have
not yet accumulated enough post-change history to show the new state.

This is only a source/parity audit.

It does not score whether a playcaller change improves targets, carries,
receptions, yards or touchdowns.

## 2. Why this is distinct from closed/protected science

This source object is not:

- raw recent-share replacement — CLOSED;
- WR/TE target-share trajectory — protected;
- RB carry/snap state — protected;
- Stage-1 active-role participation — QUALIFIED;
- Stage-2 active-set allocation — qualified/partial;
- teammate-availability Role Elasticity V1 — SOURCE BLOCKED;
- RB/TE depth-role transition — SOURCE BLOCKED/occupied family;
- same-QB target allocation — CLOSED;
- same-QB target depth — CLOSED;
- same-QB target CPOE — CLOSED;
- injury-report persistence — CLOSED;
- return-from-injury participation restoration — qualified but separate.

Existing Migration 68 playcaller work was QB residual research. It did not
create a named skill-player role-state rule and may not be cited as predictive
qualification for this V1.

No M68 outcome result is reused.

## 3. Frozen source authority

Historical source family:

`scripts/backtest/build_qb_playcaller_opening_leverage.py`

Only these embedded, manually verified source objects may be inspected:

- `SEASON_OPENING_CALLER`;
- `CALLER_OVERRIDES`;
- public source URL metadata attached to those mappings.

Historical years admitted:
- 2024;
- 2025.

The historical audit uses only playcaller identity/effective-week metadata.

It does not use:
- M68 QB predictions;
- M68 residuals;
- opening-script outcomes;
- playoff leverage;
- target-game player outcomes.

## 4. Frozen historical state construction

For every team + week in 2024 and 2025:

`current_playcaller = caller_for(season, week, team)`

`prior_playcaller = caller_for(season, prior_team_game_week, team)`

`playcaller_transition =
 current_playcaller != prior_playcaller`

Transition effective week comes only from the frozen documented override
metadata.

Season-opening caller changes relative to the prior season are allowed only as
a Week-1 regime boundary descriptor. They are not called "midseason
transitions."

No player opportunity is inferred from the playcaller label.

## 5. Named-player attachment

Historical named-player membership source:
- nflverse/nflreadpy weekly rosters where available;
- exact target season + week + team;
- stable player identifier required;
- QB/RB/WR/TE admitted.

Attach the team-week playcaller state to the named player only by exact:

`season + week + team`

Persist:
- stable_id;
- family;
- team;
- target week;
- current playcaller;
- prior team-game playcaller;
- transition flag;
- transition effective week where applicable.

No target-game snaps, carries, targets, receptions, yards or TDs are used to
define the state.

## 6. Historical support gates

Adjudicate 2024 and 2025 separately.

Require all:

1. all 32 NFL teams have a nonblank season-opening primary playcaller;
2. every documented in-season override has:
   - valid team;
   - finite effective week;
   - nonblank new caller;
   - nonblank public source URL;
3. playcaller identity is deterministic for every regular-season team-week;
4. duplicate team/week playcaller identities = 0;
5. weekly roster team-week coverage >= 95%;
6. stable named-player identity coverage >= 99% for admitted QB/RB/WR/TE rows;
7. exact team-week playcaller attachment coverage >= 99%;
8. at least one documented in-season playcaller transition exists in the
   season;
9. every transition is source-effective before the game it first labels;
10. target-game player outcomes used = 0;
11. target-game player snaps used = 0;
12. player recent-share fields used = 0;
13. sportsbook/OddsAPI used = 0;
14. coefficients fit = 0;
15. thresholds searched = 0;
16. production changed = false.

Do not lower these gates after execution.

## 7. Prospective 2026 semantic-parity gate

Historical source readiness is insufficient by itself.

Because PRISM needs a pregame current-state object, 2026 must have the same
semantic object:

> one primary offensive playcaller per team-week, with public source authority
> and effective-week chronology.

The 2026 probe must report:

- whether a complete 32-team season-opening playcaller mapping exists in the
  repository or an immutable checked-in source artifact;
- whether documented 2026 in-season playcaller handoffs can be represented by
  exact effective week;
- whether the same `caller_for(season, week, team)` semantics work for 2026;
- whether a target week can be reconstructed without using game outcomes;
- whether the current mapping is complete through the latest known 2026 week.

Prospective parity requires all 32 teams and all known handoffs.

A partial list, generic offensive-coordinator field, or current staff page is
not enough.

A public report that a playcaller changed midseason is evidence that effective
week matters; it does not by itself complete the 32-team mapping.

## 8. Prospective dispositions

If historical gates pass but 2026 complete semantic parity is absent:

`PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_BLOCKED_PROSPECTIVE_PARITY`

If historical and 2026 parity both pass:

`PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_READY`

If historical gates fail:

`PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_BLOCKED`

## 9. If source-ready

Do not change projections.

Freeze a separate market-specific predictive contract before scoring.

The first legal predictive question must remain named-player specific, e.g.:

> For this named player's exact market, does a source-certified offensive
> playcaller transition change his opportunity state beyond his own strictly
> prior role identity and already-qualified finite-opportunity mechanisms?

The future test must:
- use 2024 for development;
- use 2025 for independent confirmation;
- keep market seams separate;
- preserve teammate competition / finite opportunity;
- use no 2026 outcomes;
- fit no rule before its freeze.

## 10. If source-blocked

Fail closed.

Do not:
- infer playcaller from coordinator title;
- assume head coach calls plays;
- use target-game offensive tendencies to infer who called the game;
- forward-fill a stale caller through a documented handoff;
- use a partial 2026 list as complete parity;
- substitute current player usage;
- fit a predictive player rule on historical-only semantics.

Move only to another genuinely distinct named-player current role/state source
object.

## 11. Integrity at freeze

Before this freeze:
- player outcomes scored under this mechanism: **0**;
- player opportunity relationships inspected: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 player outcomes used: **0**;
- sportsbook/OddsAPI inputs used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYER_PLAYCALLER_TRANSITION_STATE_V1_SOURCE_AUDIT_FROZEN`.
