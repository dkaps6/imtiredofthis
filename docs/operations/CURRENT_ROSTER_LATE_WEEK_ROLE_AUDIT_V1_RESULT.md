# Current Roster / Late-Week Role Audit V1 — Result

Status: `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`

This is an operational diagnostic result. No production behavior was changed.

## Authority

- Audit branch: `audit-current-roster-late-week-role-v1`
- Frozen audit plan commit: `848e5e44b1078c9d9dd7ab40ad0ccf16136f3754`
- Parent main handoff: `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Finding 1 — Ourlads detects unavailability, then discards it from the canonical role artifact

`scripts/providers/ourlads_depth.py` detects inactive state from red styling and injury-related cell classes and records `status`. However:

- `main(... include_inactive=True)` defaults to retaining those rows;
- Full Slate invokes the provider without `--exclude-inactive`, so the default is active;
- immediately before writing `data/roles_ourlads.csv`, every column containing `status` or `injury` is dropped.

Therefore `roles_ourlads.csv` can retain an unavailable player's depth role/model role while losing the source signal that the provider itself observed that player as inactive.

The artifact also carries no source URL, fetch timestamp, as-of timestamp, or freshness state.

## Finding 2 — the canonical role contract and pre-model role gate do not require availability/freshness provenance

`scripts/artifact_contracts.py` requires only `player, team, role, position, player_key` for `roles_ourlads`.

`validate_full_slate_pre_model_v1.py` requires team/player/player_clean_key/position/role, 32-team coverage, nonblank identities and uniqueness. It does not require:
- availability status;
- source/as-of timestamp;
- freshness relative to current slate;
- reconciliation against an inactive authority.

The canonical Full Slate workflow also does not invoke `validate_full_slate_pre_model_v1.py` directly.

Thus a structurally complete but semantically stale role artifact can pass the existing role checks.

## Finding 3 — weekly injury reports are not the same authority as official game-day inactives

`scripts/build/build_injuries_weekly.py` loads the requested week's nflverse injury report with NFL.com `/injuries/` fallback. It captures game/practice report status and provider health.

It does **not** acquire the NFL's official game-day inactive list.

The repository already proved this distinction in QB Migration 78. `scripts/backtest/audit_qb_official_inactive_availability.py` and hardened descendants reconstruct official game-day inactives and identify the live `https://www.nfl.com/inactives/` path. M78 explicitly states that future production acquisition must snapshot the live inactive page pre-kickoff.

That validated source contract was never promoted into the shared Full Slate roster/availability path.

## Finding 4 — current generic injury rules do not make definitive unavailable players unavailable

`simulation_rules._injury_limited()` treats OUT, DOUBTFUL, IR and PUP as limited.

For a non-alpha player, target and rush shares are merely multiplied by 0.50.

For an injured WR alpha, the vacancy rule explicitly leaves 50% of the alpha's target share on that injured player and redistributes the other half.

Thus an `OUT`, IR or PUP player can retain positive modeled opportunity. This is not a valid current-player availability contract.

`DOUBTFUL` is also conflated with definitive unavailable states; it is an uncertainty designation, not equivalent to official inactive.

## Finding 5 — promoted RB P3 constructs its active universe before definitive availability reconciliation

`scripts/run_rb_week1_no_odds.py` states that its active RB universe comes from Ourlads + schedule. It selects every Ourlads RB/HB/FB row, constructs internal rush targets, then joins optional injuries later.

No step filters definitive unavailable RBs from the active universe or re-ranks the depth chart after removal.

Therefore an inactive/OUT lead back can remain in the promoted P3 player universe and receive a positive projection. Any generic 50% injury reduction occurs after the stale universe has already been defined and is not equivalent to role/opportunity reallocation.

## Finding 6 — Full Slate build order does not reconcile role authority before PlayerForm/promoted opportunity

Current order is:
1. build Ourlads roles;
2. build schedule/team context;
3. build weekly injuries later;
4. build PlayerForm/model contexts;
5. build promoted RB P3 and other production components.

There is no canonical reconciliation step that combines depth role + definitive current availability into one authoritative active-player/role artifact before player opportunity is built.

## Finding 7 — prior M78 work supplies reusable source semantics, so a new scraper family is unnecessary

M78 hardened official-inactive parsing around:
- complete team sections;
- complete player bullet parsing;
- schedule-window coverage;
- separation of endpoint reachability from a validated game-day payload;
- pre-kickoff snapshot semantics.

The operational fix should reuse/adapt this source contract rather than invent another data source or treating weekly injury reports as official inactives.

## Disposition

`CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`

The confirmed failure mode is architectural:

> current depth role and current availability are separate, incompletely reconciled authorities; the depth artifact can suppress its own inactive signal, the weekly injury report does not provide official game-day inactive authority, and downstream rules can retain positive opportunity for definitively unavailable players.

This can produce stale player role/entitlement in live Full Slate projections, especially around late-week/game-day changes.

No production fix is authorized by this diagnostic alone. A separately frozen operational fix plan must define authority precedence, provenance/freshness, definitive-unavailable semantics, depth-role re-ranking, conservation/reallocation, timing-aware official-inactive gates, and Full Slate verification.
