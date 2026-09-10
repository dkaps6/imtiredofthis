# Current Player Availability 35-Gate — Run1 QB C2 Coverage Mechanical Repair Plan

Status: `FROZEN_BEFORE_REPAIR / MECHANICAL_ONLY / NO_GATE_RESULT`

## Preserved failure

- certification lock head: `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`
- run: `34459655725`
- job: `102814178762`
- failure stage: baseline promoted football-stack execution, BEFORE any frozen gate 1-34 evaluation
- exact exception: `RuntimeError: QB C2 starter authority must cover 32 teams, got rows=30`
- disposition: `MECHANICAL_FAILURE_NO_DECISION`

The run had already passed frozen-blob verification, protected-model boundary verification, immutable candidate artifact/digest verification, candidate 30-team/15-game assertions, eligible-team seam application and frozen fixture construction. M38 -> TE-R5P -> WR-R15 also materialized successfully on 437 players / 30 teams / 15 games with their frozen conservation gates intact before QB C2 reached its legacy coverage assertion.

## Root cause

`annotate_primary_qbs()` in protected `scripts/modeling/qb_c2_production_adapter_v1.py` validates its current starter-authority audit with an unconditional 32-team row/team count. The availability candidate intentionally excludes the already-kicked NE-SEA game and therefore passes a certified 30-team football universe downstream.

This is the same class of current-slate coverage assumption already frozen and regression-tested for the full-universe and R26 adapters. It is not a QB C2 model, starter-selection, authority-priority, feature, coefficient, state-context or distribution failure.

## Frozen minimum repair

Create a separate ephemeral certification transformer. It may change ONLY the final current-team coverage assertion in `annotate_primary_qbs()`:

- legacy mode, when no explicit `ACTIVE_ROLES_CSV` is configured: preserve the existing exact 32-team requirement;
- availability mode, when explicit `ACTIVE_ROLES_CSV` is configured: require the observed starter-authority audit team set to equal exactly the certified eligible-team set using the already-frozen `scripts.utils.eligible_team_set_v1.validate_current_team_set` helper.

The transformer MUST NOT change:
- QB starter ranking;
- official starter authority precedence;
- Ourlads QB1 fallback;
- `qb_projection_eligible` assignment;
- QB C2 candidate parameters or distribution selection;
- QB state-context values or the existing 32-team source-context file integrity assertion;
- any non-QB component;
- any of the 35 frozen integration gates.

The existing `_load_state_context()` 32-team assertion remains unchanged because it verifies a source-context artifact that legitimately contains all 32 teams; the current eligible subset is selected by the football frame during application.

## Retry rule

After a transformer regression demonstrates both legacy 32-team behavior and exact explicit eligible-team behavior, pin its blob and protected QB C2 source blob in the certification workflow, update the implementation lock, and retry. The retry remains eligible to become the first valid 35-gate result because Run1 never evaluated gates 1-34.
