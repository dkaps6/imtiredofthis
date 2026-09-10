# Current Player Availability 35-Gate — Run2 QB C2 Primary Coverage Mechanical Repair Plan

Status: `FROZEN_BEFORE_REPAIR / MECHANICAL_ONLY / NO_GATE_RESULT`

## Preserved failure

- certification relock head: `1d38953995842aa0236edda7124a79665d0b8628`
- run: `34460227422`
- job: `102816052394`
- failure stage: baseline promoted football-stack execution, BEFORE any frozen gate 1-34 evaluation
- exact exception: `RuntimeError: QB C2 production adapter did not resolve exactly one primary QB per team`
- disposition: `MECHANICAL_FAILURE_NO_DECISION`

Run2 verified the Run1 starter-audit repair: the availability-aware QB starter-authority transform applied successfully, immutable candidate staging passed, fixture construction passed, and M38 -> TE-R5P -> WR-R15 again materialized on the exact certified 30-team / 15-game universe before this later assertion stopped QB C2.

## Root cause

Inside `apply_qb_c2_selector()` the already-selected `primary` frame is validated by a second unconditional legacy condition:

```python
primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()
if len(primary) != 32 or primary["team"].nunique() != 32:
    raise RuntimeError("QB C2 production adapter did not resolve exactly one primary QB per team")
```

The upstream `annotate_primary_qbs()` authority audit now correctly validates against the certified eligible team set, so this second check is another current-slate coverage assumption rather than evidence of duplicate or missing primary selection.

## Frozen minimum repair

Create a **separate** ephemeral transformer; do not modify the already-frozen Run1 transformer. It may replace ONLY the second current-primary coverage assertion with the same pre-frozen `validate_current_team_set` helper:

- legacy/no explicit `ACTIVE_ROLES_CSV`: still require exactly 32 unique teams;
- explicit current availability: require exactly one observed primary row for each team in the certified eligible-team set and no extra teams.

Because `primary` is selected before validation, this transformer MUST NOT change:
- `qb_projection_eligible` assignment;
- starter ranking or starter identity;
- official starter authority precedence;
- Ourlads QB1 fallback;
- QB C2 state-context source or its 32-team integrity assertion;
- C2 model parameters, candidate selection or distributions;
- M38, TE-R5P, WR-R15, R26, R22, P3;
- sportsbook boundary;
- any of the 35 frozen integration gates.

## Retry rule

A certification retry is allowed only after a dedicated regression proves that applying the Run1 transformer followed by this Run2 transformer:
1. preserves both transformed current-team guards;
2. preserves the separate 32-team state-context source-integrity guard;
3. compiles the protected QB C2 module;
4. retains legacy 32-team and explicit exact-eligible helper behavior.

Run1 and Run2 both evaluated zero frozen gates, so a post-regression retry remains eligible to become the first valid immutable 35-gate result.
