# Current Player Availability 35-Gate — Run1 QB C2 Coverage Repair Lock

Status: `LOCKED_BEFORE_MECHANICAL_REPAIR_REGRESSION`

## Preserved Run1
- run `34459655725`
- job `102814178762`
- head `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`
- disposition `MECHANICAL_FAILURE_NO_DECISION`
- no frozen integration gate was evaluated.
- exact exception: `QB C2 starter authority must cover 32 teams, got rows=30`.

## Frozen repair authority
- repair plan commit `760fbcafec3df636b671b49c16a6d1d04c134162`
- protected QB C2 source blob `7b677470b27b6776055c75c924a0ddf22d724a44`
- QB C2 coverage transformer blob `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- shared eligible-team helper blob `77b591e431378ec984c51e8a032262e673d4c843`

The transformer may replace only the current starter-audit unconditional 32-team coverage assertion with the already-frozen availability-aware team-set validator. Legacy mode remains exactly 32 teams. Explicit `ACTIVE_ROLES_CSV` mode requires exact equality to the certified eligible team set.

The QB C2 state-context 32-team source-integrity assertion remains untouched. Starter selection, official authority precedence, Ourlads fallback, C2 parameters/distribution logic, all other protected football science, sportsbook boundary and all 35 integration gates remain unchanged.

A certification retry is allowed only after the dedicated regression workflow passes. Because Run1 never evaluated the gates, the retry remains eligible to become the first valid immutable 35-gate result.
