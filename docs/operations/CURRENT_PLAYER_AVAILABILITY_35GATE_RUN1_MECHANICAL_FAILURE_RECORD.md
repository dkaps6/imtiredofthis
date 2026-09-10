# Current Player Availability 35-Gate — Run1 Mechanical Failure Record

Disposition: `MECHANICAL_FAILURE_NO_DECISION`

## First certification execution
- branch: `ops-current-player-availability-35gate-cert-v1`
- locked head: `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`
- run: `34459655725`
- job: `102814178762`
- workflow conclusion: failure
- frozen gates evaluated: **0 / 35**
- first failing step: `Execute baseline promoted football stack without sportsbook`
- exact exception: `RuntimeError: QB C2 starter authority must cover 32 teams, got rows=30`

The run is not a scientific or integration FAIL. It stopped before the gate evaluator and therefore cannot consume the immutable first-valid-result slot.

## Evidence before failure
The same run successfully verified:
- frozen implementation blobs;
- protected model/research boundary;
- immutable candidate Run `34447900206` / Artifact `10140425929` / digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`;
- 15 eligible games / 30 certified teams / 437 eligible current-role rows / sportsbook inputs 0;
- existing full-universe/R26 eligible-team seam;
- all three predeclared fixture inputs.

Before QB C2 failed, the real baseline stack had also materialized M38 -> TE-R5P -> WR-R15 for 437 players / 30 teams / 15 games with projection-neutral M38 baseline, TE pool conservation, WR1 anchor preservation, WR2+ pool conservation, WR-room mass conservation, non-WR preservation and team-total entitlement preservation all true; sportsbook inputs were false.

## Root cause and frozen repair
The current QB starter-authority audit had an unconditional legacy 32-team coverage assertion despite the certified football universe intentionally excluding already-kicked NE-SEA.

Repair plan: `docs/operations/CURRENT_PLAYER_AVAILABILITY_35GATE_RUN1_QB_C2_COVERAGE_MECHANICAL_REPAIR_PLAN.md`

Repair-plan commit: `760fbcafec3df636b671b49c16a6d1d04c134162`

Repair transformer:
- `scripts/operations/apply_current_availability_qb_c2_eligible_team_seam_v1.py`
- blob `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- protected QB C2 source blob `7b677470b27b6776055c75c924a0ddf22d724a44`

The repair changes only starter-audit current-team coverage validation. Starter ranking/selection, official authority precedence, Ourlads fallback, QB C2 parameters/distribution logic and the 32-team state-context source-integrity assertion remain unchanged.

## Repair regression
- run: `34460044387`
- job: `102815456345`
- head: `3825235af0442e7868f474e905ea11ad1214432e`
- conclusion: `success`

Regression demonstrated:
- legacy/no-explicit-availability mode accepts exactly 32 teams and rejects 30;
- explicit availability mode accepts exactly the certified 30-team set;
- explicit mode rejects both extra and missing teams;
- transformed QB module compiles;
- the state-context 32-team source guard remains present.

A certification retry is therefore authorized under the pre-frozen mechanical-failure policy. No frozen integration gate, scientific parameter or production component has been changed.
