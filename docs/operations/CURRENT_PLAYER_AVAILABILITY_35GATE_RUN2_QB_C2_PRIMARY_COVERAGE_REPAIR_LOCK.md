# Current Player Availability 35-Gate — Run2 QB C2 Primary Coverage Repair Lock

Status: `LOCKED_BEFORE_SEQUENTIAL_MECHANICAL_REPAIR_REGRESSION`

## Preserved Run2
- run `34460227422`
- job `102816052394`
- head `1d38953995842aa0236edda7124a79665d0b8628`
- disposition `MECHANICAL_FAILURE_NO_DECISION`
- frozen gates evaluated `0/35`
- exact exception: `QB C2 production adapter did not resolve exactly one primary QB per team`.

## Frozen repair authority
- repair plan commit `2518f8366a8acbad5eb9d91de6134a1923a2381b`
- protected QB C2 source blob `7b677470b27b6776055c75c924a0ddf22d724a44`
- Run1 starter-audit transformer blob `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- Run2 primary-frame transformer blob `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`
- shared eligible-team helper blob `77b591e431378ec984c51e8a032262e673d4c843`

The Run2 transformer replaces only the second unconditional current-primary 32-team coverage assertion in `apply_qb_c2_selector()` with the same frozen exact-current-team validator. It does not alter how `primary` is selected or any QB science.

Sequential regression MUST apply Run1 transformer first, then Run2 transformer, compile the protected QB module, demonstrate both availability-aware current-team guards exist, and demonstrate the separate 32-team QB state-context source-integrity guard remains unchanged. Legacy/exact-explicit helper behavior must also still pass.

A 35-gate retry is allowed only after that regression succeeds. Run1 and Run2 remain non-decisional because neither reached frozen gate evaluation.
