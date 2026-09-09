# RB Week 1 2026 Pregame Readiness Run 1 — Role-Key Mechanical Repair V1

Status: **FROZEN BEFORE REPAIR IMPLEMENTATION / RERUN**

## Preserved first authoritative execution

- run: `34412565779`
- head: `66e50f9345ecab8dbf58920526ddfba404738ca8`
- artifact: `10127822769`
- artifact digest: `sha256:fd847cfb6ff1dc9e8a1d0afc76bc98661b8b5ea02c42ecceeb59bc0ea48bb013`
- disposition: `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_FAIL_NOT_READY`
- gates: `34/35 PASS`
- sole failed gate: `29_r26r_current_roles_exact_join_107`
- matched role rows: `104/107`
- unresolved role rows: `3`

All other frozen operational gates passed, including the protected P3 rushing universe, R22 receiving-yard production authority, R26Q 107-array seal, production↔R26Q 107-key equality, R26R 30/30 authority, and all no-outcome/no-production-change boundaries.

## Exact mechanical defect

The protected production/R26Q universe uses punctuation-free canonical player keys. The frozen R26R current-role evidence preserves three Ourlads/provider-style RB keys with punctuation:

1. CHI — `D'Andre Swift`
   - R26R role key: `d'andreswift`
   - protected production/R26Q key: `dandreswift`
   - frozen role: `RB1`

2. MIA — `De'Von Achane`
   - R26R role key: `de'vonachane`
   - protected production/R26Q key: `devonachane`
   - frozen role: `RB1`

3. WAS — `Jacory Croskey-Merritt`
   - R26R role key: `jacorycroskey-merritt`
   - protected production/R26Q key: `jacorycroskeymerritt`
   - frozen role: `RB1`

This is an identity-format compatibility defect only. It is not a player ambiguity, roster ambiguity, role ambiguity, or football-model failure.

## Authorized repair only

The repair may only:

1. copy the exact verified R26R artifact into an isolated staged directory;
2. modify exactly the three `player_clean_key` cells listed above inside the staged `source_current_roles_identity_only.csv`;
3. preserve the same team, player display name, role, model role, depth-chart role, depth slot, depth index, position, and every other role-snapshot field;
4. preserve every other row/cell in the role snapshot;
5. preserve every other R26R artifact file byte-identically;
6. record source/staged hashes and a cell-level audit;
7. rerun the original locked operational-readiness evaluator byte-identically against the staged R26R compatibility copy.

No fuzzy matching is authorized. No broad provider-key rewrite is authorized. Only the three exact mappings above are allowed.

## Prohibited changes

No repair may change:
- RB-P3 rushing attempts or rushing yards;
- R22 receiving-yard means or distributions;
- production receptions means;
- R26 candidate receptions means or arrays;
- any R26Q/R26R football evidence;
- any current role/depth value;
- sportsbook values;
- any frozen readiness threshold or gate;
- production code or parameters.

## Expected repaired result

If Gate 29 was the sole defect, the unchanged evaluator should move from 34/35 to 35/35 and emit:

`RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`

That PASS remains an operational-readiness result only. R26 remains a pregame research sidecar and is not promoted into production by this repair or audit.
