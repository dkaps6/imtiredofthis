# RB R26I — Week-1 Selective Restoration V1 Frozen Plan

Status: FROZEN BEFORE CHILD SCORING
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`

## Purpose

R26E showed strong Week-1 R26 performance but failed one seasonal safety gate in 2020. R26F identified balanced turnover (`room_exits_n == room_entrants_n`) as a replicated risk state. R26G proved a blanket baseline fallback for all balanced-turnover rooms was too broad and erased real R26 gains. R26H then identified two predeclared balanced-room states in which original R26 remained beneficial with multi-season support.

R26I is the separately frozen child candidate authorized by R26H. It makes no new model fit. It constructs each Week-1 RB target/reception projection by choosing one of two already-frozen parent endpoints: production baseline or original R26.

## Immutable parent evidence

R26:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26C:
- run `34361409319`
- artifact `10108036449`
- digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`

R26H:
- run `34369024680`
- artifact `10111095834`
- digest `sha256:c3fa00de4262bb98b1f73d6931005f963d0ded4cc9a9fe53734ae421e34ef18f`
- disposition `BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL`

Parent negative dispositions remain binding:
- R26 `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- R26E `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- R26G `WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW`

## Frozen child logic

Only Week 1 is in scope.

### State 1 — non-vacancy

If `vacancy_active != 1`:
- child targets = production baseline targets exactly;
- child receptions = production baseline receptions exactly.

### State 2 — unbalanced vacancy

If `vacancy_active == 1` and `room_exits_n != room_entrants_n`:
- child targets = original frozen R26 candidate targets exactly;
- child receptions = original frozen R26 candidate receptions exactly.

### State 3 — balanced turnover

If `vacancy_active == 1` and `room_exits_n == room_entrants_n`:

Restore original R26 **only if all of the following are true**:
1. departed-player state is `MEANINGFUL_EXIT` under the already-frozen R26D threshold:
   - `max_exit_prior_targets_pg > 1` OR
   - `max_exit_prior_rb_room_share >= 0.25`;
2. and the room has at least one supported incoming state from frozen R26H:
   - `VETERAN_ENTRY_PRESENT` (`new_to_team_veteran == 1` on at least one current RB/FB), OR
   - `NO_PRIOR_ENTRY_PRESENT` (`no_prior_nfl_roster == 1` on at least one current RB/FB).

For such a room:
- child targets = original frozen R26 targets exactly;
- child receptions = original frozen R26 receptions exactly.

Every other balanced-turnover room:
- child targets = production baseline targets exactly;
- child receptions = production baseline receptions exactly.

No interpolation, coefficient scaling, per-player mixing, or threshold tuning is permitted.

## Why this is not a 2020 rescue rule

R26I uses only states predeclared in R26H and the unchanged R26D significance definition. It does not use season identity, error magnitude, or 2020 membership as a feature. R26H required each eligible state to replicate outside 2020 before authorizing child design.

The 2020 safety gate remains unchanged/strict; R26I is allowed to fail it.

## Population

Historical test seasons: 2020-2025.
Primary population:
- Week 1 only;
- vacancy-active same-team incumbent RB/FB rows.

Secondary safety populations:
- all Week-1 RB/FB rows;
- Week-1 vacancy RB1 incumbents;
- Week-1 vacancy RB2+ incumbents.

## Structural invariants

R26I must preserve exactly:
- no R9 refit and no prediction regeneration;
- original R26 values where R26 is selected;
- baseline values where baseline is selected;
- complete-room endpoint selection, never player-by-player endpoint mixing within a team-week;
- fixed RB-room target mass;
- non-RB exactness;
- receiving-yard production means;
- R22;
- sportsbook inputs upstream = 0;
- future/target-game feature use = 0;
- protected production files unchanged.

## Frozen scientific gates

Use the R26G child protections as the floor, with inheritance checks adapted to selective restoration.

1. R26 immutable parent structural integrity passes.
2. R26H authorized child-design disposition and exact artifact digest verified.
3. No regeneration and no R9 refit.
4. Sportsbook/future inputs zero; production receiving-yard mean and R22 unchanged.
5. Non-vacancy rooms are baseline exact.
6. Unbalanced vacancy rooms are R26 exact.
7. Balanced authorized-restoration rooms are R26 exact.
8. Balanced unsupported rooms are baseline exact.
9. Child RB-room target mass gap <= `1e-10`.
10. Pooled Week-1 vacancy-incumbent reception MAE improves vs baseline.
11. Pooled Week-1 vacancy-incumbent reception RMSE is non-worse vs baseline.
12. Pooled Week-1 vacancy-incumbent absolute bias is non-worse vs baseline.
13. Pooled Week-1 vacancy-incumbent reception p90 worsens <=2% vs baseline.
14. Pooled Week-1 vacancy-incumbent target MAE improves vs baseline.
15. Week-1 vacancy-incumbent reception MAE improves in at least 4 of 6 seasons.
16. **No Week-1 season worsens more than 2% vs baseline.** Do not loosen this for 2020.
17. Support >=150 incumbent rows and >=20 rows in at least 4 seasons.
18. RB1 reception MAE worsens <=1% vs baseline.
19. RB2+ reception MAE worsens <=1% vs baseline.
20. At least one of RB1/RB2+ improves reception MAE vs baseline.
21. Global Week-1 reception MAE worsens <=0.5% vs baseline.
22. Global Week-1 reception RMSE worsens <=0.5% vs baseline.
23. Global Week-1 absolute bias is non-worse vs baseline.
24. Preserve original R26 pooled vacancy-incumbent reception MAE within 0.5%.
25. Preserve original R26 pooled vacancy-incumbent target MAE within 0.5%.
26. Preserve original R26 2021-2025 vacancy-incumbent reception MAE within 0.5%.
27. Preserve original R26 global Week-1 reception MAE within 0.5%.

All gates must pass for retrospective Week-1 support.

## Dispositions

If structural inheritance fails:
`WEEK1_SELECTIVE_RESTORATION_INTEGRITY_FAILURE`

If all frozen gates pass:
`WEEK1_SELECTIVE_RESTORATION_RETROSPECTIVE_SUPPORT_FOR_2026_SHADOW`

If structure is intact but any scientific gate fails:
`WEEK1_SELECTIVE_RESTORATION_MIXED_NO_SHADOW`

Even on full pass:
- only a 2026 Week-1 prospective shadow is authorized next;
- production promotion is false;
- all-season use is false.

## Prohibited actions

- no threshold search;
- no season-specific rule;
- no R9 coefficient change;
- no per-player endpoint selection inside one RB room;
- no same-week historical depth;
- no sportsbook feature;
- no receiving-yard mean change;
- no R22 change;
- no production write;
- no weakening the 2020 seasonal safety gate after results.
