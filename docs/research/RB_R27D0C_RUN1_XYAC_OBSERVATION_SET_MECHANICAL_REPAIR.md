# RB R27D0C — Run1 xYAC Observation-Set Mechanical Repair

Status: `FROZEN MINIMUM VALUE-NEUTRAL MECHANICAL REPAIR BEFORE RERUN`

## Preserved failed run

- Run: `34432353557`
- Job: `102730315801`
- Head / original implementation lock: `99f3347abc6af3ac300b4a78ac1237454678a6e1`
- Failure stage: frozen xYAC mechanism split
- Frozen plan and exact V2 parent verification: PASS
- Production boundary: PASS
- No artifact uploaded; no canonical diagnostic conclusion.

## Exact mechanical failure

The frozen algebra requires, on the same completed-catch observations:

`actual YAC = expected YAC + YACOE`.

Run1 computed player-game `actual_yac` across all completed catches with non-null raw YAC, while `expected_yac` and `yacoe` necessarily used only completed catches with non-null `xyac_mean_yardage`. Because xYAC coverage is approximately 99% rather than exactly 100%, cohort averaging used slightly different observation sets and produced a decomposition gap of `0.0509294626941692` yards/reception.

This is an observation-set plumbing error, not a scientific result.

## Frozen repair

Change only the player-game xYAC decomposition so all three quantities use the identical `xyac_obs == True` catch set:

- `actual_yac_xyac_obs = mean(yards_after_catch | complete_pass=1 AND xYAC non-null)`
- `expected_yac = mean(xyac_mean_yardage | same observations)`
- `yacoe = mean(yards_after_catch - xyac_mean_yardage | same observations)`

No cohort, threshold, source, parent, feature, model, metric, scientific question, or production component may change. The >=98% primary-cohort coverage gate remains unchanged.

The failed Run1 must remain preserved. Only a new locked rerun after this minimum repair may become the first valid R27D0C diagnostic result.
