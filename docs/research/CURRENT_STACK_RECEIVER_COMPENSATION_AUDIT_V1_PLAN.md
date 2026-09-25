# Current-Stack Receiver Compensation Audit V1 — Frozen Diagnostic Plan

Date: 2026-09-25

Status: **FROZEN BEFORE DIAGNOSTIC COMPUTE**

This is a diagnostic-only attribution study. It does not test a production
candidate and does not authorize any model change.

## 1. Why this audit exists

Receiver Targetable-Dropback V1 established a genuine team-opportunity signal:

- the exact cumulative-count strict-prior targetable-dropback formula improved
  team receiver-target MAE independently in 2022, 2023, 2024 and 2025;
- the unchanged formula passed its 2024-2025 confirmation gates;
- no shrinkage, recency tuning, threshold, carveout or sportsbook input was used.

But the frozen player full-stack integration failed closed:

- pooled receiving-yard macro MAE improved slightly;
- RB receiving-yard MAE improved materially;
- WR and TE receiving-yard MAE worsened;
- receptions worsened for WR/TE;
- high-entitlement Q4 tails worsened;
- RB rush+receiving protection failed.

The raw MC component already contained the WR/TE regression, so the final
ML/State ensemble was not the source of the failure.

The question is therefore not whether team targetable volume exists. It is:

**Where does a valid team-level opportunity correction get lost between team
volume and final WR/TE/RB player receiving output?**

## 2. Standing anti-retest constraints

This audit must not recreate or tune closed work.

### Closed / preserved
- C1 position-group target-mass calibration is closed.
- C3 joint C1+C2 architecture is closed.
- One-Pass-State Integration V1 is closed.
- Hierarchical Receiver Mean Reconciliation V1 is closed.
- Receiver Targetable-Dropback V1 player full-stack candidate is closed.
- M36-M38 WR hierarchy is canonical.
- WR-R15 remains canonical inside its frozen WR2+ scope.
- TE-R5P remains canonical.
- No nearby C1 history-window / pseudo-count / group-calibration search is
  permitted.

### Prior-art concepts allowed for diagnosis only
- M34 receiving error decomposition;
- M35 team-volume vs player-share decomposition;
- Receiving Ecosystem Conservation Audit room-mass diagnostics.

Reuse their accounting concepts, not their obsolete generic projection
authority.

## 3. Historical authority

Diagnostic seasons:
- 2024 regular season;
- 2025 regular season;
- pooled.

Reconstruct the same current-stack historical authority used by the frozen
Targetable-Dropback full-stack test:

- strict-prior historical player/team context;
- explicit M38 target entitlement;
- fold-safe TE-R5P;
- fold-safe WR-R15 where authorized by the existing replay contract;
- exact current targetable-dropback cumulative-count rate;
- no sportsbook input.

Outcomes may be used **only after pregame quantities are frozen** for diagnostic
attribution. This is not a validation dataset for a new candidate.

## 4. Required pregame quantities

For every modeled WR / TE / RB / FB player-game:

- baseline team dropback mean;
- strict-prior targetable-dropback rate R_T;
- baseline team target-pool mean = dropbacks;
- corrected team target-pool mean = dropbacks * R_T;
- final explicit entitlement p_i after M38 / TE-R5P / WR-R15;
- position room;
- predicted catch rate;
- predicted YPT;
- evidence state / relevant uncertainty fields when available.

No target-game outcome may affect these fields.

## 5. Actual quantities

Attach after all pregame quantities are frozen:

- actual player targets;
- actual receptions;
- actual receiving yards.

Derive:
- actual team targets;
- actual room targets for WR / TE / RB_FB;
- actual player share within room;
- actual catch rate where targets > 0;
- actual YPT where targets > 0.

## 6. Stage A — team-volume attribution

Reproduce the already-established team-level signal inside the current-stack
player cohort.

For baseline and targetable volume:

- team target MAE / RMSE / bias / p90;
- 2024, 2025 and pooled;
- changed-team closer rate.

This is a parity/attribution check, not a new qualification.

## 7. Stage B — room-mass attribution

Groups:
- WR
- TE
- RB_FB

For each team-game:

Baseline predicted room targets:
`dropbacks * sum(p_i in room)`

Targetable predicted room targets:
`dropbacks * R_T * sum(p_i in room)`

Compare each against actual room targets.

Report for each room, season and pooled:
- MAE;
- RMSE;
- bias;
- p90 AE;
- candidate-minus-baseline MAE;
- candidate closer rate.

Also report room-share bias using a composition denominator:

- actual room share = room actual targets / modeled WR+TE+RB+FB actual targets;
- predicted room share = room entitlement / modeled WR+TE+RB+FB entitlement.

This share diagnostic is independent of total target volume and therefore shows
whether room composition itself remains biased.

## 8. Stage C — within-room allocation attribution

For each player with a positive predicted room entitlement:

Predicted within-room share:
`p_i / sum(p_j in same room)`

Actual within-room share:
`actual player targets / actual room targets`

Report by:
- WR / TE / RB_FB;
- WR1 vs WR2+ where the current promoted hierarchy provides the identity;
- entitlement quartile;
- season and pooled.

Metrics:
- share MAE;
- bias;
- correlation;
- target-count MAE if actual room volume is supplied but predicted within-room
  share is preserved.

This stage diagnoses hierarchy/allocation only. It must not fit a replacement.

## 9. Stage D — opportunity / efficiency compensation

Define deterministic pregame player target means:

- T0 = baseline dropbacks * p_i
- T1 = baseline dropbacks * R_T * p_i
- E_pred = current predicted YPT
- Y0 = T0 * E_pred
- Y1 = T1 * E_pred

Actual:
- T_actual = actual targets
- Y_actual = actual receiving yards
- E_actual = Y_actual / T_actual when T_actual > 0

Use the exact symmetric product decomposition:

`Y_actual - Y_pred =
 (T_actual - T_pred) * (E_actual + E_pred) / 2
 + (E_actual - E_pred) * (T_actual + T_pred) / 2`

### Frozen zero-target semantics

To keep the identity exact without silently dropping important rows:

- if T_actual = 0, set E_actual = E_pred; all error on that row is attributed
  to opportunity;
- if T_pred = 0 and T_actual > 0, set E_pred = E_actual; all error is attributed
  to opportunity;
- if both are 0, both contributions are 0.

Verify rowwise identity within 1e-9 yards.

Compute the decomposition separately for T0 and T1.

Report by WR / TE / RB, Q4, season and pooled:
- mean signed opportunity contribution;
- mean signed efficiency contribution;
- mean absolute opportunity contribution;
- mean absolute efficiency contribution;
- fraction of absolute decomposition burden from opportunity vs efficiency;
- receiving-yard deterministic MAE/bias under Y0 and Y1.

## 10. Compensation tests

The audit must explicitly answer these yes/no descriptive questions:

1. Does targetable volume improve team target error while worsening WR or TE room
   target-count error?
2. Does targetable volume improve room target count but worsen player target
   count because within-room shares are wrong?
3. Does targetable volume improve target-count accuracy while receiving-yard
   error worsens because predicted YPT is already too low?
4. Is the compensation pattern concentrated in high-entitlement/Q4 receivers?
5. Is RB different from WR/TE because its baseline room/allocation/efficiency
   errors have a different sign?

No result from these questions may be turned directly into a carveout.

## 11. Integrity gates

Diagnostic interpretation stops unless all pass:

1. same historical universe and fold-safe specialist authority as the frozen
   full-stack test;
2. targetable rate is strict-prior for every team-game;
3. target entitlement is unchanged;
4. TE-R5P conservation passes;
5. WR-R15 conservation passes where applied;
6. actual targets are attached only after pregame rows are frozen;
7. no sportsbook fields enter pregame calculations;
8. symmetric yard decomposition identity max gap <=1e-9;
9. no fitted parameter;
10. no candidate variant is scored;
11. production changed = false.

## 12. Interpretation / next-action rule

This audit does not qualify any production change.

Possible outcomes:

### A. Efficiency compensation dominates
If corrected target opportunity becomes more accurate but WR/TE yardage remains
bad because YPT/efficiency is systematically too low, the next candidate must be
a genuinely new leakage-safe efficiency-information mechanism with its own
fresh validation plan. Do not partially undo R_T.

### B. Room-mass composition dominates
If the corrected team pool is good but room target counts are systematically
wrong, C1 remains closed. A new room-mass candidate is allowed only if it uses
genuinely new information not present in C1, such as current-season
participation/state evidence already shown to persist. No nearby C1
window/pseudo-count search.

### C. Within-room hierarchy dominates
If team and room volume are sound but player allocation is wrong, the next lane
must respect M38/WR-R15/TE-R5P anti-retest history and identify genuinely new
pregame information. No generic rank multiplier retune.

### D. Mixed/no clear structure
Do not create a new candidate. Move to another independent model weakness.

Rushing/scramble opportunity remains a separate parked lane.
