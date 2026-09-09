# RB R26H — Week-1 Balanced-Turnover Role-State Atlas V1 Result

Status: **BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research execution head: `2f16ebc403306991247d2dbf220e21373226ab76`

## Authoritative lineage

- Workflow: `RB R26H Week 1 Balanced Turnover Role State Atlas V1`
- Run: `34369024680`
- Job: `102525128622`
- Artifact: `10111095834`
- Artifact digest: `sha256:c3fa00de4262bb98b1f73d6931005f963d0ded4cc9a9fe53734ae421e34ef18f`
- Workflow conclusion: mechanically `success`
- Diagnostic disposition: **`BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL`**

Immutable parents:
- R26 run `34356222339`, artifact `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- R26C run `34361409319`, artifact `10108036449`, digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`
- R26F run `34365496225`, artifact `10109658591`, digest `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`
- R26G run `34365957361`, artifact `10109840313`, digest `sha256:0523055f7b39500d5d6b9fbc7ed85185cae41bf4518889e7f6c58104021c15ca`

R26H regenerated no predictions, refit no R9 model, changed no production parameters, changed no receiving-yard means, changed no R22 behavior, and added no sportsbook inputs.

## Population / integrity

- balanced Week-1 rooms: `72`
- balanced Week-1 same-team incumbent rows: `162`
- exit-state room coverage: `1.0`
- target-game features added: `0`
- sportsbook inputs added: `0`
- R26 predictions regenerated: `false`
- R9 refit: `false`
- production parameters changed: `false`
- receiving-yard means changed: `false`
- R22 changed: `false`
- prospective shadow authorized by R26H: `false`
- production promotion authorized by R26H: `false`

## Frozen source dimensions

Departed-player receiving significance reused the already-frozen R26D definition without modification:

`MEANINGFUL_EXIT = max_exit_prior_targets_pg > 1 OR max_exit_prior_rb_room_share >= 0.25`

Incoming-player state used only pre-existing strict-prior transition flags:
- `new_to_team_veteran`
- `no_prior_nfl_roster`

No same-week historical depth or target-game participation was introduced.

## Child-design eligible states

Two predeclared states met the frozen R26H child-design rule: pooled coherent direction, at least 20 player rows, and matching direction in at least two non-2020 seasons.

### 1. MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT

Rows: `37`

Receptions:
- MAE `1.4871922808 -> 1.3648610282` (**8.22% better**)
- RMSE `1.9414478547 -> 1.6916381746`
- bias `-1.1560689324 -> -0.6653372766`
- p90 absolute error `3.5790692699 -> 2.9619469765`

Targets:
- MAE `1.8116151793 -> 1.7034888734`

Season support existed in all six seasons. Beneficial R26 direction replicated outside 2020 in four seasons.

Season-level R26 reception-MAE effect (candidate minus baseline):
- 2020: `+0.13174` harmful
- 2021: `-0.47751` beneficial
- 2022: `-0.27530` beneficial
- 2023: `-0.03439` beneficial
- 2024: `-0.20005` beneficial
- 2025: `+0.71778` harmful, but only `n=1`

### 2. MEANINGFUL_EXIT + NO_PRIOR_ENTRY_PRESENT

Rows: `77`

Receptions:
- MAE `1.1445967352 -> 1.0901142742` (**4.76% better**)
- RMSE `1.5589772379 -> 1.4873503509`
- bias `-0.5751015291 -> -0.0849293506`
- p90 absolute error `2.4664056597 -> 2.3271270719`

Targets:
- MAE `1.3774914360 -> 1.3763261782` (essentially flat/slightly better)

Season support existed in all six seasons. Beneficial R26 direction replicated outside 2020 in three seasons.

Season-level R26 reception-MAE effect:
- 2020: `+0.24244` harmful
- 2021: `+0.20854` harmful
- 2022: `-0.08324` beneficial
- 2023: `+0.07263` harmful
- 2024: `-0.38635` beneficial
- 2025: `-0.31006` beneficial

## Unsupported states

The following predeclared states did not meet the minimum support/replication contract and may not be converted into a router from R26H:
- `LOW_EXIT + VETERAN_ENTRY_PRESENT` (`n=6`)
- `LOW_EXIT + NO_PRIOR_ENTRY_PRESENT` (`n=13`)
- `UNKNOWN_EXIT_HISTORY` (`n=7`)

Their observed directions remain diagnostic evidence only.

## Scientific interpretation

R26F correctly identified balanced turnover as a risk state, but R26G's blanket baseline fallback was too broad. R26H shows why: some balanced-turnover rooms still contain genuine receiving-role transition where the original R26/R9 redistribution remains useful.

The strongest reusable signal is therefore:

> In a Week-1 balanced-turnover room, preserve original R26 when a **meaningful receiving-role exit** is paired with a supported incoming-role state; otherwise retain the conservative baseline fallback until separately supported.

This is a component-design signal, not a qualified child candidate.

## Authorized next child design

R26H authorizes the design of a separately frozen **R26I Week-1 Selective Restoration** child only.

Proposed inheritance:
1. non-vacancy room -> production baseline exact;
2. unbalanced vacancy room -> original frozen R26 exact;
3. balanced-turnover room:
   - `MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT` -> restore original R26;
   - `MEANINGFUL_EXIT + NO_PRIOR_ENTRY_PRESENT` -> restore original R26;
   - unsupported balanced states -> remain baseline exact.

Important: R26I must be frozen before its output is scored. It may not retune the R26D significance thresholds, change R9, or loosen the 2020 safety gate. R26H itself authorizes neither shadow nor production.

## Components to preserve

Preserve exactly:
- original R26/R9 identity mechanics;
- R26 finite RB target-pool conservation;
- non-vacancy production-exact behavior;
- non-RB exactness;
- receiving-yard mean exactness;
- R22;
- sportsbook separation;
- strict-prior leakage protections;
- R26, R26D, R26E, R26F, and R26G negative evidence/dispositions;
- R26H unsupported-state evidence.

No production files were changed by R26H.
