# RB R26H — Week-1 Balanced-Turnover Role-State Atlas V1 — Frozen Plan

Status: FROZEN BEFORE OUTCOME SLICING
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`

## Purpose

R26G proved that a blanket fallback for every Week-1 balanced-turnover RB room (`room_exits_n == room_entrants_n`) is too broad. It partially repaired 2020 but erased material R26 gains in later seasons, especially 2024 and 2025.

R26H is a diagnostic only. It does not create new predictions, refit R9, or authorize shadow/production. Its purpose is to distinguish **replacement/churn balanced rooms** from **real receiving-role transition balanced rooms** using football-natural, strict-prior state already frozen in parent evidence.

## Immutable parent evidence

R26 V1:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26C exit-significance source audit:
- run `34361409319`
- artifact `10108036449`
- digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`

R26F:
- run `34365496225`
- artifact `10109658591`
- digest `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`
- disposition `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED`

R26G:
- run `34365957361`
- artifact `10109840313`
- digest `sha256:0523055f7b39500d5d6b9fbc7ed85185cae41bf4518889e7f6c58104021c15ca`
- disposition `WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW`

## Fixed population

Only historical Week-1 balanced-turnover vacancy rooms:
- seasons 2020-2025;
- `week == 1`;
- `vacancy_active == 1`;
- `room_exits_n == room_entrants_n`.

Primary player scoring population remains same-team incumbents inside those rooms.

No Week-2+ rows and no unbalanced rooms may influence the primary diagnostic.

## Frozen source dimensions

### A. Departed-player receiving significance

Reuse the already-frozen R26D definition unchanged:

`MEANINGFUL_EXIT = max_exit_prior_targets_pg > 1 OR max_exit_prior_rb_room_share >= 0.25`

Classify each balanced room as:
- `MEANINGFUL_EXIT`;
- `LOW_EXIT` — prior history exists but no exit satisfies the meaningful definition;
- `UNKNOWN_EXIT_HISTORY` — no usable departed-player receiving history.

The `>1` / `0.25` thresholds may not be changed after outcome grading.

### B. Incoming-player NFL role state

Use only transition flags already emitted by frozen R26 predictions:
- `new_to_team_veteran == 1`;
- `no_prior_nfl_roster == 1`.

At room level classify:
- `VETERAN_ENTRY_PRESENT` — at least one current RB/FB is a new-to-team veteran;
- `NO_PRIOR_ENTRY_PRESENT` — at least one current RB/FB has no prior NFL roster state;
- `MIXED_ENTRY_STATE` — both are present;
- `UNRESOLVED_ENTRY_STATE` — neither flag is present despite balanced turnover.

Do not infer current depth rank or target-game participation.

### C. Predeclared combined football states

Report these combinations without tuning:
1. `MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT`
2. `MEANINGFUL_EXIT + NO_PRIOR_ENTRY_PRESENT`
3. `LOW_EXIT + VETERAN_ENTRY_PRESENT`
4. `LOW_EXIT + NO_PRIOR_ENTRY_PRESENT`
5. `UNKNOWN_EXIT_HISTORY` (entry state reported secondarily)

These are diagnostic states, not routers.

## Frozen questions

For each source state, compare original frozen R26 versus production baseline on:
- incumbent receptions MAE;
- incumbent receptions RMSE;
- incumbent signed bias;
- incumbent p90 absolute error;
- incumbent target MAE.

Also report R26's effect versus baseline by season and state.

Primary questions:
1. Is R26 harm in balanced-turnover rooms concentrated in low/unknown departed receiving significance rather than meaningful exits?
2. Does presence of an established veteran entrant distinguish balanced rooms where R9 redistribution remains useful?
3. Does `MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT` show beneficial R26 direction outside a single season?
4. Does `LOW_EXIT + NO_PRIOR_ENTRY_PRESENT` show harmful/neutral R26 direction outside a single season?
5. Can any football-natural state justify a separately frozen child design while preserving the strong original R26 Week-1 gains in rooms outside that state?

## Replication rule for child-design authorization

A state may be eligible for a later R26I child design only if:
- pooled direction is materially coherent with the football hypothesis;
- the same beneficial/harmful direction appears in at least 2 historical seasons with nonzero support outside 2020;
- total player support is at least 20;
- the state is defined entirely by the frozen football-natural source dimensions above;
- no alternative threshold, quartile, residual magnitude, or season-specific rule is introduced after outcomes are viewed.

R26H itself never authorizes prospective shadow or production.

## Component-preservation requirements

Regardless of R26H result, preserve:
- R26 broad Week-1 vacancy/R9 gains outside a separately supported guard state;
- fixed RB receiving pool and conservation;
- production-exact non-vacancy rooms;
- non-RB exactness;
- receiving-yard mean exactness;
- R22;
- sportsbook separation;
- strict-prior leakage protections;
- R26/R26E/R26F/R26G negative evidence and dispositions.

Do not automatically reuse the failed blanket R26G fallback.

## Prohibited actions

- no R9 refit;
- no new prediction generation;
- no threshold search;
- no same-week historical depth;
- no sportsbook feature;
- no receiving-yard mean change;
- no R22 change;
- no production write;
- no child candidate before this diagnostic is completed and documented.
