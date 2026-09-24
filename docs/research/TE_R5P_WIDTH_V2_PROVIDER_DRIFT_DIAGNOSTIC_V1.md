# TE-R5P Width V2 — PR #549 Provider-Drift Diagnostic V1

Date: 2026-09-24

Status: FROZEN MECHANICAL DIAGNOSTIC ONLY

## Purpose

Diagnose why the current deterministic historical rebuild does not reproduce the frozen PR #549 baseline `mc_proj` on exact frozen keys. This is a replay/source-integrity problem, not a Width V2 scientific result.

No Width V2 hypothesis, width factor, gate, tolerance, football mean, or sportsbook separation rule is changed here.

## Authorities

- frozen PR #549 source SHA: `f04a8a775f4a56fe282cb292f6a52bd509bc8f24`
- frozen replay run: `34722725629`
- compact artifact: `10307242156`
- compact digest: `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`
- expired raw-draw artifact: `10306649017` — must not be approximated
- failed Width V2 run #3: `35944066618`, job `107458022324`
- run #3 frozen-key max `mc_proj` drift: `14.020649281259615` yards

## Narrow observations already established

1. The 35 current-only output rows are all one identity:
   - Tampa Bay WR Chris Godwin / current key `chrisgodwinjr`
   - 2024 Weeks 1-7
   - five markets per week: rec_yards, receptions, rush_att, rush_rec_yards, rush_yards.
2. The frozen PR #549 trace has no Godwin output rows in those seven weeks.
3. This does **not** by itself prove the pregame universe gained one player. Source PR #549 and current run #3 report identical 2024 pregame-universe row counts for every Week 1-18.
4. Source and current historical player-log builders also report identical regular-season row counts:
   - 2023: 5,387
   - 2024: 5,324
   - 2025: 5,370
5. The core pinned runtime versions relevant to this replay remain the same in source/current logs (Python 3.11 path, pandas 1.5.3, numpy 1.26.4, scipy 1.11.4, scikit-learn 1.4.2, nflreadpy 0.1.5).
6. The compact artifact contains frozen traces and per-week distribution metadata, but no `.npz` 2,000-draw arrays. The source run exposes the compact artifact plus the expired raw-distribution artifact; no surviving duplicate raw-draw artifact has been identified in the checked source-run/commit evidence.
7. `walk_forward.py`, `historical_inputs.py`, and the core component-prediction path were not changed between PR #549 source and this branch. The material current code change in `historical_player_logs.py` only changes historical regular-season week selection semantics and leaves 2023-2025 row counts unchanged.

## Working hypotheses

Ranked for this diagnostic only:

1. historical provider payload values or identity fields mutated while row counts stayed constant;
2. a provider identity/actual-attachment mutation explains the Godwin output difference and may coexist with another football-input mutation;
3. team-week/PBP or another upstream historical feature changed in-place;
4. changed hidden universe membership with the same total counts changed team allocation/conservation;
5. code/runtime drift is lower probability but remains testable through component-field parity.

## Frozen diagnostic test

Re-run the exact current baseline reconstruction once, but before failing the existing parity gate:

- merge current and frozen PR #549 rows on the exact frozen identity key;
- preserve the exact `1e-10` baseline parity tolerance;
- write a row-level diagnostic ordered by absolute `mc_proj` delta;
- write a field-level drift summary for upstream football components already present in the frozen trace, including:
  - `rules_plays_est`
  - `rules_pass_rate`
  - `rules_tgt_share`
  - `rules_rush_share`
  - `rules_ypt`
  - `rules_catch_rate`
  - `rules_ypc`
  - `rules_ypa`
  - context availability flags
  - MC team volume / pass-rate / efficiency fields
  - ML/state/ensemble/actual as downstream checks
- record current-only rows separately;
- then fail closed exactly as before if parity is not exact.

The diagnostic files are evidence only and cannot be used to fit Width V2.

## Time-box / stopping rule

This authorizes one diagnostic replay.

If it identifies one deterministic historical source seam that can be restored to the frozen PR #549 authority without changing the V2 scientific contract, one exact repair replay is allowed.

If it instead reveals broad historical provider mutation, multiple unresolved seams, or any reconstruction that would require approximating the expired 2,000-draw arrays, Width V2 is parked as source-blocked. The project then pivots immediately to the sanctioned RB teammate-availability / injury-created-vacancy rushing-opportunity science lane.

No Normal approximation. No k search. No relaxed parity. No sportsbook input upstream. No 2026 outcome fit.


## Diagnostic run #4 result

Run: `35948975494`  
Artifact: `10788097328`  
Artifact digest: `sha256:53bd3156339432f877024f4bafdb63d2aeb3a9649a5bf7f53332d8cbf4863b66`

Run #4 failed closed at the unchanged canonical baseline parity gate, as intended. No Width V2 science ran.

Observed replay drift:

- frozen rows: 51,197
- current rows: 51,232
- current-only rows: 35
- missing frozen rows: 0
- frozen-key MC drift rows above `1e-10`: 8,457
- max absolute `mc_proj` drift: 14.020649281259615 yd
- **all 8,457 MC-drift rows are 2024 Weeks 1-7**
- **zero MC drift exists in 2024 Weeks 8-18 or in 2025**
- Tampa Bay is the dominant location; the largest receiving-yard drift is TB Week 7.
- smaller drift propagates league-wide in Weeks 1-7, consistent with a shared player-consensus/Bayesian layer changing when one historical identity becomes resolvable.

Upstream component localization:

- exact/no drift:
  - `rules_plays_est`
  - `rules_pass_rate`
  - every context-availability flag checked
  - MC team plays/dropback/pass-attempt fields
  - `ml_proj`
  - `state_proj`
  - `actual`
- drifted on 4,125 frozen rows:
  - `rules_tgt_share`
  - `rules_rush_share`
  - `rules_ypt`
  - `rules_ypc`
  - `rules_ypa`
  - `rules_catch_rate`
  - corresponding Bayesian/rules efficiency fields

This pattern, together with the exact Chris Godwin W1-7 current-only output seam, supports the post-#549 manual identity override as the deterministic MC replay cause.

A second independent mechanical drift is also proven:

- `ensemble_proj` differs on 23,403 frozen rows while ML/state are exact.
- PR #549 source `data/model_ensemble_weights.csv` contains only pass_yards, rush_att, and rush_yards weights.
- current `data/model_ensemble_weights.csv` adds promoted rec_yards and receptions weights after #549.
- therefore exact #549 baseline reconstruction must use the source-era ensemble-weight file, not today's production weights.

## Authorized exact repair replay

One repair replay is authorized under the time-box.

Inside the Actions workspace only, before historical reconstruction:

1. restore `data/manual_name_overrides.csv` from source SHA `f04a8a775f4a56fe282cb292f6a52bd509bc8f24`;
2. restore `data/model_ensemble_weights.csv` from the same source SHA;
3. do **not** revert either file in production/main;
4. run the unchanged frozen parity gates.

This is source-environment reconstruction, not model retuning.

If exact baseline parity still fails after these two identified deterministic source seams are restored, Width V2 is parked source-blocked and the project pivots immediately to the sanctioned RB teammate-availability / injury-created-vacancy rushing-opportunity lane. No further open-ended replay plumbing is authorized.
