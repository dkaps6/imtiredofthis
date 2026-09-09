# RB Week 1 Pregame Readiness Run 1 — Mechanical Repair V1

Status: **FROZEN BEFORE REPAIR / RERUN**

## Preserved run

- run: `34414018379`
- job: `102674683265`
- head: `d6d17c98c3a59571e79d019c820065e26d05658f`
- result: mechanical failure before readiness disposition / football simulation
- partial artifact: `10128358371`

## What passed before failure

The run successfully completed:
- frozen plan / implementation / lock byte checks;
- protected production-code boundary checks against `f8417f55b04ce0e19baf260e9d532765034c47f1`;
- exact Full Slate artifact digest/head verification;
- exact R22 artifact digest verification;
- exact R26Q artifact digest/head verification;
- exact R26R artifact digest/head verification;
- exact protected selector staging;
- fresh 2026 Ourlads capture: 468 rows / 32 teams, including 94 RB + 13 FB = 107 RB/FB rows.

No readiness scientific/structural gate was reached and no football simulation result was emitted.

## Mechanical failure 1 — protected production identity representation

The protected production artifact stores:
- `player_form_consensus.csv.player_clean_key` as compact identity keys, e.g. `bijanrobinson`;
- `model_context_bridge.csv` without `player_clean_key`, so the protected Full Slate reconstruction falls back to display `player`, e.g. `Bijan Robinson`.

The protected builder correctly fails closed because its certified model-context roster keys do not equal PlayerForm compact keys. The traceback shows one-for-one display/compact pairs, not different players.

This is the same already-proven representation defect encountered in R26N. The exact existing frozen helper is already present in branch ancestry:
- `scripts/backtest/stage_r26n_production_identity_key_repair_v1.py`
- original helper commit `0b4c2df7c14b16bd0e945b425aac7f35e015fa3d`

That helper:
1. copies the immutable downloaded production artifact to an isolated staging tree;
2. requires exact 468-row one-to-one `(team, player display name)` equality between PlayerForm and model-context;
3. adds only `model_context_bridge.player_clean_key` from immutable PlayerForm;
4. performs no fuzzy matching or normalization heuristic;
5. proves all original model-context columns/values remain byte-value-equivalent at the DataFrame level;
6. changes no football value and adds/removes no player;
7. leaves the downloaded source parent untouched.

### Authorized repair

Reuse that exact helper byte-identically. Do not write a new mapping method. Hash/commit-lock it in the workflow and run the frozen readiness evaluator against only the isolated staged production copy.

## Mechanical failure 2 — evidence-display shell heredoc

The `Show readiness evidence` step contained a nested shell/Python heredoc indentation error and failed after the main evaluator had already failed.

### Authorized repair

Replace only that display command with an equivalent non-heredoc read/print command. This step is evidence presentation only and cannot alter evaluator inputs, gates, or disposition.

## Frozen science remains unchanged

No changes are authorized to:
- the 22 frozen readiness gates;
- any PASS/WARNING/FAIL disposition;
- P3 values or logic;
- R22 model/assets or logic;
- R26Q arrays, means, scope, or hashes;
- R26R market/role snapshot;
- the fresh Ourlads roster snapshot;
- the 107-player exact-key requirement;
- the 2,892-array exact reconstruction requirement;
- seed 42 / 25,000 draws;
- sportsbook boundary;
- Week-1 outcome prohibition;
- production authority ceiling.

This repair is strictly mechanical representation compatibility and evidence display.