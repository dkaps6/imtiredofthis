# RB STACK6C — Rotation-Gated Contraction Hurdle Results

## Authoritative execution

- Branch: `research-rb-stack6c-contraction-hurdle`
- Run: `33571608136`
- Job: `100066569291`
- Tested SHA: `48505071764d5323fcd92c133f4a1bf328268e34`
- Artifact: `9825130726`
- Artifact SHA256: `451df50897f4ac87de72dd71a47ae755f09f92657ad56f81ead0b0fecbfe210d`
- Frozen plan: `docs/migrations/RB_STACK6C_CONTRACTION_HURDLE_PLAN.md`
- P3 / broad STACK6 parent run: `33549529203`
- rotation source run: `33571118247`
- corrected identity source run: `33548910561`
- downstream-only market benchmark run: `33499129109`

## Protocol integrity

Coverage:

- target RB/FB player-games: `1,393`
- W6-18 rows: `1,001`
- eligible W6-18 rows: `550`
- M95F-risk W6-18 rows: `319`
- depth-rank-1 W6-18 rows: `380`
- same-team rotation identity match anywhere: `99.6411%`
- strictly prior rotation-history coverage: `96.3388%`
- strict as-of leakage pass: `100%`

Feature contracts remained frozen:

- `ROTATION_HURDLE`: exactly `10` rotation features;
- `AGG_PLUS_ROTATION_HURDLE`: exactly `24` features (`14` aggregate + `10` rotation);
- positive carry corrections allowed: `0`;
- M95F-risk rows unchanged exactly;
- depth-rank-1 rows unchanged exactly;
- feature / threshold / hyperparameter / weight / cap / population search: `0`.

The football-first disposition was computed before sportsbook data was loaded.

## Football results

### Eligible W6-18 — 550 rows

| Arm | Carry MAE | Carry bias | Yard MAE |
|---|---:|---:|---:|
| P3 parent | `2.960281` | `+0.003033` | `15.887072` |
| ROTATION_HURDLE | `3.137385` | `-1.697886` | `15.803761` |
| AGG_PLUS_ROTATION_HURDLE | `3.144588` | `-1.667601` | `15.812415` |

The hurdle slightly reduced mean absolute yard error but did so by systematically removing too many carries. The central carry distribution became materially worse.

### Eligible W13-18 — 269 rows

| Arm | Carry MAE | Carry bias | Yard MAE |
|---|---:|---:|---:|
| P3 parent | `3.124120` | `-0.087352` | `16.437330` |
| ROTATION_HURDLE | `3.316347` | `-2.100638` | `16.648484` |
| AGG_PLUS_ROTATION_HURDLE | `3.357271` | `-2.102822` | `16.680801` |

Unlike the STACK6B descriptive contraction replay, the frozen hurdle does not preserve the late-season yard improvement. It overcontracts late-season workloads as well.

## Retention gates

### ROTATION_HURDLE

- eligible carry MAE gain: `-0.177104` — FAIL
- eligible yard MAE gain: `+0.083311` — FAIL vs required `+0.15`
- W13-18 eligible yard gain: `-0.211154` — FAIL
- all-RB W6-18 yard MAE regression: `-0.045775` (improvement) — PASS
- eligible absolute carry-bias worsening: `+1.694854` — FAIL
- M95F-risk unchanged: PASS
- depth-rank-1 unchanged: PASS
- no expansion: PASS

### AGG_PLUS_ROTATION_HURDLE

- eligible carry MAE gain: `-0.184307` — FAIL
- eligible yard MAE gain: `+0.074657` — FAIL
- W13-18 eligible yard gain: `-0.243470` — FAIL
- all-RB W6-18 yard MAE regression: `-0.041020` (improvement) — PASS
- eligible absolute carry-bias worsening: `+1.664569` — FAIL
- M95F-risk unchanged: PASS
- depth-rank-1 unchanged: PASS
- no expansion: PASS

Frozen disposition:

`STACK6C_NO_RETAINABLE_ROTATION_CONTRACTION_INCREMENT`

P3 remains the champion point parent. No production change.

## Depth behavior

The one-sided architecture is not rescued by depth segmentation.

Depth 2, W6-18:

- P3 yard MAE: `19.391209`
- ROTATION_HURDLE: `19.782842`
- AGG_PLUS_ROTATION_HURDLE: `19.524593`

Depth 3+, W6-18:

- P3 yard MAE: `15.249457`
- ROTATION_HURDLE: `14.644452`
- AGG_PLUS_ROTATION_HURDLE: `14.955624`

The depth-3+ yard improvement is real descriptive evidence, but both hurdle arms create large negative carry bias and fail the frozen global eligible gates. It is not eligible to justify a depth-specific retune on exposed 2025 data.

## Downstream market benchmark — descriptive only

Exact market-covered all-899:

- P3: `24.315798`
- ROTATION_HURDLE: `24.513841`
- AGG_PLUS_ROTATION_HURDLE: `24.485895`
- Vegas: `23.701891`

Market-covered eligible subset, 269 rows:

- P3: `19.979784`
- ROTATION_HURDLE: `20.641645`
- AGG_PLUS_ROTATION_HURDLE: `20.548248`
- Vegas: `19.048327`

The downstream benchmark agrees with the football-first rejection.

## Scientific interpretation

STACK6C resolves a key ambiguity from STACK6B.

The problem is **not simply that the prior model was allowed to expand carries**. When expansion is prohibited and a fixed classifier decides whether to contract, the model still contracts far too often and/or too strongly. P3 begins the eligible population almost perfectly centered (`+0.003` carry bias); the hurdle pushes that population to roughly `-1.7` carries of mean bias.

Therefore:

1. prior-game rotation hierarchy contains useful descriptive information but is insufficient by itself to determine which target-game secondary backs should lose workload;
2. the missing state is increasingly consistent with **target-game backfield availability / competitor return / usable-RB composition**, not another transformation of lagged workload alone;
3. the current feature family should not be retuned through classifier thresholds, depth thresholds, cap changes, class weights, or late-season gates on exposed 2025 data;
4. P3 remains structurally strong enough that broad contractions are more damaging than leaving the parent untouched.

## Retained capability map

- P3 remains central point parent.
- M95F-risk and depth-rank-1 protections remain unchanged.
- live-capable PBP rotation proxy remains a valid information family for future architectures, but not a retained point correction.
- broad STACK6 bidirectional regression: rejected.
- STACK6B compact bidirectional regression: rejected.
- naive/frozen one-sided contraction replay: diagnostic only, not retained.
- STACK6C two-stage contraction hurdle: rejected.

## Next justified direction

Do **not** retune the hurdle.

Audit exact target-game backfield availability / competitor-state information with explicit historical timing proof. Priority variables:

1. active vs inactive status of each credible competing RB at the pregame decision timestamp;
2. returning competitor after absence;
3. newly unavailable competitor relative to prior game;
4. usable active RB count;
5. depth-chart movement conditional on actual game-day availability;
6. same-game injury-report status only where historical publication time is demonstrably pre-kickoff.

Target-game participation remains forbidden as a substitute for pregame availability.
