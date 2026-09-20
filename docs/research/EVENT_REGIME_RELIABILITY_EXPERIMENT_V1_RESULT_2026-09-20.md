# Event/Regime Reliability Experiment V1 — Result

**Status:** CLOSED — NO REPLICATED RELIABILITY SIGNAL
**Execution commit:** `355fa384375e22811957efd1170bde385d12df20`
**Workflow run:** `35481142421`
**Job:** `105999058019`
**Artifact:** `10595672956`
**Artifact digest:** `sha256:9f9a12f9591c48d003960b30f4ef59bf7f260c1c7a4defb99ce35e0637b437d0`
**Sportsbook read:** false

## Frozen-policy result

The evaluator executed the preregistered 2019–2023 baseline fit, 2024 primary holdout, and conditional 2025 replication exactly as frozen. 2025 was inspected only for families that passed the 2024 primary gate.

No family reached `RELIABILITY_SIGNAL_REPLICATED`.

### Primary families

- `RB_JOINT_TRANSITION`: 2024 primary PASS. Event/non-event ratios: MAE 1.2547, RMSE 1.2490, p90 1.3557, p95 1.3277, residual SD 1.2422, catastrophic-miss rate 2.0504. 2025 replication FAILED because the frozen event-row floor was not met and only one of five dispersion metrics cleared the frozen 3% degradation threshold. Final: `RELIABILITY_SIGNAL_PRIMARY_PASS_REPLICATION_FAILED`.
- `WR_JOINT_TRANSITION`: 2024 primary FAIL. MAE was only marginally worse (1.0509) while RMSE (1.0328), p90 (0.9229), and catastrophic-miss rate (0.7941) failed the frozen degradation gates. 2025 was not inspected. Final: `RELIABILITY_SIGNAL_PRIMARY_FAIL_CLOSED_V1`.

### Secondary churn families

- `QB_RUSH_ROOM_CHURN`: 2024 primary PASS; 2025 replication FAILED because catastrophic-miss rate reversed below non-event (0.8908). Final: `RELIABILITY_SIGNAL_PRIMARY_PASS_REPLICATION_FAILED`.
- `RB_RUSH_ROOM_CHURN`: 2024 primary PASS; 2025 direction remained strongly worse on all reported error ratios, but the frozen 2025 event-row floor was not met. Final: `RELIABILITY_SIGNAL_PRIMARY_PASS_REPLICATION_FAILED`.
- `RB_TARGET_ROOM_CHURN`, `TE_TARGET_ROOM_CHURN`, `WR_TARGET_ROOM_CHURN`, and `WR_RUSH_ROOM_CHURN`: 2024 primary FAIL CLOSED. 2025 was not inspected.

## Interpretation

Discrete room/regime events are real, pregame-observable and non-redundant, but V1 does not establish a temporally replicated reliability signal under the frozen gates. In particular, RB rush-regime events showed large 2024 uncertainty degradation and directional 2025 degradation, but the preregistered replication support requirement prevents promotion. That support failure may not be rescued by lowering the row floor, pooling events post hoc, changing windows, or selecting favorable metrics.

## Governance

- No production change.
- No uncertainty widening, confidence adjustment, abstention, mean boost, or sportsbook use is authorized from V1.
- `ROLE_ROOM_CONCENTRATION_OPPORTUNITY_V1` remains failed closed.
- `EVENT_REGIME_RELIABILITY_EXPERIMENT_V1` is now closed under its frozen definition.
- Any continuation must be a genuinely distinct mechanism with a new pre-outcome contract; it may not retune or rescue this V1.
