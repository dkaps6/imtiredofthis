# Full-Stack Vegas Benchmark V1 — result record

**Disposition:** `FULL_STACK_EDGE_THRESHOLD_HISTORICAL_BENCHMARK_NO_EDGE_QB_AND_NON_QB`

## What this measures

The follow-up to `historical_market_vegas_benchmark_v1` (base MC engine only,
naive bet-every-disagreement rule). This benchmark instead uses:

- the full production stack (MC + ML + State + the calibrated production
  ensemble in `data/model_ensemble_weights.csv`) for every market;
- for QB `pass_yards` specifically, an out-of-sample reconstruction of the
  frozen M89/M90 football-only synthesis (trained on 2023, tested on
  2024-2025, never touching its own test data during fitting);
- the real production decision gate from
  `scripts/master_betting_workbook_core_v2.py` (`implied_prob`, `no_vig`,
  `ev_roi`; STRONG EDGE at EV >= .05 and prob-edge >= .03, LEAN EDGE at EV >
  0) instead of a naive rule, reported by tier.

RB `P3`/`R26` are correctly excluded: they are qualified for the 2026
Week-1 route only and out of scope for a 2024-2025 historical grade.

**Known gap, not a scope decision:** unlike P3/R26, `WR_R15_PRODUCTION_MODEL_V1`
and `TE_R5P_PRODUCTION_MODEL_V1` are *not* Week-1-locked -- their own
validation record covers multiple historical seasons (WR-R15: 2022-2025,
TE-R5P: 2023-2025), so they are legitimately applicable to this backtest.
They are **not yet included** in the `rec_yards`/`receptions` numbers below;
those numbers reflect the shared MC+ML+State+ensemble engine only. This
understates the actual deployed WR/TE stack and should be corrected in a
follow-up pass (`scripts/modeling/wr_r15_entitlement_adapter_v1.py`,
`scripts/modeling/te_r5p_entitlement_adapter_v1.py`, with
`backtest-te-r5-participation-entitlement-v1.yml` on the
`research-te-r5-participation-entitlement-v1` branch as the precedent for a
backtest-compatible application) before treating the WR/TE numbers here as
final.

## QB pass_yards result

Trained the synthesis on 2023, tested out-of-sample on 2024-2025 real
DraftKings/FanDuel lines (`qb_base_summary.csv`, `qb_synthesis_summary.csv`,
`qb_market_assisted_summary.csv`):

| Candidate | PLAY tier win rate | PLAY tier ROI/unit | Model MAE | Vegas MAE |
|---|---:|---:|---:|---:|
| Base ensemble (no synthesis) | 50.3% | -5.3% | 60.2 | 56.9 |
| Football-only synthesis reconstruction | 50.8% | -4.6% | 60.3 | 57.3 |
| Market-assisted (sees Vegas spread/total) | 53.2% | -0.06% | 58.9 | 57.6 |

The real STRONG/LEAN/PASS gate did not rescue the football-only candidates;
tightening the filter did not clearly improve ROI for QB. The
market-assisted candidate approaches breakeven, but it is explicitly not a
football-only result (it uses Vegas spread/total/moneyline as input
features) and must not be read as evidence of a football-only edge.

### Reconstruction-fidelity note (important)

`qb_synthesis_scoreboard_reconstruction_v1.csv` and
`qb_synthesis_gates_reconstruction_v1.json` are from an **independent
reconstruction** of the M89/M90 methodology using this repo's general
`scripts/backtest/walk_forward.py` harness plus the exact frozen
`correct_m89_team_semantics.py` / `build_historical_injuries.py` /
`build_historical_weather.py` steps the original M89/M90 workflows used —
not a replay of the original artifact/run. The documented production result
(`docs/production/QB_PASS_SYNTHESIS_V1.md`) for the same train-2023/test-
2024-2025 split is base MAE ~57.64 -> synthesis ~55.06. This reconstruction
did not fully reproduce that gain (see follow-up investigation before
treating this as a finding against the promoted synthesis; a first pass
without the team-semantics/injury/weather corrections showed an even larger
gap, so at least part of the discrepancy is reconstruction fidelity, not
model failure).

## Non-QB markets result (rush_yards, rec_yards, receptions, rush_rec_yards)

Full stack (ensemble applied, no position-specific overlay since P3/R26 are
out of scope), graded across 2024-2025 (`non_qb_summary.csv`,
`non_qb_detail.csv`):

| Market | PLAY tier win rate | PLAY tier ROI/unit | ALL_NO_FILTER ROI/unit |
|---|---:|---:|---:|
| rush_yards | 51.5% | -2.5% | -3.6% |
| rec_yards | 51.6% | -2.9% | -3.5% |
| rush_rec_yards | 50.7% | -4.9% | -5.2% |
| receptions | 54.1% | -0.6% | -1.3% |

Unlike QB, the edge-threshold gate shows a real, monotonic effect here: ROI
improves as the filter tightens (no filter -> lean-or-strong -> strong-only)
in every one of these four markets. None crossed breakeven, but the
confidence signal is doing directionally real work for these markets in a
way it was not for QB pass_yards.

## Fidelity caveats (disclosed, not hidden)

- Fair probability uses a Normal(mean, component_sd) approximation (spread
  across mc_proj/ml_proj/state_proj), not the literal simulated Monte Carlo
  outcome distribution production uses for `fair_prob`.
- The free archive's line/odds are "latest captured," not a fixed
  30-minutes-before-kickoff snapshot.
- QB market-assisted numbers must never be read as a football-only result.

## Anti-reinvention rule

Do not repeat this exact base/threshold comparison expecting a different
answer. The next legitimate steps are: (1) resolve the QB reconstruction-
fidelity gap against the documented ~57.64->~55.06 result before drawing any
conclusion about the promoted synthesis itself, and (2) if a real edge
exists anywhere, it is more likely in non-QB markets under a tighter edge
threshold than in QB pass_yards.
