STATUS: RESEARCH ONLY — NOT PROMOTED. No production/model/weight/threshold
change. First-ever valid Vegas comparison for QB M89 synthesis — the
previously-committed `qb_synthesis_summary.csv` was produced by an untracked,
since-lost ad-hoc script and could not be re-verified as-is (see Issue #535
checkpoint 12).

# QB M89 Clean Vegas Rebuild V1 — Result

Two separate questions, answered separately, not conflated:
1. Does the frozen M89 recipe improve over the base ensemble on its own
   internal promotion gates? (a football-only accuracy/stability question)
2. Does it beat Vegas? (the actual question this whole thread exists to answer)

Built via `.github/workflows/backtest-qb-m89-m90-clean-rebuild-v1.yml`
(removed after this run per repo governance — see below), run
`34710055190`, identity-clean on 2022-2025 historical inputs (0 identity
failures, independently traceable). Ridge α=20, trained on 2023 only,
evaluated on 2024-2025, exactly per `M90_QB_SYNTHESIS_CONFIRMATION_PROMOTION.md`'s
frozen contract — nothing tuned for this run.

## 1. Internal M90 promotion gates (football-only accuracy, not vs Vegas)

| model | n | MAE | bias | correlation | 100+yd misses |
|---|---:|---:|---:|---:|---:|
| base (2024+2025 combined) | 894 | 61.73 | -10.65 | 0.194 | 159 |
| football_synthesis | 894 | 60.49 | +4.31 | 0.257 | 160 |
| market_assisted | 894 | 59.54 | +7.95 | 0.302 | 154 |

`football_synthesis` gates: `mae_gain_gate=true`, `both_seasons_nonworse=true`,
`rmse_nonworse=true`, `correlation_gate=true`, `bootstrap_gate=true`
(p=0.942), `market_coverage_gate=true`, **`tails_nonincrease=false`**
(159→160) → **`all_gates_pass=false`**. This is the same result as the
first (identity-buggy) attempt — not caused by the game_id issue, this is
a real property of the recipe on this cohort.

`market_assisted` clears all 7 gates (`all_gates_pass=true`) but per
`M90_QB_SYNTHESIS_CONFIRMATION_PROMOTION.md` is explicitly **not eligible
to become the football-only projection** — it's a secondary
decision-layer benchmark only, because it uses market-derived features.

## 2. Vegas comparison (STRONG_ONLY_PLAY_TIER, the real bet-selection gate)

| variant | matched | model_mae | vegas_mae | win_rate | roi_per_unit |
|---|---:|---:|---:|---:|---:|
| base_proj | 681 | 60.137 | 56.196 | 49.19% | **-7.36%** |
| football_synthesis (M89) | 672 | 59.471 | 57.612 | 51.19% | **-3.69%** |
| market_assisted | 629 | 57.828 | 56.955 | 53.58% | **+0.75%** |

`ALL_NO_FILTER` and `LEAN_OR_STRONG` tiers show the same direction and are
in the committed `qb_base_summary.csv`/`qb_synthesis_summary.csv`/
`qb_market_assisted_summary.csv`.

## Honest read

**M89 football_synthesis helps, real and consistent across both seasons
(not noise) — win rate improves 49.2%→51.2%, ROI loss roughly halves
(-7.36%→-3.69%) — but it still loses to Vegas on both raw accuracy and
ROI, and it fails its own internal tail-risk gate** (a small increase in
catastrophic misses, 159→160, essentially flat but technically a fail).
Same pattern as every other layer tested tonight (base ensemble,
WR-R15/TE-R5P): real, measurable improvement, gap not closed.

**market_assisted is the one genuinely positive-ROI result of the entire
night (+0.75%)** and comes closest to Vegas on raw MAE. But it is not a
fair test of "does our football model beat Vegas" — it's allowed to see
market-derived features as input, so it's closer to "does a small
model-informed correction on top of Vegas's own line find anything" than
an independent projection. Per your team's own frozen rules, it cannot
become the production football-only projection regardless of this result.
Flagging as interesting, not as a candidate.

## Governance note

`.github/workflows/backtest-qb-m89-m90-clean-rebuild-v1.yml` (the CI
workflow that produced this result) has been removed from this branch
after extracting these results, per `tests/test_repository_hygiene.py::
test_frozen_qb_research_is_not_an_active_actions_surface` — that test
enforces M90's own closure rule ("M90 ends broad QB point-projection
research either way") by permanently forbidding any committed
`.github/workflows/backtest-qb-*` file. The reusable scripts
(`attach_qb_synthesis_game_identity_v1.py`, `grade_qb_synthesis_vegas_v1.py`)
and their tests remain, per that same test's explicit allowance for
"historical scripts/docs."
