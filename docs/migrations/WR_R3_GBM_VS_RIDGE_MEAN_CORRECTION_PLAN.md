# WR-R3 GBM-vs-Ridge Mean-Correction Trial — Frozen Plan

**STATUS: FROZEN BEFORE ANY CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

Checkpoint 38 (Issue #535) approved a parallel hypothesis test: every residual/
mean-correction attempt in this repo so far (M89/M90, RB-PD2 through PD6, WR
bias-shrink V1, WR-R3 lane A) either uses Ridge (a linear model) or a
hand-specified linear/clipped rule. None have tried a more expressive,
threshold-aware model family on the same signal. WR-R3's checkpoint 26 result
(PR #552, merged) gives a concrete, already-multi-season-validated candidate
to test this on: the correction genuinely improves 2025 MAE (-0.21 yd, 6/6
seasons directionally consistent) but narrowly misses the frozen >=1%
improvement gate (0.95%), and creates a small moderate-miss (20yd band)
regression while improving larger tails. A model family that can represent a
non-linear/threshold relationship between the same strict-prior features and
the true residual might close that last ~0.05pp gap or fix the tail
redistribution — or might not; either result is informative.

**RB-PD2's third authorized leg is explicitly NOT the subject of this trial.**
`docs/research/overnight/RB_PD_CHAIN_STATUS.md` documents that RB residual
work is genuinely blocked on its own unresolved multi-season-cohort
discrepancy (PD5/PD6 only ever ran a 2025-only cohort; the discrepancy doc
forbids further RB residual experiments until a real 2020-2025 RB evidence
set is built and a replication is dispositioned first). That is a separate,
larger prerequisite this trial does not attempt to satisfy. This trial uses
WR-R3 instead, which already has a real, dispositioned, six-season
(2020-2025) cohort.

## Exact design

**Signal tested**: the same strict-prior WR-R3 features already frozen and
reused unmodified from PR #552 — `prior8_m38_bias`, `prior8_m38_mae`,
`prior8_m38_miss30_rate`, `prior_games` — read from the same
`wr_r3_walkforward_casebook.csv` artifact (`EXPECTED_R3_ROWS = 12396`, run
`34064572328`). No new feature engineering.

**Target**: per-row residual `actual - m38_baseline_proj` (rec_yards only,
same market WR-R3 corrects), same as every prior residual-calibration attempt
in this repo.

**Genuine train/holdout split** (per checkpoint 38's methodological-consistency
item, applied more strictly here than WR-R3 lane A's own hand-rule, which had
no fit step to hold anything back from): fit on 2020-2024 (5 seasons, `prior_games
>= 4` rows only, matching `MIN_PRIOR` in the existing evaluator), freeze both
models completely, then apply frozen predictions blind to 2025 only. No
refitting, no peeking at 2025 during either model's fit.

**Model arms**, both fit on the identical train rows/features/target:
1. `Pipeline([StandardScaler(), Ridge(alpha=20)])` — matches
   `run_m89_pregame_synthesis.py`'s own frozen precedent exactly (the one
   place in this repo that already uses literal `sklearn.linear_model.Ridge`
   for a mean correction): same alpha, and the same standardization step,
   since the four WR-R3 features are on materially different scales (yards,
   a [0,1] rate, a raw game count) and Ridge's L2 penalty is scale-sensitive.
   An unscaled Ridge arm would be an avoidably weakened comparator, not a
   clean test of model family alone.
2. `HistGradientBoostingRegressor` (scikit-learn, already a pinned dependency;
   no new library) with `max_depth=3, max_iter=200, learning_rate=0.05,
   min_samples_leaf=30, random_state=42` — conservative depth/leaf-size
   choice appropriate for a ~10k-row training set to avoid the GBM simply
   overfitting the training seasons' noise.

Both frozen models' predicted correction is clipped to the exact same
`[-8, +8]` yard range WR-R3 lane A already uses, so any difference in result
is attributable to the model family, not to a wider allowed correction.

**Baseline for comparison**: the exact frozen M38 baseline (`m38.csv` per
season, same `EXPECTED_2025_ALL_REC` parent constants already certified in
`evaluate_wr_r3_combined_calibration.py`) and the exact already-known WR-R3
lane-A-only 2025 result (mean-lane only, no width lane, so this is an
apples-to-apples mean-correction-only comparison — the width lane is
orthogonal and is not touched by this trial).

**Evaluation gates on the 2025 holdout, applied identically to both arms**:
- `>=1%` MAE improvement vs the exact M38 parent (WR-R3's own frozen gate).
- RMSE/bias/correlation non-worsening (WR-R3's own guards).
- WR1/WR2/WR3 role-level MAE improvement (diagnostic, not a promotion gate,
  same as WR-R3).
- miss20/30/40/under50 tail guards, computed identically to WR-R3's own
  evaluator, so the moderate-miss regression WR-R3 lane A showed can be
  directly compared model-family-to-model-family.
- Train-set (2020-2024) diagnostic MAE reported for both arms, to check for
  gross overfitting (a GBM arm that improves training MAE by far more than it
  improves 2025 holdout MAE is a red flag on its own, independent of the
  promotion gates).

**No mean-neutrality-across-arms requirement**: unlike the base
ensemble/distribution-widening work, this trial's whole point is comparing
mean-correction *magnitude and shape*, not verifying two translators produce
the identical mean — so this gate class does not apply here.

**What this trial is not**: not a production integration test, not a
replacement for WR-R3's combined candidate (which is already closed per
checkpoint 26's disposition), and not an attempt to satisfy RB-PD2's blocked
multi-season prerequisite. A positive result here would be reported as "GBM
beats/matches Ridge/hand-rule on this specific signal" and would still require
its own separate promotion review before any production change — no such
change is proposed by this plan.

## Reproducibility inputs

Reuses, unmodified: `scripts/backtest/build_historical_inputs.py` (per-season
input build), the same M38 rebuild steps and `wr_r3_walkforward_casebook.csv`
artifact WR-R3 already produced, and `evaluate_wr_r3_combined_calibration.py`'s
own `load_r3_features`/`m38_map`/`actual_map`/`score` helpers where directly
applicable, to avoid re-deriving already-validated plumbing.

## Amendment log

**Amendment 1 (before any run-1 result was inspected)**: GPT-5.6 (checkpoint
28, Issue #535) identified that the first committed implementation used a bare
unscaled `Ridge(alpha=20)`, not the `StandardScaler -> Ridge(alpha=20)`
pipeline `run_m89_pregame_synthesis.py` actually uses. Corrected per the
"Model arms" section above before reading run 1's candidate output — run 1
(workflow run `34730965831`) is **non-authoritative for model-family
attribution** and is superseded by the corrected re-run. Run 1 also failed
outright on an unrelated `ModuleNotFoundError` (a PYTHONPATH package-resolution
conflict between the pinned M38-parent checkout and this research branch's own
`evaluate_wr_r3_combined_calibration.py`, fixed by loading that module directly
by file path instead of via package import), so it produced no candidate
output at all.

No sportsbook/odds inputs. No production/model/weight/threshold change.
