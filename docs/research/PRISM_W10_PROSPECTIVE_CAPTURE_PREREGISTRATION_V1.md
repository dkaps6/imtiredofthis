# W10 2026 prospective capture — preregistration

**Date:** 2026-11-10 · **Status:** PREREGISTRATION. No result here. Nothing promoted.
**Script:** `scripts/research/capture_prospective_board_v2.py`

## Why this exists

`w9_prospective_frozen_v1` froze `baseline` and `D3` and listed `D1 overlay` as
pending. D1 was inert at that capture for a specific reason:
`build_injuries_weekly` labelled any non-empty pull `official_report` even when
no row carried a game designation, so `status` was null and the overlay had
nothing to read. That defect was fixed on 2026-11-10 (`classify_report_state`,
`practice_only_report`, 19 regression tests). **D1 is testable for the first
time.**

D3 was graded on the W9 holdout and **rejected**
(`PRISM_D3_W9_HOLDOUT_RESULTS_V1`): it wins MAE by shrinking, leaves RMSE flat,
worsens bias, and changes zero rushing side calls. It is carried here as a
control, not as a candidate.

## Arms

| arm | definition |
|---|---|
| `base` | production-equivalent. Designations have no effect — implied retention OUT 1.00 / DOUBTFUL 1.00 / QUESTIONABLE 1.00. This is the defect. |
| `d1` | A2 from `PRISM_D1_INJURY_STATUS_RESULTS_V1`: per-designation opportunity retention, calibrated on strictly prior weeks, **mass discarded**. |
| `d3` | phantom-yards fix: zero allocated opportunity scores zero yards. Control arm — already rejected on W9. |
| `d1d3` | both. `PRISM_JOINT_D1_D2_D3_RESULTS_V1` found the triple over-corrects on rushing and the admissible split is D1 for receiving, D3 for rushing. This arm tests that split prospectively. |

**A3 is excluded by prior result** — the redistributing variant lost half of A2's
gain and its CI crossed zero. **A1 is excluded** — A2 beat it by +0.00257
[+0.00139, +0.00392], CI excluding zero, so the intended ×0.50 was not adequate.

## D1 retention, calibrated on 2026 weeks 2–9 only

| designation | n | % played | target retention | rush retention |
|---|---|---|---|---|
| OUT | 429 | 0.0% | 0.0000 | 0.0000 |
| DOUBTFUL | 28 | 0.0% | 0.0000 | 0.0000 |
| QUESTIONABLE | 450 | 59.3% | **0.8199** | **0.5477** |

The calibration window ends at week 9. Week 10 is never read.

Note the asymmetry: a Questionable player loses carries far faster than targets
(0.5477 vs 0.8199). A banged-up back is removed from the run game before a
banged-up receiver is removed from the route tree.

## Where D1 is applied, and why it matters

**After** `_sharpen_wr_target_shares` / `_sharpen_rb_rush_shares`, immediately
before `_allocate_counts`.

The sharpeners preserve team mass. Multiplying before them would hand the freed
opportunity back to the designated player's own teammates — that is A3's
redistribution, which failed. Applying after sharpening lets the freed share
fall into `_allocate_counts`'s residual bucket (`residual = 1 - shares.sum()`,
allocated and then discarded by the `[:n_players]` slice) and vanish. That is A2.

## Capture protocol

- **Run Friday 2026-11-13, after game designations publish, before Sunday
  kickoff.** Designations publish Friday; a Tuesday capture reproduces exactly
  the W9 failure. Verified 2026-11-10: the week 10 injury feed has **0 rows**.
- **TNF (WAS @ NYG, Thursday 2026-11-12) is excluded.** The script drops teams
  whose game is already played and records them in
  `teams_excluded_already_played`. The holdout is the 13 Sunday/Monday games.
- Player universe is built from 2026 weeks ≤ 9 and 2025 only, with a hard
  assertion (`assert cur.wk.max() <= prior_week`) that nothing at or beyond the
  target week is read.
- `--include-played` exists for retrospective verification of the arms and
  **never produces a valid holdout**; the meta records the flag.

## Gates

- **G1 — D1 must zero OUT and DOUBTFUL.** Verified on the W9 universe:
  targets 2.26 → 0.00 (OUT, n=24), 2.27 → 0.00 (DOUBTFUL, n=3).
- **G2 — mass must be discarded, not redistributed.** Team totals must FALL.
  Verified: modeled targets −0.18%, carries −1.62%.
  **G2 is stated differently from the original study on purpose.** That study
  applied arms to static share values, so it could demand undesignated players be
  untouched to 0.0e+00. This capture re-simulates, so changing the probability
  vector changes the multinomial draw and undesignated players drift by a mean
  of 0.14 targets. That is Monte Carlo noise, not redistribution, and the
  falling total is what distinguishes them. Reusing the exact-zero gate here
  would be a gate that cannot pass, which signals nothing.
- **G3 — grading uses side accuracy against the line**, not averaged MAE.
  AGENTS.md and the handoff both forbid MAE as a promotion basis for an
  individual-player distribution claim. D3 won MAE on W9 and was still rejected.
- **G4 — the denominator is games played** (`load_snap_counts`,
  `offense_snaps > 0`). Players who did not play are excluded; players who
  played and recorded nothing count as real zeros.

## Kill conditions, stated before scoring

- **K1** — D1 changes fewer side calls than the disagreement count can explain,
  as D3 did (zero rushing disagreements on n=74). Then D1 is a shrinkage
  artifact and is not promoted regardless of MAE.
- **K2** — D1 worsens the bias term. Both arms already under-project and the
  ceiling-compression defect is the largest known source of missed high-usage
  plays; an arm that deepens it does not promote.
- **K3** — `d1d3` beats both single arms on *neither* market. Then the joint
  recommendation (D1 receiving, D3 rushing) is not supported prospectively.
- **K4** — fewer than 20 designated players land in the graded universe. Then
  the slate is underpowered for this question and the result is reported as
  inconclusive rather than negative.

## Prediction and named risk

**Prediction:** D1 improves receiving modestly and rushing more, because the
Questionable rush retention (0.5477) is a far larger correction than the target
retention (0.8199).

**Named risk, which is the productive half:** the retention constants are
averages over 450 Questionable players, and that designation is used very
differently by different clubs. A single team that lists its starters
Questionable as routine practice will inject systematic error that an average
cannot see. The W9 record is 8 hits and ~17 misses on predictions; weight the
risk, not the prediction.

## What is NOT being tested

D2 — its recommendation was withdrawn on yards evidence in
`PRISM_JOINT_D1_D2_D3_RESULTS_V1`. No sportsbook variable is read anywhere in
this capture (AGENTS.md §3).
