# M96E — Role Router with Frozen Workload-Risk Guard Results

Authoritative:
- workflow `M96E RB Role Workload Risk Guard`
- run `33467630395`
- job `99730679349`
- tested SHA `db1a139a270b7c246d1b5b07dc1a3490cb8fa3a0`
- artifact `9785416331`
- artifact SHA256 `c73a728570516b77c04c4a68ec1541e4a94fb830e144f40f16df63dbcc36dfbe`
- execution success
- disposition `M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP`
- model fit 0; threshold search 0; feature search 0; sportsbook 0; production change 0

## Frozen candidate
Starting from the M96D role-only insight, D was active only for non-entrenched backs and was suppressed when any frozen pregame workload-risk guard fired:
- M95F calibrated 20+ probability >= `.25`;
- M95F p90 workload >= `20` carries;
- M95I prior top-1 unavailable / vacancy-transition state.

No carry projection, tail probability, feature, threshold, or coefficient was refit.

## Results
Weeks 6-18 (`n=961`):
- C/M94C: MAE `21.571881`, RMSE `30.449965`, bias `+0.381967`, corr `.604528`.
- M96E primary: MAE `21.430091`, RMSE `30.431137`, bias `-0.225570`, corr `.605692`.
- MAE gain: `+0.141791` yards.
- RMSE gain: `+0.018828` yards.
- Weeks 13-18 MAE gain: `+0.097105` yards.
- 75+ AUC change: `-0.000407`.
- 100+ AUC change: `+0.001508`.

High-workload safety was substantially repaired:
- actual 15-19 MAE regression `+0.294702` — PASS.
- actual 20+ MAE regression `+0.059047` — PASS.
- actual 25+ MAE regression `+0.159106` — PASS.

The frozen guard protected 69 of 75 actual 20+ games and 20 of 21 actual 25+ games (evaluation-only accounting). Primary D activation fell to `58.69%`; M95F workload guard fired on `33.19%` of rows and vacancy guard on `8.85%`.

## Gate
Eight of nine frozen retention checks passed. The **only** failure was the predeclared materiality requirement:
- required all-RB MAE gain >= `0.150000` yards;
- observed gain `0.141791`;
- shortfall `0.008209` yards.

The line is not waived. M96E is therefore not retained or promoted.

## Scientific interpretation
The modular hypothesis was directionally correct: D contains useful conditional efficiency information, and independent workload/transition modules can almost completely prevent the high-workload damage seen in M96C/M96D. But after paying the safety cost, the residual global point improvement does not clear the frozen materiality gate. Further threshold or feature tuning on the already-opened 2025 sample would be overfitting, not new evidence.

## Final retrospective RB state
- **C / M94C:** retain as conservative global rushing-yard point and central opportunity anchor.
- **M95F:** retain as workload-distribution / stable-workhorse tail evidence, not a universal point-mean boost.
- **M95I:** retain as vacancy/transition diagnostic evidence, not a universal point adjustment.
- **D / M96C efficiency expert:** retain as scientific conditional signal only; not a retained point module after M96E.
- **E/P:** conditional clues only; no retained additive role.
- **X:** rejected as isolated separable tail increment.
- No M91-M96E RB research is production-promoted by this result.

`AUTONOMOUS_RB_RESEARCH_STOP`

Reason: the final precommitted retrospective router repaired the safety problem but missed the frozen materiality gate. Any further retrospective router threshold/feature variants would reuse exposed 2025 outcomes and risk overfitting. New RB architecture evidence must now come from genuinely prospective/untouched 2026 games or a separately justified new-data source that does not retune against the exposed historical outcomes.
