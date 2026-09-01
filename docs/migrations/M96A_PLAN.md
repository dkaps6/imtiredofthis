# M96A — RB Opportunity vs Efficiency Attribution Audit

## Purpose

M95T closed new retrospective stable-workhorse carry-tail candidate development, but RB research is not complete until the selected workload foundation is translated into rushing yards. M96A is a **no-fit attribution audit** designed to answer one question before any new yardage model is trained:

> How much of the remaining player-game rushing-yard error is attributable to workload/opportunity versus rushing efficiency, and do the retained post-M94C workload components improve the downstream yardage representation when used in the roles they actually validated for?

## Frozen workload arms

M96A scores exactly three workload representations on the exact 2025 RB validation universe shared by the frozen M94C/M95F/M95I artifacts.

1. **M94C central** — `m94c_rush_att`; this remains the conservative central workload reference.
2. **M94C + M95F distribution** — the frozen M95F empirical hurdle distribution. For point-yard sensitivity only, its already-exported `m95f_mix_mean` is used as the distribution expectation. Its p50/p75/p90/p95 remain distribution diagnostics, not replacement point forecasts.
3. **M94C + M95F + separate M95I vacancy branch** — use the M95F distribution expectation for incumbents; only when frozen M95I marks `prior_top1_unavailable == 1`, use the already-selected M95I carry transform (`m95i_rush_att`). This is a mutually exclusive vacancy branch and does **not** double-add M95I uplift to M95F tail mass.

M95I is not promoted wholesale. Stable incumbents do not receive the M95I transform in Arm 3.

## Frozen efficiency forecast for attribution

To keep M96A interpretable, no new efficiency model is fit and M95C environment features are not added yet.

The pregame efficiency forecast is the exact frozen efficiency implied by the authoritative M94C yardage output:

`pred_ypc = candidate_rush_yards / candidate_rush_att`

with the same defensive numerical guard already used by the M95I downstream sensitivity audit (`2.0 <= pred_ypc <= 7.0` when projected carries are positive).

This means any difference among the three primary arms is workload-only.

If M96A shows efficiency is the dominant remaining bottleneck, the retained M95C environment-only signal becomes the first justified M96B candidate. If workload remains dominant, RB opportunity research is reopened only against the specific error mode demonstrated here rather than by resuming generic M95 tail search.

## Oracle diagnostics — postgame only

Actual game outcomes are used only after prediction for attribution. They are forbidden as pregame inputs.

For each workload arm:

- **Pregame translation:** projected carries × frozen pregame expected YPC.
- **Perfect-opportunity oracle:** actual carries × frozen pregame expected YPC.
- **Perfect-efficiency oracle:** projected carries × actual game YPC. For zero-carry games, actual YPC is defined as the frozen predicted YPC solely so the efficiency contribution is zero and the entire miss is correctly attributed to opportunity.
- **Perfect-both oracle:** actual rushing yards.

The exact residual identity is also exported:

`actual_yards - projected_carries*pred_ypc = (actual_carries-projected_carries)*pred_ypc + actual_carries*(actual_ypc-pred_ypc)`

The first term is the signed opportunity component in yards; the second is the signed efficiency component. Absolute-component summaries are diagnostic and are not assumed additive under MAE.

## Metrics

Primary point metrics by arm and slice:

- n
- rushing-yard MAE
- RMSE
- signed bias
- Pearson correlation
- mean actual and mean prediction

Slices:

- all RB
- actual carries 0–5
- 6–10
- 11–14
- 15–19
- 20+
- 25+
- incumbent
- vacancy
- stable workhorse where frozen fields permit

Attribution outputs:

- MAE with real pregame workload
- MAE with perfect opportunity
- MAE with perfect efficiency
- recoverable MAE improvement from each oracle
- per-game absolute opportunity vs efficiency dominance rate
- mean/median absolute component size
- correlation of each signed component with total signed rushing-yard residual

Tail diagnostics:

- 75+ and 100+ rushing-yard AUC using each arm's frozen point/expectation score
- event counts
- M95F yard-quantile coverage from `m95f_p50/p75/p90/p95 × pred_ypc`

The quantile translation intentionally holds efficiency deterministic. Undercoverage therefore indicates missing efficiency variance rather than permission to retune the carry distribution.

## Integrity rules

- Research only; no production change.
- No sportsbook inputs.
- Model fit/search = 0.
- Feature search = 0.
- Coefficient/hyperparameter search = 0.
- Do not change M94C, M95F, or M95I frozen architecture.
- Do not use M95K/T failed stable-workhorse rerankers.
- Do not waive M94C's inherited legacy rush-yard guard.
- 2025 is already inspected research data; M96A is attribution/development evidence, not pristine confirmation.

## Decision rule

M96A does not promote a model. It routes the next research step.

- **Opportunity-dominant:** perfect-opportunity oracle improves all-RB MAE materially more than perfect-efficiency oracle, or opportunity component dominates per-game absolute residuals. Return to a narrowly defined opportunity issue informed by the slice diagnostics.
- **Efficiency-dominant:** perfect-efficiency oracle materially exceeds the opportunity oracle. Advance to M96B efficiency/environment synthesis, beginning with retained M95C environment signal.
- **Joint:** neither component clearly dominates and both oracles recover meaningful error. Advance to a joint M96B synthesis with workload distribution and efficiency distribution kept separate.

Regardless of route, M94C remains the central workload reference unless a later precommitted candidate earns replacement.