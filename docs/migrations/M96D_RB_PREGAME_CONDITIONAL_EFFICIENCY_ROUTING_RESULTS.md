# M96D — Pregame Conditional Efficiency Routing Audit Results

Authoritative:
- workflow `M96D RB Pregame Efficiency Router`
- run `33467325153`
- job `99729782983`
- tested SHA `dc57217aaa8312edc6c97c43486330ba9894bbc4`
- artifact `9785311314`
- artifact SHA256 `e65ef0cc861e8f27863e5d5fda8ba91b34d921d37724cf431212a9cb4026bf30`
- execution success
- disposition `M96D_PRIMARY_ROUTER_FAILED`
- model fit 0; threshold search 0; feature search 0; sportsbook 0; production change 0

## Result
The frozen primary router turned M96C D on only when M94C projected carries were <15 and the back was not an entrenched workhorse (`role_is_workhorse=1` and prior-five RB share >=.65).

Weeks 6-18 (`n=961`):
- C MAE `21.571881`, RMSE `30.449965`, bias `+0.381967`, corr `.604528`.
- routed D primary MAE `21.416473` (gain `+0.155408`), RMSE `30.461670` (regression `.011705`), bias `-.512217`, corr `.604541`.
- late Weeks 13-18 MAE improved by `0.112548`.
- 75+ AUC was exactly preserved; 100+ AUC improved `+.001386`.

The primary router failed only the frozen high-workload safety gates:
- actual 15-19 MAE regression `+0.327550` — PASS vs <=.50.
- actual 20+ regression `+0.748891` — FAIL.
- actual 25+ regression `+0.537256` — FAIL.

The pregame gate itself was informative:
- projected <10: actual 20+ rate only `.693%`.
- projected 10-<15: actual 20+ rate `13.66%`.
- projected 15+: actual 20+ rate `25.48%`.
- entrenched workhorse: actual 20+ rate `25.25%`.
- not entrenched: actual 20+ rate only `3.16%`.

The controlled role-only diagnostic (not selection-eligible) was stronger globally:
- MAE `21.364096` (gain `+0.207785`), RMSE `30.379446` (improvement), corr `.606687`.
- 75+ AUC `+.001398`; 100+ AUC `+.003367` vs C.
- however it still regressed the evaluation-only 20+/25+ slices by roughly `.675/.396` yards, so it cannot be promoted from this diagnostic.

## Scientific interpretation
M96D supports the routing concept but shows that M94C projected carries alone are not sufficient to guard the rare unexpected workload-spike games. The better role-only diagnostic suggests D is useful for non-entrenched backs broadly; the remaining failure is specifically identifying the small subset of non-entrenched/transition backs whose workload distribution still has meaningful 20+ upside.

That exact job already belongs to existing pregame workload/transition evidence: M95F calibrated 20+/25+ distribution summaries and M95I vacancy/role-transition state. A single final precommitted router-type test is justified. It must use those frozen signals only as a **safety guard** around the role-based D router, not retune carry-tail probabilities or change M94C carries.

## Next
M96E — Role Router with Frozen Workload-Risk Guard.

If M96E cannot preserve the global D gain while eliminating the 20+/25+ damage, retrospective RB efficiency routing stops and C/M94C remains the conservative point architecture pending prospective 2026 evidence.
