# RB-STACK4 — Frozen Efficiency Capability Portability

Status: PRECOMMITTED DEVELOPMENT PLAN
Parent: STACK3 retained central composition = STACK2 architecture with full-stack Week-1 override.

## Purpose

Test whether the already-frozen M96C efficiency capabilities remain incremental after the stronger STACK2/STACK3 opportunity architecture exists.

This is a portability/compatibility audit. There is no feature search, coefficient refit, ridge-alpha change, threshold search, or sportsbook input.

2025 is exposed retrospective development data. Results can define a frozen prospective candidate but are not independent confirmation.

## Parent

`P3 = STACK1 full-stack rushing-yard projection in Week 1; otherwise STACK2 arch_enriched_opp_stack_eff_yards.`

This parent is frozen before any M96C residual is read.

## Frozen M96C signals

Source run `33462888850`, artifact `migration-96c-rb-m94c-efficiency-residual`, row trace `m96c_oof_trace.csv`.

- `delta_E`: blocking/environment residual YPC signal.
- `delta_P`: player-created residual YPC signal.
- `delta_D`: opponent/run-resistance residual YPC signal.

M95D X remains rejected as an isolated tail increment and is not reopened here.

M96C residuals exist only for the original expanding-week evaluation window (Weeks 6-18). Rows without a frozen residual remain at P3 unchanged.

## Precommitted arms

1. `P3_PARENT` — unchanged.
2. `P3_PLUS_E_ENRICHED_SCALE` — P3 + enriched M94C carries * frozen delta_E.
3. `P3_PLUS_P_ENRICHED_SCALE` — P3 + enriched M94C carries * frozen delta_P.
4. `P3_PLUS_D_ENRICHED_SCALE` — P3 + enriched M94C carries * frozen delta_D.
5. `P3_PLUS_D_NATIVE_SCALE` — P3 + original M94C carries * frozen delta_D. This diagnoses whether D portability depends on its native opportunity scale.
6. `P3_PLUS_D_NONRISK` — apply enriched-scale D only when the already-frozen M95F workload-risk guard is false; otherwise leave P3 unchanged. Frozen risk definition: calibrated P(20+) >= .25 OR p90 workload >=20. This is a compatibility guard, not a threshold search.

No E+P/E+D/P+D combination is added in this migration because the original M96C combinations failed to dominate the individual blocks and the objective here is incremental capability attribution, not a new combination search.

## Evaluation

Report all-RB, Week 1, Weeks 2-5, Weeks 6-18, actual carry bands for diagnosis only, pregame M95F risk/non-risk, and the exact same 899 archived-market subset downstream.

For market rows report MAE/RMSE/bias/correlation and disagreement buckets 0-2.5, 2.5-5, 5-10, 10+.

## Interpretation

- If E/P/D do not improve P3, mark them redundant/non-portable relative to the new parent rather than forcing them in.
- If one improves a narrow pregame-identifiable regime without harming the parent elsewhere, retain it as a conditional capability.
- Do not route on actual carries.
- Do not feed Vegas upstream.
- Do not retune M96C from 2025 outcomes.
