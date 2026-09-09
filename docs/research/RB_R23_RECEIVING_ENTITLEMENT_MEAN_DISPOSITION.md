# RB R23 — Receiving Entitlement / Receptions / Receiving-Yard Mean Disposition

Status: **MIXED / FAIL — NO PROMOTION**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r23-receiving-entitlement-mean-v1`
Authoritative confirmation run: `34332613867`
Authoritative confirmation head: `0b4e641ec9fa6ba7ea9464a140268dfdbb897d28`
Artifact: `10096546836`
Artifact SHA256: `e3e3087f03d0a8d697225e61de435e9a1668ef74e35ca850e60ffe451e9c66b6`

## Integrity

PASS. The confirmation used zero sportsbook inputs, zero future/2026 outcomes, and conserved the finite RB-room mass to numerical tolerance (`max gap 5.551e-17`). P3 rushing and R22 tail authority were not changed.

## Scientific result

R23 improved pooled target and reception prediction but did not improve receiving-yard point means robustly enough to pass the frozen plan.

Pooled 2023-2025:

- Targets MAE: `1.375079 -> 1.352808` (improved); RMSE `1.911327 -> 1.863159` (improved).
- Receptions MAE: `1.161790 -> 1.151914` (improved); RMSE `1.606351 -> 1.571609` (improved).
- Receiving yards MAE: `10.683986 -> 10.725492` (worse, about +0.39%).
- Receiving yards RMSE: `15.799224 -> 15.696315` (improved).
- Receiving yards bias: `-2.635610 -> -1.133611` (substantially improved).
- Receiving yards p90 absolute error: `23.679984 -> 24.473633` (worse, about +3.35%; frozen protection failed).
- 30+ yard miss rate: `6.420% -> 6.469%` (protected).

Season receiving-yard MAE change versus baseline:

- 2023: `-0.91%` (improved)
- 2024: `+1.08%` (worse)
- 2025: `+0.98%` (worse)

Role receiving-yard MAE change:

- RB1: `+2.68%` (worse; frozen role-robustness gate failed)
- RB2+: `-1.77%` (improved)

Frozen failed gates: directional replication, pooled receiving-yard MAE >=1% improvement, p90 protection, and role robustness. Therefore the frozen disposition is `MIXED_OR_FAIL_NO_PROMOTION`.

## Interpretation

The evidence supports the R23-A diagnosis that within-room RB opportunity allocation is a real source of error: target/reception allocation improved cleanly under conserved finite mass. The failure appears downstream in the conversion from improved opportunity to receiving-yard point mean, especially for RB1s and p90 errors. The combined Candidate 3 cannot be promoted and its coefficients/windows/gates must not be retuned.

## Authorized next hypothesis

Move to R24 as a genuinely different, pre-frozen decomposition test. R24 will isolate the successful entitlement/reception mechanism from the failed new receiving-efficiency component. It will test whether improved strictly-prior RB target/reception entitlement can be paired with the existing production receiving-efficiency mean logic (rather than R23's new shrunk YPR component), with all production authorities and R22 tail logic unchanged. This is not a threshold rescue or parameter retune; it removes the failed mechanism and asks a different causal question.

R23 remains preserved as a scientific null for combined opportunity + new efficiency mean.