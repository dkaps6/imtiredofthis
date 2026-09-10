# RB R27C — Vacancy RB1 / 2023 Receiving-Yard Forensic V1 Result Record

Status: `R27C_FORENSIC_COMPLETE_NO_SINGLE_MECHANISM_IDENTIFIED`

This is a diagnostic-only result. It authorizes no production change, no R26 change, no R22 change, no routing rule, and no receiving-yard mean candidate.

## Canonical execution authority

- Branch: `research-rb-r27c-rb1-2023-forensic-v1`
- Frozen plan commit: `2fd7b1d0078c6104909849116c78888618981b7e`
- Frozen plan blob: `dc1883c40dd194c723453d76e1564de0c204a8e9`
- Diagnostic script blob: `b8a97f82e31725e7673957f0c999a2152d4ea993`
- Workflow blob: `ad4b6a0cddc2d7004bd2fb67ee05379e525cc7a2`
- Implementation lock / canonical head: `2f411fd83a3f72361619572b8c87b4e40fb6fed7`
- Workflow run: `34430754033`
- Job: `102725570088`
- Artifact ID: `10134387002`
- Artifact name: `rb-r27c-rb1-2023-forensic-v1`
- Artifact digest: `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`
- Parent R27B V2 run: `34428917229`
- Parent R27B V2 artifact: `10134023092`
- Parent digest verified exactly: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Integrity

PASS.

- Exact immutable R27B V2 artifact consumed.
- 8429 evaluable rows reproduced.
- Vacancy-active n=1761 reproduced.
- Vacancy RB1 incumbent n=503 reproduced.
- Vacancy RB2+ incumbent n=941 reproduced.
- 2023 vacancy n=259 reproduced.
- Parent B1/C1 MAE values reproduced exactly within frozen tolerance.
- No new model fit.
- No new candidate created.
- Sportsbook inputs: 0.
- Production changed: false.
- R26 changed: false.
- R22 changed: false.

## Finding 1 — RB1 and RB2+ differ primarily in how R26 moves opportunity

Vacancy RB1 incumbents:
- mean R26 target delta: +0.924808 targets
- candidate targets: 3.439681
- B0 rec-yard MAE: 14.305919
- B1 rec-yard MAE: 14.709395
- mean B1 AE minus B0 AE: +0.403476 yards (worse)
- C1 rec-yard MAE: 14.666496
- mean C1 AE minus B1 AE: -0.042899 yards (small repair)

Vacancy RB2+ incumbents:
- mean R26 target delta: -0.143460 targets
- candidate targets: 1.221186
- B0 rec-yard MAE: 10.325630
- B1 rec-yard MAE: 9.978918
- mean B1 AE minus B0 AE: -0.346712 yards (better)
- C1 rec-yard MAE: 9.945148
- mean C1 AE minus B1 AE: -0.033770 yards (additional small repair)

Interpretation boundary: this does NOT prove a simple diminishing-return relationship between added targets and efficiency. The frozen target-volume bins are not monotonic. It does show that the RB1 failure is associated with a very different opportunity-translation regime than RB2+ and cannot be treated as one homogeneous vacancy cohort.

## Finding 2 — 2023 RB1 is the strongest concentrated failure

2023 vacancy RB1 incumbents, n=74:
- B0 MAE: 12.349471
- B1 MAE: 13.992139
- C1 MAE: 14.331884
- mean B1 AE minus B0 AE: +1.642668 yards
- median B1 AE minus B0 AE: +2.682548 yards
- 60.81% of rows worsen B1 versus B0
- mean C1 AE minus B1 AE: +0.339745 yards
- median C1 AE minus B1 AE: +0.221467 yards
- 58.11% of rows worsen C1 versus B1

Non-2023 vacancy RB1 incumbents, n=429:
- mean B1 AE minus B0 AE: +0.189722 yards
- median B1 AE minus B0 AE: -0.098056 yards
- mean C1 AE minus B1 AE: -0.108903 yards
- median C1 AE minus B1 AE: -0.025396 yards

The 2023 RB1 problem is therefore broad enough to appear in medians and worsened-row shares, not only a handful of outliers.

## Finding 3 — 2023 RB1 target count improves slightly while yard value breaks

2023 RB1 target opportunity:
- baseline target MAE: approximately 1.5367
- R26 candidate target MAE: approximately 1.4947
- baseline target bias: approximately -0.7720
- R26 candidate target bias: approximately +0.1336
- mean R26 target delta: +0.905646

Thus R26 moves target count close to unbiased and slightly improves target MAE.

But on targeted 2023 RB1 rows:
- production YPT mean: approximately 5.8393
- actual YPT mean: approximately 5.2278
- realized YPT residual versus production: -0.611555 yards/target
- V2 YPT correction mean: +0.063289 yards/target
- correction/residual correlation: -0.098755

Catch conversion is not the obvious culprit:
- production catch-rate mean on targeted rows: 0.772201
- actual catch-rate mean: 0.784243
- actual minus production catch rate: +0.012042

For non-2023 vacancy RB1 targeted rows:
- realized YPT residual versus production: +0.012885, essentially centered
- V2 correction mean: -0.023288

Therefore the 2023 RB1 failure is specifically associated with a realized efficiency/value-per-target shortfall that production YPT and the V2 context correction did not forecast.

## Finding 4 — no existing V2 context family explains 2023 RB1 by itself

Explanation-only ablations versus B1 MAE for 2023 vacancy RB1:
- player target-shape only: +1.2894% worse
- team/QB environment only: +0.7413% worse
- opponent context only: +1.2972% worse

For non-2023 vacancy RB1:
- player target-shape only: -0.3653% better
- team/QB environment only: +0.2491% worse
- opponent context only: +0.0127% worse / effectively flat

The pre-specified prior context distributions also show only modest shifts between 2023 and non-2023 RB1. This means the existing 13-feature strict-prior context family does not visibly contain a single explanatory state for the 2023 efficiency collapse.

## Finding 5 — the p90 repair and 30+ miss-rate failure are different phenomena

V2 improved pooled vacancy p90 absolute error versus B1, but created six rows that crossed from <30-yard AE under B1 to >=30-yard AE under C1 while repairing five rows in the opposite direction.

For the six new 30+ vacancy misses:
- all six are RB1
- none are from 2023
- mean R26 target delta: +1.521591
- mean V2 YPT correction: -0.439151
- mean actual YPT: 17.527778
- 83.33% remained underpredictions

Within vacancy RB1 specifically, the four new 30+ misses:
- mean R26 target delta: +1.663663
- mean V2 YPT correction: -0.845931
- mean actual YPT: 22.666667
- all four are underpredictions

This explains how p90 can improve while the binary 30+ miss rate slightly worsens: V2 compresses errors broadly but pushes a very small set of explosive RB1 realizations farther into the right-tail miss region.

This tail phenomenon is distinct from the 2023 central-efficiency problem and should not be conflated with it. R22 remains the production receiving-yard tail authority.

## What from V2 is retained

V2 is not promoted as an integrated candidate, but its information is retained as scientific evidence:

1. pooled vacancy MAE modestly improved versus B1;
2. p90 absolute error improved materially versus B1;
3. RB2+ incumbents improved;
4. four of six seasons improved;
5. all-RB and Week1 remained stable/slightly better;
6. negative YPT corrections helped several RB1 cohorts, while positive corrections were often harmful in 2023 RB1;
7. the existing context features are useful for localization but do not explain the full RB1/2023 mechanism.

These facts may shape new questions. They may not be cherry-picked into production or used to create an after-the-fact router on the same evidence.

## Forensic disposition

`R27C_FORENSIC_COMPLETE_NO_SINGLE_MECHANISM_IDENTIFIED`

R27C successfully localizes the unresolved problem but does not establish one causal mechanism strongly enough to freeze a predictive candidate.

The strongest next scientific question is not another Ridge/YPT correction. It is a new diagnostic decomposition of realized target quality for the failing RB1 states, separating target depth, screen/behind-LOS usage, YAC, and explosive-play contribution. This should determine what physically caused the 2023 RB1 YPT shortfall and the separate explosive-tail misses before any R27D candidate is designed.
