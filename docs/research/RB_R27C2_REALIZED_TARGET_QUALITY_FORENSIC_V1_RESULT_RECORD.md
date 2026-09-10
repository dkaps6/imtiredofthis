# RB R27C2 — Realized Target-Quality Forensic V1 Result Record

Status: `R27C2_FORENSIC_COMPLETE_PHYSICAL_TARGET_QUALITY_HYPOTHESIS_IDENTIFIED`

This is diagnostic-only evidence. It authorizes no production change, no R26 change, no R22 change, no router, and no receiving-yard mean candidate.

## Canonical authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Parent R27C result commit: `b2907345c50dceffdfb5c074ce9d556ff6d764c6`
- Frozen R27C2 plan commit: `28dfb2d5d35513f8a9369155bf394dea38d83792`
- Frozen plan blob: `57e26c75f8f551c28c86d49c55375ef4f590b6e9`
- Diagnostic script blob: `179cabe788afc9d995a827b597ba269fb7dcdded`
- Run1 preserved mechanical failure: run `34431105624`, job `102726619356`, head `aa9406750e23efa77f8186c3054512d55a7ff316`
- Run1 repair record: `docs/research/RB_R27C2_RUN1_IMPORT_PATH_MECHANICAL_REPAIR.md`
- Minimum import-path repair lock / first valid head: `c3a0f05200c1d75b3c325ed2aabc9d1cc7d51dc0`
- First valid canonical run: `34431286455`
- Job: `102727137722`
- Artifact: `10134581843`
- Artifact name: `rb-r27c2-realized-target-quality-forensic-v1`
- Artifact digest: `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`
- Exact R27B V2 parent artifact verified: `10134023092`, digest `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- Exact R27C parent artifact verified: `10134387002`, digest `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`

## Integrity

PASS.

- 8429 parent rows preserved.
- Overall targeted player-game PBP join coverage: `0.9963347587`.
- Minimum primary-cohort targeted join coverage: `0.9898843931`, above frozen 0.98 minimum.
- 2023 vacancy RB1 targeted join coverage: 1.000.
- No model fit.
- No candidate projection.
- Target-game PBP used only as retrospective labels.
- Sportsbook inputs: 0.
- Production changed: false.
- R26 changed: false.
- R22 changed: false.

## Primary physical finding — 2023 vacancy RB1 YPT failure is YAC/YPR compression, not catch/depth/screen-frequency collapse

Compare 2023 vacancy RB1 incumbents with non-2023 vacancy RB1 incumbents.

### Opportunity / output

2023:
- PBP targets mean: `3.1892`
- receptions mean: `2.5000`
- receiving yards mean: `16.4730`

Non-2023:
- targets mean: `3.5921`
- receptions mean: `2.8531`
- receiving yards mean: `21.0862`

### Catch rate

- 2023 actual catch rate: `0.78345`
- non-2023 actual catch rate: `0.78724`
- difference: `-0.00379`

Catch conversion is effectively unchanged and therefore does not explain the YPT gap.

### Target depth

- 2023 actual air yards/target: `0.0887`
- non-2023: `0.1065`
- difference: `-0.0178` yards/target

Target depth is effectively unchanged.

### Screen / behind-LOS rate

- 2023: `0.58059`
- non-2023: `0.56231`
- difference: `+0.01828`

No meaningful collapse in target-shape frequency is visible.

### Explosive-20 target frequency

- 2023: `0.05046`
- non-2023: `0.05097`
- difference: `-0.00051`

Explosive-play *frequency* is essentially identical.

### YAC / YPR / YPT

- 2023 YAC/reception: `7.3395`
- non-2023 YAC/reception: `8.0667`
- difference: `-0.7272`

- 2023 YPR: `6.4254`
- non-2023 YPR: `7.4206`
- difference: `-0.9952`

- 2023 YPT: `5.2236`
- non-2023 YPT: `5.9001`
- difference: `-0.6765`

- 2023 max receiving gain: `10.6812`
- non-2023: `13.1952`
- difference: `-2.5140`

Thus the physical degradation is concentrated downstream of the catch: lower YAC / lower yards per reception / shorter realized maximum gains despite essentially unchanged catch rate, target depth, screen frequency and explosive-target frequency.

## Production arithmetic decomposition confirms the same mechanism

On targeted 2023 vacancy RB1 rows:
- actual catch rate: `0.78345`
- production catch rate: `0.77220`
- catch-rate residual: `+0.01125`

- actual YPR: `6.42540`
- production implied YPR: `7.57740`
- YPR residual: `-1.18777`

- actual YPT: `5.22361`
- production YPT: `5.83933`
- YPT residual: `-0.61572`

For non-2023 vacancy RB1:
- actual YPT: `5.90014`
- production YPT: `5.90292`
- YPT residual: `-0.00278`

So the 2023 problem is not that production catch rate is too optimistic; actual catch rate is slightly higher. The miss is almost entirely in the value of completed catches relative to production implied YPR.

## Prior-to-realized drift explains why V2 did not foresee it

2023 vacancy RB1:
- YAC/reception drift versus strict-prior player YAC: `-0.48288`

Non-2023 vacancy RB1:
- YAC/reception drift versus strict-prior player YAC: `+0.21439`

Difference in YAC drift is about `-0.6973` yards/reception.

By contrast:
- air-yards drift is similarly negative in both cohorts;
- screen-rate drift is similar;
- explosive20-rate drift is essentially the same.

This is consistent with a 2023 vacancy-RB1 **realized YAC/YPR compression regime that was not encoded in the frozen prior context features**. This is a mechanism hypothesis requiring new pregame-identifiable explanatory variables before any predictive candidate can be justified.

## Tail finding — separate high-YAC/explosive phenomenon

The four vacancy-RB1 rows that V2 moved from <30-yard AE under B1 to >=30-yard AE under C1 are physically very different from the 2023 central-efficiency failure:

- actual catch rate: `0.9583`
- actual YAC/reception: `22.5`
- explosive20 target rate: `0.375`
- actual YPR: `23.075`
- actual YPT: `22.667`
- explosive20 share of receiving yards: `0.4764`
- max receiving gain: `30.75`

All four have exact PBP joins. These are genuine explosive outcomes dominated by post-catch production, not evidence that ordinary receiving-yard means should be raised broadly.

The 44 RB1 rows that were 30+ misses under both B1 and C1 also show a high-YAC/high-explosive state:
- YAC/reception `11.677`
- explosive20 target rate `0.1478`
- YPT `9.436`
- explosive20 receiving-yard share `0.4285`
- max gain `28.045`

This reinforces the existing architectural separation:
- point-mean research should explain repeatable pregame target value;
- R22 remains the authority for stochastic right-tail redistribution.

## What is retained from R27B V2

The mixed/fail V2 candidate is not thrown away. Its valid signals remain evidence:
- pooled vacancy MAE modestly improved;
- p90 improved materially;
- RB2+ improved;
- 4/6 seasons improved;
- all-RB and Week1 stayed stable/slightly better;
- contextual features helped localize where target-quality information matters.

These facts can motivate future hypotheses but cannot be cherry-picked directly into production from the same evaluated sample.

## Forensic disposition

`R27C2_FORENSIC_COMPLETE_PHYSICAL_TARGET_QUALITY_HYPOTHESIS_IDENTIFIED`

The identified hypothesis is:

> In vacancy-active lead-back states, the unresolved receiving-yard mean error can arise when newly/correctly allocated RB1 opportunities are converted at a lower post-catch value than production implied YPR; the 2023 failure is specifically consistent with an unforecast YAC/YPR compression regime, while the remaining catastrophic right-tail misses are a separate high-YAC/explosive phenomenon already conceptually owned by R22.

This is not yet a predictive mechanism. The next study must determine whether the YAC/YPR compression regime is **pregame-identifiable using genuinely new, strict-prior football information** rather than reusing the V2 historical YAC/team/opponent averages that already failed to anticipate it.
