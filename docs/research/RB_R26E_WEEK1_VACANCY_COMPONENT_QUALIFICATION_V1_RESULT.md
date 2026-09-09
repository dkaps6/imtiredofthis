# RB R26E — Week 1 Vacancy Component Qualification V1 Result

Status: **WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research head: `0bc405ff324e34c8e413177ef6f3d1fe6361e4b5`

## Authoritative lineage

- Workflow: `RB R26E Week 1 Vacancy Component Qualification V1`
- Run: `34364872300`
- Job: `102510879589`
- Artifact: `10109398212`
- Artifact digest: `sha256:4d136cf20bccebbdd874b24392d95669d11ad01e55b4ac4af6cdee06e7e99651`
- Parent R26 artifact: `10106271075`
- Parent R26 digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- Workflow conclusion: mechanically `success`
- Scientific disposition: **WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW**
- Frozen gates passed: **17 / 18**

R26 V1 remains `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`. R26E authorizes no production change and no 2026 prospective shadow.

## Supported components to preserve

The Week-1 R26 mechanism remains strongly supported pooled and across most seasons.

### Week-1 vacancy incumbents (`n=246` labeled rows)

Receptions:
- MAE: `1.424485 -> 1.312299` (**7.88% better**)
- RMSE: `1.899368 -> 1.741118`
- bias: `-0.852810 -> -0.445509`
- p90 absolute error: `3.347325 -> 2.975309`

Targets:
- MAE: `1.687842 -> 1.588760` (**5.87% better**)
- RMSE: `2.259780 -> 2.094701`
- bias: `-1.020503 -> -0.495419`

### Week-1 role protection

RB1 vacancy incumbents (`n=95`):
- receptions MAE `1.803491 -> 1.588733` (**11.91% better**)
- RMSE `2.359021 -> 2.055480`
- bias `-1.318895 -> -0.260018`
- p90 `4.022272 -> 3.407286`

RB2+ vacancy incumbents (`n=151`):
- receptions MAE `1.186037 -> 1.138384` (**4.02% better**)
- RMSE `1.541472 -> 1.510169`
- p90 `2.632654 -> 2.408982`

### Global Week-1 RB/FB safety (`n=509`)

- receptions MAE `1.312603 -> 1.235284` (**5.89% better**)
- RMSE `1.837827 -> 1.758467`
- bias `-0.537235 -> -0.401004`

### Temporal replication

Week-1 vacancy-incumbent receptions MAE improved in **5 of 6 seasons**:

- 2020: `1.303640 -> 1.420534` (**8.97% worse**)
- 2021: `1.800590 -> 1.631473` (**9.39% better**)
- 2022: `1.670218 -> 1.606050` (**3.84% better**)
- 2023: `1.198932 -> 1.095979` (**8.59% better**)
- 2024: `1.498819 -> 1.234223` (**17.65% better**)
- 2025: `0.959650 -> 0.730193` (**23.91% better**)

All six seasons had at least 20 labeled Week-1 vacancy-incumbent rows.

## Failed / unsupported component

The only failed frozen gate was:

`11_no_w1_season_worsens_more_than_2pct = false`

because 2020 worsened 8.97%.

2020 is unusual in shape: R26 dramatically corrected negative bias (`-0.616 -> -0.167`) while MAE, RMSE, and p90 worsened. That is consistent with an **overcorrection / within-room allocation problem**, not evidence that the general Week-1 vacancy mechanism lacks signal.

This interpretation is diagnostic only until a no-refit 2020 forensic study is frozen and executed.

## Next child-candidate delta

Under `RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`, preserve the 17 passing Week-1 components and investigate only the 2020 failure.

Next step: **R26F 2020 Week-1 Vacancy Failure Forensic Atlas**.

R26F must not refit or regenerate R26. It must compare 2020 against the five successful Week-1 seasons using immutable R26 prediction/source state and predeclared dimensions such as:
- within-room allocation error versus room-total error;
- RB1 versus RB2+;
- R9 reliability / calibrated residual magnitude;
- prior-history availability;
- exits/entrants and room-turnover intensity;
- new-to-team / no-prior-NFL state;
- candidate target-share movement and overcorrection direction.

The goal is to identify a reproducible pregame mechanism that distinguishes the harmful 2020 regime. R26F may not simply exclude 2020, label it an outlier, or weaken the 2% gate.

## Do-not-change list

Preserve:
- original R26 R9 identity mechanics;
- broad Week-1 vacancy concept;
- finite RB target-pool conservation;
- production-exact non-vacancy behavior;
- non-RB exactness;
- receiving-yard mean exactness;
- R22 authority;
- sportsbook separation;
- strict-prior leakage protections;
- all negative 2020 evidence;
- R26/R26D/R26E failed/mixed dispositions.
