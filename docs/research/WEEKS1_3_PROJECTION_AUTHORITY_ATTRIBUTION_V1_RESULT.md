# Weeks 1-3 Projection Authority Attribution V1 — Result

Status: COMPLETE — DIAGNOSTIC ONLY  
Date: 2026-09-29  
Branch: `research-week3-postmortem-execution-v1`  
Authority run: `36626121661`  
Artifact: `11060101978`  
Artifact digest: `sha256:8833b5611cb979d585932d94ea071b26d9dfbdad3784c196381ff376681d23e3`

Frozen plan:
`docs/research/WEEKS1_3_PROJECTION_AUTHORITY_ATTRIBUTION_V1_PLAN.md`

## Disposition

`MIXED_BY_MARKET_OR_POSITION`

No production change is authorized.

## Overall MC -> final

On 1,240 settled selected bets:

- MC MAE: 17.9683
- final-model MAE: 17.5501
- paired absolute-error improvement: +0.4182
- MC signed bias: -8.0912
- final signed bias: -5.3625
- rows materially moved: 840
- movement-toward-actual rate on moved rows: 56.19%

Game-cluster bootstrap for mean paired absolute-error improvement:
- mean: +0.4182
- 95% CI: [-0.0631, +0.8862]
- clusters: 45

Therefore the pooled final-vs-MC improvement is directionally positive but not
cleanly separated from zero at this three-week support level.

## By market — MC -> final

- pass_yards: 56.8491 -> 53.2781 MAE; +3.5710 yd improvement;
  bias -22.1508 -> -1.9749.
- rush_yards: 21.4290 -> 20.2504 MAE; +1.1786 yd improvement;
  bias -13.9377 -> -7.7309.
- rec_yards: 21.7300 -> 21.7600 MAE; -0.0299 yd (slightly worse);
  bias -7.5169 -> -6.4960.
- receptions: 1.6903 -> 1.6739 MAE; +0.0164 improvement;
  bias -0.7755 -> -0.6066.
- rush_rec_yards: 31.7461 -> 31.7461 MAE; no observable movement in the
  archived `mc_proj -> model_proj` comparison.

Important rush+receiving interpretation:
for Week 3, RB Rush+Receiving Conservation V2 replaces the combo draw array
before `mc_proj` is recorded in pricing. Therefore equality between
`mc_proj` and `model_proj` is expected and does **not** imply V2 failed to
apply. Week-3 selected RB combo rows correctly carry the V2 application flag.

## By position — MC -> final

- QB: 36.9178 -> 34.6215 MAE; +2.2964 improvement.
- RB: 19.4276 -> 18.9650 MAE; +0.4626 improvement.
- WR: 13.4796 -> 13.4502 MAE; +0.0294 improvement.
- TE: 11.7917 -> 11.8497 MAE; -0.0580 (slightly worse).

The final stack's visible current-season value is concentrated in QB and rushing
means. Receiving means are largely unchanged from their upstream authority.

## Ensemble -> final

Overall:
- ensemble MAE: 17.7203
- final MAE: 17.5501
- paired improvement: +0.1702
- ensemble bias: -6.6033
- final bias: -5.3625

The largest final-over-ensemble change is again QB pass yards:
- 55.7245 -> 53.2781 MAE
- bias -20.5753 -> -1.9749

Most receiving-market final means are effectively identical to ensemble means.

## Application-flag cohorts

Application cohorts are descriptive only and must not be interpreted causally
because the flags are strongly confounded with market and position.

The corrected audit distinguishes:
- APPLIED
- NOT_APPLIED
- UNAVAILABLE (field did not exist on older board versions)

Week-3 production flags are present and correctly parsed for:
- RB Rush+Receiving Conservation V2
- Discrete Count Mean Alignment V1

No evidence in this audit invalidates either production-active lock.

## Interpretation

The cumulative low-bias problem is not being created uniformly by late-stage
production logic. Upstream MC is already substantially low-biased, and the
final stack reduces that bias meaningfully for QB passing and rushing, while
leaving receiving means mostly untouched.

This makes a global rollback of synthesis/ensemble/state logic unsupported.

The next useful work is not another generic calibration or width experiment.
The remaining frozen Week-3 prospective authorities must be graded exactly as
preregistered, especially:
- RB Vacancy Opportunity V1 + DEN/PIT public-intent concentration labels;
- Receiving Rule Semantics V1 frozen A/B cells;
- Availability -> Opportunity descriptive behavior;
- Week-3 outcome attachment to the 46 RB-PD2 forward locks.

No postgame redesign is allowed while grading those frozen objects.
