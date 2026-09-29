# Weeks 1-3 Residual Science Reconciliation V1

Date: 2026-09-29  
Status: **COMPLETE — BOUNDED RECONCILIATION, NO PRODUCTION CHANGE**  
Branch: `research-week3-postmortem-execution-v1`

## Purpose

The completed 2026 Weeks 1-3 postmortem found:
- 629-611, 50.7%, -40.72u;
- model MAE 17.55 vs selected line MAE 16.36;
- low mean bias (-5.36), concentrated most heavily in RB / rush+receiving;
- severe probability overconfidence (mean stated fair probability ~68.7% vs 50.7% realized; 70-100% band mean ~81.0% vs 51.8% realized);
- no simple market/position/side slice surviving game-cluster-aware BH-FDR.

Before creating new science, this note reconciles those findings against already-closed or active research so the next experiment does not recycle a dead idea.

## 1. Probability / confidence

Already established historically:
- the legacy `component_sd` Normal translator was not production-equivalent and was defective;
- production itself uses empirical simulated outcomes for `fair_prob`, so that legacy translator defect is not the live explanation;
- empirical-MC distributions were historically under-dispersed, but frozen widening materially reduced rather than eliminated overconfidence;
- pure heldout isotonic probability calibration collapsed STRONG coverage but **worsened heldout ROI in both directions**, so scalar probability recalibration is closed as a production candidate;
- the 3pp probability-edge condition was historically algebraically non-binding after the 5% EV hurdle on the tested cohort.

What Weeks 1-3 newly add:
- severe overconfidence is now observed on the actual 2026 production route, not only the old historical proxy;
- therefore the live problem survives the historical translator repair and cannot be dismissed as a `component_sd` artifact;
- but the failed heldout isotonic candidate means the next step must **not** be another scalar recalibration, threshold tweak, or global width rescue.

Residual open question:
> Are there pregame production-authority states in which the model's own final projection is demonstrably less trustworthy, rather than merely globally overconfident?

## 2. Mean construction

The Weeks 1-3 projection-authority attribution result is `MIXED_BY_MARKET_OR_POSITION`:
- final authority materially helps QB passing and rushing;
- receiving means are essentially unchanged from upstream MC/ensemble authority;
- rush+receiving equality in archived `mc_proj -> model_proj` is expected because RB Rush+Receiving Conservation V2 acts upstream of the recorded MC mean.

Implications:
- do not roll back the final stack globally;
- do not reopen closed WR/TE/RB residual or width families under new names;
- do not reinterpret the RB-PD2 Week-3 forward lock before its frozen support floor;
- the large RB rush+receiving low bias remains a real current-season concern, but its active conservation architecture and forward width lane stay protected.

## 3. Specialist RNG

The specialist RNG lane is a separate finite-simulation reproducibility issue, not evidence of a football-mean candidate. It must be repaired to exact frozen research parity before any interpretation of its downstream board fingerprint.

Do not use RNG repair results as a Weeks 1-3 outcome-driven model change.

## 4. GSIS

The private Week-3 Lineup Detail / Formation Usage snapshot is baseline-only. It cannot be used to fit or backtest a Week-3 outcome explanation after the fact. Repeated immutable temporal snapshots are required before predictive use.

## Residual science frontier

The cleanest genuinely new question left by the current evidence is **projection-authority conflict at the betting line**:

> When upstream MC and the final production authority lie on opposite sides of the sportsbook line, does that disagreement identify a materially unreliable forecast state?

This is distinct from:
- raw edge-size slicing;
- `component_sd` as outcome variance;
- QB-PD3 component-disagreement / synthesis-cap diagnostics;
- pure probability calibration;
- market/position/side carveouts.

It tests a pipeline-seam state: whether late football authority changes the *direction* of the mean relative to the quoted line.

The first pass is descriptive on the already-settled Weeks 1-3 board only and cannot authorize a rule. A separate frozen prospective or clean historical confirmation would be required before any production action.
