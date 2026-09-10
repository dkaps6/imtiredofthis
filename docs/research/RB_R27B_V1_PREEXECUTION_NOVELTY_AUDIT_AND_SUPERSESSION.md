# RB R27B V1 — PRE-EXECUTION NOVELTY AUDIT / SUPERSESSION RECORD

**Status:** SUPERSEDED BEFORE WORKFLOW / CANDIDATE EXECUTION / SCIENTIFIC RESULTS

## Why this record exists

R27 V1 isolated a real downstream RB receiving-yard problem after exact R26 opportunity improved targets/receptions and aggregate receiving-yard error but failed RB1/p90/2023 stability. An initial R27B V1 plan was frozen and implementation began. Before any workflow or candidate execution, the user explicitly challenged whether this efficiency avenue duplicated prior RB work. The repository was audited before proceeding.

This record preserves that interruption as part of the scientific lineage. No R27B V1 result exists and no R27B V1 scientific conclusion may ever be inferred from the existence of its plan or scripts.

## Exact pre-execution V1 lineage

- R27 result-record parent: `886c8432ff811882e006d84a61385862e3be7839`
- V1 frozen-plan commit: `c9c7906748065f25cf7cb0d5ce24259144465dc5`
- V1 feature-dataset builder commit: `7ec3ccf53607e42a5bfdfbc1e2e6596f0c3d913c`
- V1 evaluator commit: `a55e42ea7a7eaf7f318024ea0e4c1d7c4688f640`
- V1 branch: `research-rb-r27b-receiving-efficiency-v1`
- R27 parent run: `34423546037`
- R27 parent artifact: `10132290573`
- R27 parent digest: `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`

There is no R27B V1 workflow in this branch, no R27B V1 workflow run, no R27B V1 artifact, no R27B V1 gate result, and no R27B V1 candidate metric.

## Audit findings

### Already present in production

`scripts/modeling/bayesian_v2.py` already produces `bayes_ypt` through empirical-Bayes shrinkage using position-family priors plus prior/current player evidence. `scripts/modeling/simulation_rules.py` already uses this player efficiency state and applies the existing football matchup/pass-efficiency multiplier to create `rules_ypt`.

Therefore a new candidate whose novelty is mainly career/recent YPT persistence or generic YPT shrinkage would duplicate production concepts rather than add a new football signal.

### Already tested scientifically in R23

R23 Candidate 3 already tested a strict-prior receiving-efficiency construction on improved opportunity:

`expected receptions × shrunk YPR`

It used frozen 6-game recent and 16-game stabilizing histories with empirical shrinkage. R23 improved targets/receptions but receiving-yard MAE worsened overall; RB1 receiving-yard MAE worsened about 2.68% and p90 error worsened about 3.35%. R23 is a preserved scientific null for that combined opportunity + shrunk-YPR mechanism.

Therefore R27B must not rebrand recent/career YPR or simple player-history efficiency shrinkage as a new hypothesis.

### Already decomposed in earlier receiving diagnostics

`scripts/backtest/decompose_receiving_error.py` explicitly decomposed receiving error into target opportunity, catch conversion and YPT error. It was diagnostic, not a novel YPT feature model. This confirms that merely identifying YPT as a remaining error source is not new research by itself.

### R24 and R27 context

R24 intentionally removed R23's failed new YPR component and paired improved opportunity with existing production YPT. R27 then repeated that decomposition using the exact R26 opportunity mechanism that later qualified for receptions. R27 showed material aggregate support but failed RB1/p90/2023 robustness.

The justified next question is therefore not `does player efficiency history matter?`; that avenue already exists in production and was directly tested in R23. The justified question is whether pregame football context not already represented in those mechanisms explains residual receiving efficiency.

### R19/R22 do not close the novel mean lane

R19/R22 are receiving-yard tail/distribution authorities. R19 tail scoring uses opportunity/mean/tail-state features such as baseline projected targets, baseline receiving-yard mean, prior RB-room share, R8/R9 identity state and frozen YPT. It does not establish a point-mean model from RB target depth, YAC style, screen usage, QB checkdown environment or RB-specific opponent receiving vulnerability. R22 remains mean-preserving.

## V1 overlap decision

The following planned V1 feature families are ruled out as the primary source of novelty for the next study:

- career-to-date YPT
- current-season-to-date YPT
- trailing-4/trailing-8 YPT
- career catch-rate persistence
- career YPR persistence
- generic empirical-Bayes YPT/YPR re-shrinkage
- a generic learned correction whose predictive content can be explained primarily by the above

These may remain only as existing production state/control variables if mathematically necessary to predict a *residual to production YPT*. They may not constitute the claimed new information.

## Novel information frontier authorized for V2

The next study may investigate only information demonstrably incremental to the existing production/R23/R24/R27 path:

1. strict-prior RB target depth / air-yards-per-target;
2. strict-prior YAC-per-reception / YAC style;
3. strict-prior screen or behind-line-of-scrimmage target rate;
4. strict-prior explosive receiving-play propensity;
5. strict-prior team/QB RB checkdown environment (RB targets per official team pass attempt / related football-only tendency);
6. strict-prior team RB target-shape/YAC environment where not reducible to player raw YPT persistence;
7. strict-prior opponent RB-specific receiving vulnerability, especially YAC, catch rate, target depth and explosive allowance.

All must be sportsbook-free and cutoff-safe. Source/schema availability must be audited before candidate execution. Unavailable features are removed only through a documented pre-execution mechanical source audit, never because they score poorly.

## Supersession

R27B V1 is stopped and preserved. Its frozen plan is not edited or retuned. Its implementation scripts are research-only historical evidence and are not authorized for execution.

The authorized continuation is a separately frozen V2 branch beginning from the R27 result-record parent, excluding the V1 implementation from its ancestry where practical:

`research-rb-r27b-v2-novel-efficiency-context`

Production authority remains `bb76ba9eabb08e2f0875a9af49301c3877f4141f`. R26 and R22 remain unchanged.