# RB R27D — Strict-Prior YACOE Residual V1 First Valid Scientific Result

Status: `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`

This is the first valid R27D scientific result. It is immutable evidence and does not authorize integration or production change.

## Canonical execution

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen plan commit: `69edc12a16c691e3838eadcd75559b85dbba7865`
- Frozen plan blob: `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- First valid head: `641b25419c4f5ff3c234d1c000222fb4909ef940`
- Run: `34436178615`
- Job: `102741600329`
- Artifact: `10136250846`
- Artifact name: `rb-r27d-strict-prior-yacoe-residual-v1`
- Artifact digest: `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- Exact R27B V2 parent artifact: `10134023092`
- Exact parent digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Preserved pre-valid failures

Run1 mechanical failure:
- run `34435661834`
- job `102740069405`
- head `1757d7549c0fc8c44ad207f55971fdd8c3eab754`
- duplicate deterministic Week1 column collision
- no model fit / no scientific result
- record: `RB_R27D_RUN1_WEEK1_COLUMN_COLLISION_MECHANICAL_REPAIR.md`

Run2 integrity/scoring failure:
- run `34435897671`
- job `102740771822`
- head `775d9bae7f264ae343209a75cd4947bc88994da9`
- artifact `10136169669`
- digest `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`
- parent rows with null outcomes polluted metric arrays, producing NaN primary scores
- emitted disposition rejected; no scientific decision
- record: `RB_R27D_RUN2_OUTCOME_NULL_SCORING_MECHANICAL_REPAIR.md`

## Integrity

PASS.

- all 18 frozen integrity/structural gates passed
- exact parent verified
- B0/B1 reproduce parent exactly
- reception/YPR bridge max gap `1.7763568394002505e-14` <= `1e-10`
- exact RB1 application scope: 565 projected rows
- every outside-scope row C1 == B1 exactly
- vacancy RB2+ C1 == B1 exactly
- all six 2020–2025 outer folds present
- training seasons strictly before each test season
- sportsbook inputs: 0
- target/future PBP features: 0
- correction max absolute magnitude: `1.29285319249` <= frozen `1.5`
- production changed: false
- R26 changed: false
- R22 changed: false

Observed-outcome scoring universe is 8,429 RB rows, matching the established R27C2 outcome-known universe.

## Primary RB1 result

VACANCY_RB1_INCUMBENT, n=503 observed outcomes:
- B0 MAE: `14.305919`
- B1 MAE: `14.709395`
- C1 MAE: `14.710366`
- C1 change vs B1: `+0.006600%` worse
- frozen requirement: >=1.00% improvement — FAIL
- C1 remains `+0.404446` yards MAE worse than B0 — FAIL

RB1 RMSE:
- B1: `19.863416`
- C1: `19.900574`
- FAIL non-worse gate

RB1 p90 AE:
- B1: `29.064913`
- C1: `29.555785`
- FAIL non-worse gate

RB1 30+ miss rate:
- B1: `0.095427`
- C1: `0.093439`
- PASS safety gate; slight improvement

RB1 bias:
- B1: `-0.079846`
- C1: `-0.166481`
- absolute-bias worsening `0.086634`, within frozen 0.25 tolerance — PASS

## 2023 pre-identified stress cohort

2023 vacancy RB1, n=74:
- B0 MAE: `12.349471`
- B1 MAE: `13.992139`
- C1 MAE: `14.055947`
- C1 change vs B1: `+0.456024%` worse
- frozen >=2% improvement gate — FAIL
- C1 remains `+1.706475` yards worse than B0 — FAIL

Thus the strict-prior YACOE residual model did not repair the specific 2023 lead-back failure that motivated the study.

## Season stability

Vacancy RB1 C1 vs B1 MAE:
- 2020: `16.223181 -> 16.477643` (`+1.568506%`, worse)
- 2021: `13.470760 -> 13.374817` (`-0.712228%`, better)
- 2022: `17.226542 -> 16.931502` (`-1.712705%`, better)
- 2023: `13.992139 -> 14.055947` (`+0.456024%`, worse)
- 2024: `14.873969 -> 14.969923` (`+0.645117%`, worse)
- 2025: `12.349578 -> 12.310219` (`-0.318705%`, better)

Only 3/6 seasons improved; frozen requirement was 4/6 — FAIL.
No season worsened by more than 2%; worst was +1.5685% — PASS.

## Aggregate / Week1 safety

VACANCY_ACTIVE, n=1761:
- B1 MAE `11.159268`
- C1 MAE `11.159545`
- C1 worse by `0.000277` yards — frozen exact non-worse gate FAIL.

ALL_RB, n=8429:
- B1 MAE `10.967611`
- C1 MAE `10.967669`
- C1 worse by `0.000058` yards — frozen exact non-worse gate FAIL.

WEEK1, n=509:
- B1 MAE `10.830435`
- C1 MAE `10.815477`
- change `-0.138113%` (better) — PASS.

## Scorecard

- gates passed: `22 / 31`
- integrity gates: `18 / 18 PASS`
- scientific gates: `4 / 13 PASS` (gates 27, 28, 30, 31)
- scientific pass: false
- disposition: `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`

## Scientific interpretation

R27D answered the specific question it was designed to answer: strict-prior relative YAC-over-expected state, modeled conservatively from player/offense/opponent xYAC history, does **not** provide a reliable mean correction for the vacancy-incumbent-RB1 receiving-yard translation problem.

There are weak/local positives (Week1, 2021, 2022, 2025, 30+ miss rate), but the frozen architecture explicitly forbids post-result routing or feature/cohort cherry-picking. The candidate therefore cannot be integrated or retuned from this evaluated sample.

Combined with R23, R24, R27, R27B V2 and the R27C/R27C2 forensics, this materially narrows the RB receiving-yard mean frontier: generic historical YPR/YPT/YAC, raw target-shape context and now strict-prior xYAC/YACOE persistence have all failed to produce a qualifying lead-back mean correction. Future RB receiving-mean work must be based on genuinely new pregame information/mechanism rather than another transformation of these same historical efficiency families.

R26 opportunity/receptions remain production authority. R22 remains receiving-yard tail authority. No R27D integration design is authorized.
