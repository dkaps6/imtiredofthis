# RB Receiving Research Handoff — R23 through R26H

Status: ACTIVE
Last updated: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Current research branch: `research-rb-r26h-week1-balanced-turnover-role-state-atlas-v1`
Current frontier: freeze and test R26I Week-1 selective-restoration child.

## Standing governance

This lane follows `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`.

A failed full candidate does not imply all of its mechanics are discarded. Supported components are inherited into the next candidate whenever their contribution can be isolated, while the original failed run and failed gate remain preserved exactly. Do not weaken frozen gates, redefine cohorts post hoc, or discard independently supported football logic merely because a full candidate misses one protection.

After RB is resolved/bounded, perform a parked component-salvage audit of older failed/mixed QB/WR/TE work for useful subcomponents that may previously have been discarded under all-or-nothing reasoning.

## Production authority that must not be disturbed

- RB rushing: `RB_P3_SYNTHESIS_V1` Week-1 production authority.
- RB receiving distribution/tail: R22 production-certified and main-active.
- R22 preserves receiving-yard means and receptions; it changes only certified RB receiving-yard distribution/tail behavior.
- Current RB receiving entitlement/receptions/receiving-yard mean remain unresolved research lanes.

## R23/R24/R25 summary

- R23 found useful target/reception signal but receiving-yard mean/tail behavior was not robust. No promotion.
- R24 preserved production YPT while testing entitlement. Targets/receptions improved, but receiving-yard gates failed. No promotion.
- R25 isolated a receptions specialist. Low-history MAE improved, but RMSE/p90/bias showed harmful compression. No promotion.

Cumulative inference: do not force one model to solve targets, receptions, receiving-yard mean, and tail simultaneously. R22 already owns tail shape.

## R26 V1 — parent mechanism to preserve, full candidate not promoted

Canonical:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- gates passed `19 / 20`

Supported components:
1. Non-vacancy RB rooms remain exact production baseline.
2. Vacancy is a legitimate pregame football state.
3. R9 receiving identity is generally useful for redistributing a fixed RB receiving pool.
4. RB-room target mass, team entitlement, non-RB entitlement, receiving-yard production means, and R22 remained exact.
5. Sportsbook inputs upstream and future/target-game leakage remained zero.
6. Global receptions improved modestly; pooled vacancy-incumbent targets/receptions improved.
7. Week-1 behavior was strong.
8. Vacancy RB1 underprojection was materially repaired pooled.
9. Vacancy-incumbent reception MAE improved in 5 of 6 seasons.

Failure:
- 2023 vacancy-incumbent receptions MAE `1.174518 -> 1.227870`, about 4.54% worse;
- violated frozen no-season-worsens-more-than-2% gate;
- no shadow.

Do not change R9 coefficients, conservation, receiving-yard means, R22, sportsbook separation, or leakage contract to rescue R26.

## R26B — 2023 forensic atlas

Run `34360534200`.
Disposition `FORENSIC_MECHANISM_IDENTIFIED`.

Findings:
- 2023 Week 1 was not the failure;
- damage concentrated Weeks 2+, especially late-season churn;
- failure was primarily within-room player allocation/order, not total RB-room receiving volume;
- generic vacancy remained useful, but later-season roster churn could falsely activate the mechanism.

## R26C — exited-player significance source audit

Canonical:
- run `34361409319`
- artifact `10108036449`
- digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`
- disposition `EXIT_SIGNIFICANCE_SOURCE_READY`

Integrity:
- 685/685 parent vacancy team-weeks reconstructed;
- 910 exited RB rows;
- prior receiving-history coverage 89.23%;
- vacancy team-weeks with at least one exited-player history signal 89.34%;
- target-game outcomes/participation 0;
- sportsbook 0;
- same-week historical depth false;
- production parameters unchanged.

## R26D — exit significance × R26 effect

Canonical:
- run `34364089085`
- artifact `10109078127`
- digest `sha256:827d45611a0ff588093f2ab6a5b74eb739dd5602afa9461a713083a8c44e8756`
- disposition `EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER`
- 7/8 replication gates passed.

Frozen meaningful-exit definition:
`max prior targets/game > 1 OR max prior RB-room target share >= 0.25`
Do not retune.

Meaningful exits pooled (`n=936`): receptions MAE `1.304162 -> 1.280611` (1.81% better), RMSE/bias/p90 improved; targets MAE 2.02% better.
Low exits (`n=342`): receptions MAE worsened 0.91%, targets MAE worsened 0.92%.
RB1 meaningful improved 2.70%; RB2+ improved 1.17%.

All-season significance router failed because meaningful-exit MAE improved in only 3/6 seasons versus frozen 4/6 requirement.

Critical phase finding:
- Week-1 low exit: 7.22% better;
- Week-1 meaningful exit: 7.62% better;
- Week-1 unknown history: 11.67% better;
- Weeks 2+ low exits worsened 2.18%; meaningful exits ~flat.

Inference: Week 1 and Weeks 2+ are distinct transition regimes.

## R26E — Week-1 component qualification

Canonical:
- branch `research-rb-r26e-week1-qualification-v1`
- execution head `81203a00995164c3a57a83f683c5b29fec07bdbe`
- run `34368268224`
- job `102522540620`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- gates passed `19 / 20`.

Pooled Week-1 vacancy incumbents:
- receptions MAE `1.4244850719 -> 1.3122994738` (~7.88% better)
- RMSE `1.8993682539 -> 1.7411176321`
- bias `-0.8528102135 -> -0.4455093698`
- p90 `3.3473247651 -> 2.9753086015`
- targets MAE `1.6878420656 -> 1.5887600865`

RB1 ~11.9% better; RB2+ ~4.0% better.
All-RB Week-1 reception MAE ~5.89% better.

Five of six seasons improved. Sole failure:
- 2020 `1.30363975 -> 1.42053430`, +8.97% worse;
- frozen cap was +5%; no shadow.

## R26F — 2020 Week-1 failure forensics

Canonical:
- run `34365496225`
- artifact `10109658591`
- digest `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`
- disposition `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED`

2020 failure was within-room allocation, not room total:
- room-total reception absolute-error delta sum `-0.7960466935` (improved)
- summed player reception absolute-error delta `+3.3146046536` (worse)

Replicated harmful state: balanced turnover, `EXITS_EQ_ENTRANTS`.
In 2020 (`n=26`) R26 reception MAE worsened ~7.93%; this state explained ~51.22% of 2020 net worsening. Harmful direction also replicated in 2021 and 2023.

R26F authorized only the design of R26G; no shadow/production.

## R26G — blanket balanced-turnover guard

Canonical:
- run `34365957361`
- artifact `10109840313`
- digest `sha256:0523055f7b39500d5d6b9fbc7ed85185cae41bf4518889e7f6c58104021c15ca`
- disposition `WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW`

Child logic:
- non-vacancy -> baseline;
- vacancy exits != entrants -> original R26;
- balanced turnover exits == entrants -> whole-room baseline fallback.

R26G partially repaired 2020 (`+8.97%` R26 harm reduced to about `+4.37%`) but the fallback was too broad and erased real R26 gains, particularly in 2024/2025. Therefore balanced turnover is a legitimate risk state, but not every balanced room should fall back.

## R26H — balanced-turnover role-state atlas

Canonical:
- execution head `2f16ebc403306991247d2dbf220e21373226ab76`
- run `34369024680`
- job `102525128622`
- artifact `10111095834`
- digest `sha256:c3fa00de4262bb98b1f73d6931005f963d0ded4cc9a9fe53734ae421e34ef18f`
- result doc `docs/research/RB_R26H_WEEK1_BALANCED_TURNOVER_ROLE_STATE_ATLAS_V1_RESULT.md`
- disposition `BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL`
- shadow false; production false.

Integrity:
- 72 balanced Week-1 rooms;
- 162 balanced Week-1 incumbent rows;
- exit-state coverage 1.0;
- no prediction regeneration/refit/new target-game features/sportsbook/production changes.

R26H reused the frozen R26D meaningful-exit threshold and existing strict-prior incoming-state flags.

### Child-design eligible states

1. `MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT` (`n=37`)
- reception MAE `1.4871922808 -> 1.3648610282` (~8.22% better)
- RMSE/bias/p90 improved;
- target MAE improved;
- beneficial direction replicated outside 2020 in 4 seasons.

2. `MEANINGFUL_EXIT + NO_PRIOR_ENTRY_PRESENT` (`n=77`)
- reception MAE `1.1445967352 -> 1.0901142742` (~4.76% better)
- RMSE/bias/p90 improved;
- targets essentially flat/slightly better;
- beneficial direction replicated outside 2020 in 3 seasons.

Unsupported due insufficient support/replication:
- `LOW_EXIT + VETERAN_ENTRY_PRESENT` (`n=6`)
- `LOW_EXIT + NO_PRIOR_ENTRY_PRESENT` (`n=13`)
- `UNKNOWN_EXIT_HISTORY` (`n=7`)

Interpretation: R26G's blanket balanced-room fallback was too blunt. In balanced rooms with a **meaningful departed receiving role** and a supported incoming role state, original R26 should be selectively restored. Unsupported balanced states remain conservative until separately supported.

## Current frontier — R26I Week-1 Selective Restoration

R26H authorizes the design of R26I only. Freeze before scoring.

Proposed exact child inheritance:
1. non-vacancy -> production baseline exact;
2. unbalanced vacancy -> original frozen R26 exact;
3. balanced turnover:
   - `MEANINGFUL_EXIT + VETERAN_ENTRY_PRESENT` -> original R26 exact;
   - `MEANINGFUL_EXIT + NO_PRIOR_ENTRY_PRESENT` -> original R26 exact;
   - unsupported balanced states -> baseline exact.

R26I must:
- construct predictions only by selecting exact parent baseline/R26 values; no new fit;
- keep the R26D significance threshold frozen;
- keep R9 frozen;
- preserve RB room mass, non-RB, receiving-yard means, and R22;
- use sportsbook inputs 0 and no target-game features;
- retain R26E/R26G seasonal safety gates or stronger, especially the 2020 guard;
- if it passes, maximum next status is a 2026 Week-1 prospective shadow candidate, not production.

## Weeks 2+ later-season lane

After Week-1 component is resolved/bounded, separately study Weeks 2+ vacancy/churn routing using strict-prior current-season recency/role/availability. Do not force the Week-1 offseason-transition rule onto in-season churn.

## Parked items

- 2026 ESPN/nflverse current hierarchy audit: parked until RB receiving lane is resolved/bounded.
- RotoBaller Week-1 WR/CB article parser: pinned for later CB-matchup work; weekly URL-driven ingest, underlying assets/data before OCR.
- Prior-position component-salvage audit: after RB, review old failed/mixed QB/WR/TE experiments for independently supported subcomponents.
