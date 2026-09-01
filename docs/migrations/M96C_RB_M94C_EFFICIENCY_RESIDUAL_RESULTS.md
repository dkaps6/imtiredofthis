# M96C — M94C-Anchored RB Efficiency Residual Synthesis — RESULTS

## Authoritative run

- workflow: `M96C RB M94C Efficiency Residual`
- run: **`33462888850`**
- job: **`99716610968`**
- tested SHA: **`708f9ff23b96cde8e023b6317fcaec30b76e76b0`**
- artifact: **`9783799265`**
- artifact name: `migration-96c-rb-m94c-efficiency-residual`
- artifact SHA256: **`6109a8b3afc6d2fdb963db9149bf3fb238cc476e291bf743cc4b496ad39abf72`**
- execution: **SUCCESS**
- disposition: **`M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED`**
- next step: **`M96D_PRECOMMITTED_CONDITIONAL_EFFICIENCY_ROUTING_AUDIT`**
- feature search `0`; weight search `0`; hyperparameter search `0`; sportsbook `0`; production change `0`

## Source / protocol integrity

M96C did not invent a 2024 M94C player-level rushing-yard point. The frozen M94C artifact persists that player-level yard point only for 2025, so M96C used strict expanding-week 2025 out-of-fold evaluation:

- test Weeks 6–18;
- each week trained only on earlier 2025 weeks;
- M94C carries and M94C central rushing-yard point frozen;
- correction target was an efficiency/YPC residual, not a carry residual;
- training rows required at least 5 actual carries;
- training residual winsorization/clipping used training-only 5th/95th percentiles;
- Ridge alpha fixed at `10.0`;
- X tail-only logistic `C=0.1` fixed.

Source coverage:

- M95D 2025 rows `1,357`; joined `1,340`; coverage **98.7472%** — PASS vs >=97% source gate.
- M94C 2025 rows `1,393`; joined `1,340`.
- exact yard-truth parity max diff `0.0`.
- exact carry-truth parity max diff `0.0`.

Available feature counts:

- E blocking/environment: `14`
- P player-created efficiency: `8`
- D opponent run efficiency/resistance: `16`
- X explosive/upside: `16`

## Global point results — Weeks 6–18 OOF (`n=961`)

Frozen M94C C baseline:

- MAE **`21.5719`**
- RMSE `30.4500`
- bias `+0.3820`
- correlation `.6045`

Point residual arms:

| Arm | MAE | MAE gain vs C | RMSE | RMSE gain vs C | Bias | Corr |
|---|---:|---:|---:|---:|---:|---:|
| C | 21.5719 | — | 30.4500 | — | +0.3820 | .6045 |
| E | 21.5063 | +0.0656 | 30.7008 | -0.2509 | -1.1636 | .5951 |
| P | 21.4261 | +0.1458 | 30.6153 | -0.1654 | -1.2354 | .5971 |
| D | **21.3474** | **+0.2245** | **30.4341** | **+0.0159** | -1.2799 | .6039 |
| E+P | 21.5880 | -0.0161 | 30.7906 | -0.3407 | -1.1116 | .5915 |
| E+D | 21.5676 | +0.0043 | 30.7155 | -0.2655 | -1.2216 | .5951 |
| P+D | 21.4684 | +0.1035 | 30.6069 | -0.1569 | -1.3289 | .5976 |
| E+P+D | 21.6526 | -0.0807 | 30.7987 | -0.3488 | -1.1452 | .5916 |

No arm reached the frozen `>=0.25` all-RB MAE-gain requirement. D came closest and was the only arm with non-negative RMSE gain.

Late Weeks 13–18 showed the same directional ordering:

- C MAE `21.3513`
- E `21.2567`
- P `21.1519`
- D **`21.1301`**
- P+D `21.1766`

So D was not merely an early-season artifact. But global point retention still failed because of regime damage.

## The important M96C finding: efficiency signal is workload-conditional

D was the best global arm, but its effect changed sign by actual workload:

| Actual carries | C MAE | D MAE | D gain |
|---|---:|---:|---:|
| 0–5 (`n=387`) | 13.4145 | **12.7837** | **+0.6308** |
| 6–10 (`n=197`) | 21.8220 | **21.3148** | **+0.5071** |
| 11–14 (`n=176`) | 25.7482 | **25.0106** | **+0.7376** |
| 15–19 (`n=126`) | **29.5957** | 30.3544 | **-0.7587** |
| 20+ (`n=75`) | **39.7267** | 41.8936 | **-2.1669** |
| 25+ (`n=21`, diagnostic) | **50.8255** | 52.1686 | -1.3432 |

P and E showed the same broad pattern: modest-to-material improvement in low/middle workload buckets, then damage in higher workloads. This caused every point arm to fail the frozen non-degradation gate; worst gated-slice regression was >2 yards for every arm.

This is a meaningful scientific result, not a mechanical failure:

- opponent/environment/player efficiency information contains real predictive information;
- a universal efficiency residual correction is not safe across workload regimes;
- high-workload yard misses remain dominated by opportunity/tail-state miss mechanics, consistent with M96A;
- low-to-middle workloads are where efficiency corrections appear most useful;
- therefore the next architecture should route an efficiency expert using **pregame workload-state information**, not apply one correction universally.

Do not use actual carries as a future router; actual carries are postgame diagnostic truth only. Any M96D gate must be defined from pregame M94C/M95F/role-state variables and frozen before outcome evaluation.

## E/P/D capability interpretation

### E — blocking/environment

- all-RB MAE improved only `+0.0656`;
- RMSE worsened `0.2509`;
- improved low-workload buckets but damaged 15+/20+;
- status: **CONDITIONAL_CLUE**, not global retain.

### P — player-created efficiency

- all-RB MAE improved `+0.1458`;
- RMSE worsened `0.1654`;
- strongest 11–14 carry improvement among simple E/P/D arms (`+0.8176` MAE), but damaged 15+/20+ and 25+ materially;
- status: **CONDITIONAL_CLUE**, not global retain.

### D — opponent run efficiency/resistance

- best all-RB arm: MAE `21.5719 -> 21.3474` (`+0.2245`);
- RMSE slight improvement `+0.0159`;
- late Weeks 13–18 MAE improved `+0.2212`;
- strong low/mid workload gains, but true 20+ games regressed `2.1669` yards;
- status: **CONDITIONAL_CLUE** and strongest candidate for precommitted routing research.

### Combinations

None of E+P, E+D, P+D, or E+P+D improved enough to justify additive stacking. The largest feature combination was worse than C globally. This reinforces the M96B lesson: positive blocks are not automatically additive.

## X — explosive/upside tail-only audit REJECT

Expanding-week probability models compared C (M94C yard point only) vs C+X.

75+ yards, Weeks 6–18 (`152/961` events):

- C AUC `.806478`, Brier `.114757`, logloss `.364843`
- C+X AUC `.800355`, Brier `.115719`, logloss `.366235`
- AUC change `-0.006124`; Brier gain `-0.000962`

100+ yards (`74/961`):

- C AUC `.790822`, Brier `.067047`, logloss `.238820`
- C+X AUC `.785170`, Brier `.067255`, logloss `.241086`
- AUC change `-0.005652`; Brier gain `-0.000209`

Late Weeks 13–18 also did not rescue X. The isolated explosive feature block does not improve the already-strong M94C yard-point ranking and is rejected in this additive/logistic form.

This does not contradict prior M95D native-model upside evidence; it reinforces that some explosive information may be interaction-dependent rather than separable.

## Casebook interpretation

D often corrected obvious low/mid-workload overprojections in the right direction, for example reducing large M94C point errors in several 11–18 actual-carry games. But it also pulled down genuine high-output games such as Derrick Henry, James Cook, Saquon Barkley and De'Von Achane when M94C was already underprojecting the game. This is exactly why a universal D adjustment is unsafe.

## M96C disposition

**`M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED`**

Keep:

- C/M94C as the global central carry and yard point;
- E/P/D as conditional efficiency evidence only;
- D as the strongest simple conditional clue;
- M95F workload distribution and M95I vacancy evidence remain separate workload/role capabilities from prior work.

Reject:

- universal E/P/D corrections;
- all tested additive E/P/D combinations;
- isolated X tail increment.

## Next migration — M96D

**M96D — Pregame Conditional Efficiency Routing Audit**

Primary question:

> Can a precommitted pregame workload/role-state gate identify the player-games where an M94C-anchored efficiency correction (especially D, with E/P as controlled alternatives) should be active, preserving the low/mid-workload gains without damaging high-workload games?

Rules:

- no actual carries or postgame outcomes in the router;
- no reopening arbitrary carry-tail coefficient search;
- use pregame M94C projected carries, M95F workload-tail probability/distribution, stable-workhorse/vacancy role state, and other already-validated workload state only;
- freeze a small diagnostic gate grid before evaluation; no matchup-by-matchup hand selection;
- compare C vs routed D, routed P, and only predeclared compatible routed combinations;
- require global incremental MAE/RMSE value plus no material high-workload/tail degradation;
- 2025 remains development evidence; any winner requires prospective 2026 confirmation before production.
