# M96A — RB Opportunity vs Efficiency Attribution — Authoritative Results

## Run identity

- Workflow: `M96A RB Opportunity-Efficiency Attribution`
- Authoritative run: **`33459376333`** (Run #2)
- Job: **`99706110345`**
- Tested SHA: **`9e3b152e7e756b8a29798ef82cfa9a0730c51f89`**
- Branch: `research-rb-m96a-opportunity-efficiency-attribution`
- Artifact: `migration-96a-rb-opportunity-efficiency-attribution`
- Artifact ID: **`9782611047`**
- Artifact SHA256: **`a2d4f99b9e7b3f2b75c8694e79c6856dd09e4b5ee9921b2d53488e89cdab3d6e`**
- Execution: **SUCCESS**
- Scientific disposition: **`JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS`**
- Model fit: `0`
- Feature search: `0`
- Coefficient search: `0`
- Sportsbook inputs: `0`
- Production change: `0`

## Mechanical history

Run #1 (`33459156556`, job `99705449700`, SHA `df723092d2f37e16df02d767223a02fbe4d047d7`) was green but exposed a truth-source completeness issue before the result was accepted. The M95F research trace had only 1,340 nonmissing rushing-yard truth rows because 53 low-volume RB/FB player-games were not enriched with yardage truth there, while the authoritative M94C trace contained complete rushing-yard truth for all 1,393 rows.

No candidate formula, routing threshold, workload arm, or scientific rule changed. A mechanics-only wrapper (`evaluate_rb_m96a_opportunity_efficiency_attribution_v2.py`) made M94C the authoritative rushing-yard truth source while preserving M95F/M95I as frozen workload sources. On the 1,340 rows where M95F yard truth existed, it matched M94C exactly. Run #2 is therefore authoritative.

Source parity on Run #2:

- M94C/M95F/M95I joined rows: **1,393 / 1,393 (100%)**
- M94C yard truth nonmissing: **1,393**
- M95F yard truth nonmissing: `1,340`
- shared rushing-yard truth max difference: **0.0**
- actual carry truth max difference: **0.0**
- M94C carry-projection parity max difference: `1.07e-14`

## Frozen M96A architecture

M96A did not train a new model. It evaluated three already-learned workload representations against the same frozen M94C effective rushing-efficiency forecast:

1. **M94C central carries** — the conservative point workload reference.
2. **M94C + M95F distribution expectation** — the frozen M95F empirical hurdle distribution collapsed to its expectation for point-yard sensitivity; quantiles were retained separately.
3. **M94C + M95F + M95I vacancy branch** — M95F expectation for incumbents, frozen M95I carry transform only in pregame vacancy rows.

Primary efficiency was the exact finite M94C-implied effective YPC:

`candidate_rush_yards / candidate_rush_att`

This reproduces M94C's own frozen rushing-yard prediction exactly. The older 2–7 YPC clamp was retained only as a sensitivity check; only 16 rows changed and the MAE effect was negligible.

## Primary point-yard results — 2025 all RB

| Workload arm | Rush-yard MAE | RMSE | Bias | Corr | Mean prediction |
|---|---:|---:|---:|---:|---:|
| M94C central | **21.0312** | **29.8615** | +0.708 | .6016 | 37.37 |
| M95F distribution expectation | 22.4455 | 30.4311 | +5.436 | **.6048** | 42.09 |
| M95F + M95I vacancy branch | 22.3967 | 30.4773 | +5.086 | .6018 | 41.74 |

Actual mean rushing yards were `36.66`.

This is an important confirmation of the M95 lesson: **the retained tail information should not be collapsed into one universally higher expected carry/yard number.** M95F's distribution expectation improves true high-workload games but overstates ordinary-game yardage enough to worsen aggregate MAE.

The M95I vacancy point branch partially removes that overstatement but does not beat M94C globally. This does not invalidate M95I's previously validated vacancy *probability/ranking* evidence; it means the frozen deterministic vacancy carry transform is not a universal point-yard improvement.

## Tail slices — the post-M94C workload information is still useful

M95F distribution expectation versus M94C:

- actual 20+ carry games (`n=98`): rush-yard MAE **40.0051 -> 36.3540** (`+3.6511` yards)
- actual 25+ carry games (`n=24`): **49.3105 -> 44.8607** (`+4.4498` yards)

But ordinary workloads worsen:

- actual 0–5 carries: **13.2884 -> 15.9972**
- 6–10: **21.1908 -> 22.9559**
- 11–14: **25.8123 -> 26.6864**
- 15–19: essentially flat/slightly worse, `29.7637 -> 29.8117`

Thus the M95 workload distribution contains real upper-tail information, but the **distribution expectation is not the correct replacement for the central point forecast**.

## Opportunity-vs-efficiency oracle attribution

The precommitted routing rule used the M94C arm only.

### All RB (`n=1,393`)

- pregame M94C rush-yard MAE: **21.0312**
- with perfect actual carries but frozen pregame efficiency: **13.3535**
- opportunity MAE recovery: **7.6777 yards**
- with perfect game efficiency but frozen projected carries: **14.3055**
- efficiency MAE recovery: **6.7256 yards**
- recovery difference: **0.9520 yards** in favor of opportunity
- opportunity is the larger absolute residual component in **59.73%** of games
- efficiency is larger in **40.27%**

The frozen route required a >=1.0-yard recovery advantage plus >=55% component dominance. Opportunity clears the component-share gate but misses the yard-margin gate by only **0.048 yards**. Therefore the precommitted result is **JOINT**, not opportunity-dominant.

This is not a null result. Both components independently recover roughly seven yards of MAE under oracle conditions, so both are meaningful remaining bottlenecks.

## The bottleneck changes by workload regime

M94C oracle MAE recovery by actual carry slice:

| Slice | Pregame MAE | Perfect carries MAE | Perfect efficiency MAE | Opportunity recovery | Efficiency recovery |
|---|---:|---:|---:|---:|---:|
| 0–5 | 13.288 | 5.245 | 10.027 | **8.043** | 3.261 |
| 6–10 | 21.191 | 13.561 | 13.188 | 7.630 | **8.002** |
| 11–14 | 25.812 | 19.270 | 15.208 | 6.542 | **10.605** |
| 15–19 | 29.764 | 23.749 | 16.989 | 6.015 | **12.775** |
| 20+ | 40.005 | 28.636 | 36.409 | **11.369** | 3.596 |
| 25+ | 49.310 | 37.549 | 54.390 | **11.762** | -5.079 |

Interpretation:

1. **Low-volume and huge-volume games are primarily opportunity problems.** This is exactly where carry-count misses create large yardage misses.
2. **Middle/high-normal workloads (roughly 11–19 carries) are much more efficiency-sensitive.** Once workload is approximately known, YPC/environment/explosiveness drives much of the remaining yardage error.
3. The 25+ perfect-efficiency oracle can worsen point MAE when projected carries remain severely compressed. Feeding realized YPC through a badly underpredicted carry count does not repair the missing workload. This reinforces—not weakens—the upper-tail opportunity diagnosis.
4. Stable workhorses (`n=254`) are efficiency-sensitive overall: opportunity recovery `8.481`, efficiency recovery **12.536**. Their remaining yardage problem is not solely a bellcow-carry problem.
5. Vacancy rows remain joint: M94C opportunity recovery `7.286`, efficiency recovery `6.621`.

## Residual-component decomposition

For M94C across all 1,393 rows:

- mean absolute opportunity component: **15.156 yards**
- median: **11.246**
- mean absolute efficiency component: **13.353 yards**
- median: **7.583**
- opportunity-dominant share: **59.73%**
- efficiency-dominant share: **40.27%**
- opportunity component correlation with total signed residual: `.7015`
- efficiency component correlation with total signed residual: `.7103`
- residual identity max numerical error: `2.84e-14`

Both components are therefore structurally material.

## Rushing-yard tail discrimination

Using point/expectation yard scores:

### 75+ rushing yards (`202` events)

- M94C AUC: **`.809799`**
- M95F expectation: `.809242`
- hybrid vacancy arm: `.808473`

### 100+ rushing yards (`95` events)

- M94C: **`.808085`**
- M95F expectation: `.807388`
- hybrid: `.805831`

Collapsing M95F's workload distribution to its expectation therefore does not improve tail ranking. M95F should be used as a distributional input, not merely an inflated point score.

## M95F carry-quantile to yard-quantile translation

With efficiency held deterministic at frozen M94C implied YPC:

- p50 coverage: `64.11%` vs 50% nominal — central distribution too conservative/high
- p75: `78.61%` vs 75% nominal
- p90: `87.72%` vs 90% nominal — upper tail undercovers by `2.28 pp`
- p95: `91.60%` vs 95% nominal — upper tail undercovers by `3.40 pp`

This is consistent with two simultaneous facts: M95F adds too much central expected tail mass when collapsed to a mean, while deterministic efficiency fails to express all high-yardage variance. The next yardage layer should therefore preserve **separate workload and efficiency distributions**, not merge everything into one higher expected carry count.

## Scientific disposition

**`JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS`**

M96A answers the user's concern directly:

- Carries are **not** finished or irrelevant. Perfect workload would recover about `7.68` rushing yards of MAE overall and more than `11` yards in actual 20+/25+ games.
- Efficiency is also not a secondary afterthought. Perfect efficiency would recover about `6.73` yards overall and is the larger lever in 11–19 carry games and stable-workhorse yardage generally.
- The positive M95 tail work is real but should live in the **workload distribution**, not overwrite M94C's central point estimate.
- The next RB model should be a joint opportunity × efficiency distributional synthesis.

## Required next migration — M96B

**M96B — RB Joint Workload × Efficiency Distribution Synthesis**

Precommitted direction:

1. Keep M94C central carries as the point/central opportunity anchor.
2. Preserve M95F as a separate workload-tail/distribution signal; do not substitute its mixture expectation for the M94C center.
3. Preserve M95I vacancy/role-transition probability evidence as a separate regime candidate; do not silently apply its deterministic transform to incumbents.
4. Build the efficiency side beginning with the retained M95C finding that offensive/blocking/environment information was the stable mean signal, while runner-created/explosive features were more useful for upside.
5. Model rushing yards as **workload distribution × efficiency distribution**, with explicit 75+/100+ calibration and ordinary-game MAE/RMSE guards.
6. Include targeted diagnostics for 0–5 and 20+/25+ workloads, because M96A shows these are the most opportunity-sensitive slices.
7. No sportsbook inputs in football-model construction and no production promotion before temporal/prospective validation.

M96B must not reopen generic M95 tail coefficient search. If it demonstrates that a specific carry error remains dominant after joint synthesis, only that demonstrated opportunity failure may be reopened.