# Weeks 1-4 Production Failure-Mode Atlas V1

Status: **DIAGNOSTIC COMPLETE — NO PRODUCTION CHANGE**  
Date: 2026-10-06  
Branch: `research-week4-postmortem-execution-v1`

Canonical source:
- Week-4 postmortem run `37485600879` = SUCCESS
- artifact `11423535484`
- digest `sha256:59185d2aa4178971f6b0cc806eddd172ca0e7fbae559ad44da37a2b99656d791`
- exact frozen Weeks 1-3 graded artifact was concatenated with newly graded Week 4; Weeks 1-3 were not regraded
- decided rows: **1,667**

This atlas is diagnostic only. It fits no model, searches no betting
threshold, uses no new sportsbook pull and authorizes no production change.

## 1. The aggregate low bias is mostly a catastrophic right-tail problem, not a universal center shift

All decided rows:
- model mean error: **-5.61**
- model median error: **-0.43**
- model MAE: **17.89**
- market-line MAE: **16.86**
- model closer than line: **46.07%**

The mean is materially negative while the median is near zero. A relatively
small number of very large outcomes above the model drag the mean downward.
Therefore a universal +5/+6 yard recentering is not justified by this record.

### Standardized tail concentration

Define for diagnosis only:

`z = (actual - model_proj) / model_sd`

No Normal-distribution assumption is required for this diagnostic.

Rows with `|z| > 2`:
- **270 / 1,667 = 16.20%**
- actual above model: **93.33%**
- selected side UNDER: **83.70%**
- loss rate: **82.59%**
- units: **-178.96u**

Rows with `|z| > 3`:
- **116 / 1,667 = 6.96%**
- actual above model: **98.28%**
- selected side UNDER: **85.34%**
- loss rate: **87.07%**
- units: **-86.10u**

For context, the outcome-defined complement `|z| <= 2` finished:
- 1,397 rows
- 57.41% win rate
- **+134.86u**

That complement is **not a pregame betting selector** because membership is
defined using the realized outcome. It is evidence that catastrophic
underestimated upside outcomes account for the entire aggregate unit loss.

## 2. The tail miss is strongly asymmetric by market

Count of realized residuals beyond +/-2 model SD:

| Market | n | actual > model +2SD | actual < model -2SD |
|---|---:|---:|---:|
| pass_yards | 101 | 8 | 5 |
| rec_yards | 581 | **89** | **4** |
| receptions | 556 | **68** | **0** |
| rush_rec_yards | 136 | **33** | **2** |
| rush_yards | 293 | **54** | **7** |

The receiving/reception/rushing families are not merely too narrow
symmetrically. Realized extreme misses overwhelmingly occur on the **high side**
of the football mean.

This helps explain why a symmetric generic widening can improve calibration
without solving the betting problem. It does not, by itself, authorize an
asymmetric-tail candidate.

## 3. Stated confidence gets more extreme exactly where tail failure grows

Selected-side fair-probability bands:

| Stated band | n | mean stated | realized | >2SD miss rate | model closer rate |
|---|---:|---:|---:|---:|---:|
| <.55 | 192 | 51.49% | 40.10% | 10.42% | 52.08% |
| .55-.60 | 295 | 57.43% | 50.17% | 13.90% | 46.78% |
| .60-.65 | 289 | 62.51% | 52.25% | 12.46% | 47.40% |
| .65-.70 | 238 | 67.53% | 51.68% | 13.87% | 47.90% |
| .70-.80 | 354 | 74.55% | 52.82% | 15.54% | 45.76% |
| .80-.90 | 201 | 84.35% | 53.73% | 21.39% | 40.80% |
| .90-1.00 | 98 | **94.19%** | **56.12%** | **42.86%** | **35.71%** |

Proper-score diagnosis across all 1,667 selected bets:
- model fair probability Brier: **0.28629**
- constant 0.50 Brier: **0.25000**
- selected book no-vig probability Brier: **0.24996**
- model fair probability log loss: **0.80865**
- constant 0.50 log loss: **0.69315**
- selected book no-vig log loss: **0.69309**
- model fair-probability AUC: **0.5380**

Interpretation:
- there is a small amount of ordinal/ranking information in the model
  probability;
- the magnitude is badly miscalibrated;
- the most extreme stated probabilities are also the population with the
  largest standardized projection failures.

This reinforces the existing closure that raw `fair_prob` / raw edge is not a
validated confidence or staking variable.

## 4. The late projection stack is not the primary source of aggregate error

Across all 1,667 decided rows:
- raw MC MAE: **18.42**
- final production MAE: **17.89**
- paired improvement: **+0.53 yards**

Week 4:
- raw MC MAE: **19.75**
- final production MAE: **18.89**
- paired improvement: **+0.86 yards**

Week-4 pass yards specifically:
- raw MC MAE: **70.86**
- final QB production MAE: **61.76**
- paired improvement: **+9.10 yards**

Therefore the Week-4 Burrow/Goff-type explosions are not evidence that the
promoted QB synthesis should be removed. The final stack improved the base
football mean even though large realized tails remained.

Receiving yards are different:
- all-weeks MC MAE: **22.73**
- final MAE: **22.75**

The late stack is essentially neutral there, so receiving-yard failures are
largely born upstream of the final projection stage.

## 5. Rush+receiving remains a true center-of-distribution problem

All rush+receiving rows:
- mean error: **-16.44 yards**
- median error: **-8.82 yards**

RB-only:
- mean combo error: **-16.60**
- median combo error: **-8.82**

On RB rows with standalone component outcomes available:
- rushing component mean error: **-10.95**
- rushing component median error: **-3.67**
- receiving component mean error: **-4.04**
- receiving component median error: **-0.12**

The combined-market low bias is therefore driven primarily by the rushing
authority, with a smaller receiving contribution.

This is structurally different from the overall-board mean-vs-median pattern and
remains a football-mean research target.

## 6. Week-4 live-state trace

The exact Week-4 pregame artifact shows that PlayerForm current-season rush
share often updated faster than the downstream Bayesian/rules opportunity state.

Examples:
- Bijan Robinson: current .635 -> PlayerForm blend .616 -> rules .542
- Jonathan Taylor: current .795 -> PlayerForm blend .758 -> rules .647
- Chuba Hubbard: current .612 -> PlayerForm blend .458 -> rules .407
- Aaron Jones: current .612 -> PlayerForm blend .530 -> rules .470

This is consistent with the already-completed
`BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED` result.

However, the exact obvious mean replacement was already tested in
Opportunity Authority Priority V1 and **FAILED CLOSED**. That candidate may not
be rescued as an RB-only post-hoc variant.

The new authorized research question is therefore narrower and genuinely
different: whether PlayerForm-vs-Bayes opportunity disagreement is an
**uncertainty signal**, without moving the mean.

A separately frozen historical diagnostic is now running on branch:
`research-opportunity-state-conflict-uncertainty-v1`.

## 7. Current root-cause disposition

The four highest-value problems are now separated:

1. **probability magnitude / confidence** — severely overconfident;
2. **right-tail / uncertainty representation** — catastrophic high-side misses
   dominate aggregate loss;
3. **RB rush+receiving mean** — structurally low, driven primarily by rushing;
4. **receiving-yard upstream authority** — late projection stages do not repair
   the base receiving-yard error.

Do not collapse these into one global correction.

No universal mean boost, arbitrary UNDER exclusion, top-N edge rule, global
probability rescale or generic symmetric widening is authorized from this atlas.
