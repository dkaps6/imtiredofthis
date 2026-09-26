# Model System Integrity Audit — 2026-09-26

Status: **ACTIVE PARALLEL SYSTEMS-AUDIT LEDGER — NO MODEL CHANGE**

Purpose: preserve the post-feature-hunt pivot toward finding implementation, mathematical, weighting, rule, and architecture errors that can prevent a heavily researched model from converting good upstream information into accurate final player projections.

This ledger does not reopen closed migrations and does not authorize production changes by itself.

## 1. Confirmed implementation / data-lineage bugs already found and repaired

These are literal plumbing/correctness defects, not weak hypotheses:

- historical rebuild workflow referenced a nonexistent script name ('historical_player_logs_m95q.py') instead of the existing historical player-log builder;
- historical schedule/player-log reconstruction exposed an invalid 2020 Week-18 assumption in a shared historical path;
- 2026 production audit found hidden 2025 TeamForm contamination inside an otherwise season-parameterized run;
- production previously allowed model-universe drift between PlayerForm/simulation and current availability/pricing authority;
- runtime NFL week could change mid-run across UTC midnight before the frozen job-scoped week repair;
- partial-slate production previously retained a stale full-week CIN control assertion after CIN had correctly left the live universe;
- strict production dependency auditing had omitted imported dependencies and could miss hidden stale logic.

These bugs matter because a model can appear statistically sophisticated while silently evaluating or producing the wrong football state.

## 2. Confirmed mathematical / architecture contradictions

### 2.1 RB rush-att / rush-yards cross-market inconsistency

'POST_SPECIALIST_CROSS_MARKET_CONSISTENCY_V1_CONTRADICTION_CONFIRMED'

The joint MC layer preserves an RB/FB player's implied YPC under proportional opportunity changes, but downstream independent market-specific ensemble blends do not preserve that same relationship.

Week-3 no-odds diagnostic:
- MC implied-YPC vs rules-YPC median absolute gap: **0.0086**
- final ensemble implied-YPC vs MC median absolute gap: **0.4896**
- p90: **1.2172**
- max: **1.9739**

This is not merely noise; it proves that separately blended 'rush_att' and 'rush_yards' can produce a final football state inconsistent with the opportunity/efficiency state that generated them.

### 2.2 Finite rushing-volume contradiction

On 7/30 Week-3 teams, summed final RB/FB rush-attempt means exceeded the entire team MC rush-attempt mean.

This demonstrates that downstream player-market means are not constrained to a single conserved team-volume state.

### 2.3 Historical MC player-carry mass shortfall

In preserved 2024-2025 full-stack evidence, actual top-five rushers account for about **99.5%** of team carries, while reconstructed MC player carry mass captured only about **70-71%** of actual team carries on average.

This explains why a mathematically neat reconciliation back to MC mass failed badly: the final independent ensembles were partially compensating for an upstream MC/reconstruction shortfall.

Current Week-3 production does **not** show the same broad residual signature (29/30 teams sit near the intended 5% simulator residual), so this is treated as a historical-reconstruction/architecture warning, not permission to normalize live Week-3 shares upward.

## 3. Weighting / blending errors investigated

### 3.1 Missing receiving-market ensemble weights — found and partially corrected

A component audit found that production once had calibrated weights only for 'pass_yards', 'rush_att', and 'rush_yards'. 'rec_yards', 'receptions', and 'rush_rec_yards' silently fell back to 100% MC.

This was a genuine architecture gap rather than a code crash.

A clean 2023-fit / blind-2024-2025 holdout showed real point-accuracy improvement for all three previously unweighted markets. The interaction test with the improved empirical probability translator then showed:
- the heldout mean improvements were real;
- the old probability translator was also genuinely wrong;
- combining the two did **not** uniformly improve ROI, so a blanket three-market promotion was not justified.

Current production has since promoted blind-heldout weights for:
- 'rec_yards': MC 0.659889 / ML 0.292427 / State 0.047684;
- 'receptions': MC 0.554082 / ML 0.444566 / State 0.001352.

'rush_rec_yards' was **not** given an independent standalone weight. For non-Week-1 RBs it is instead governed by RB Rush+Receiving Conservation V2, which constructs the combined mean from the already-final standalone rushing and receiving components.

Therefore this specific missing-weight seam is **not an unfinished current bug**. Do not reopen the old three-market weight-fit experiment unless a new production defect is found.

### 3.2 Rush post-ensemble reconciliation — contradiction real, repair failed

'RUSH_POST_ENSEMBLE_RECONCILIATION_V1_FAILED_CLOSED'

A parameter-free repair forced final player carry mass back to joint-MC player carry mass while preserving player implied YPC exactly.

Mechanics passed to numerical precision, but prediction quality worsened badly in both 2024 and 2025 for:
- all-player rush attempts;
- all-player rush yards;
- RB rush attempts;
- RB rush yards;
- RB rush+receiving;
- QB rushing.

Interpretation: the independent ensemble inconsistency is real, but simply forcing all markets back onto the current MC mass is **not** the solution. Do not rescue with partial factors, carveouts, caps, floors, or position exclusions.

## 4. Rule / hierarchy transmission errors investigated

### WR role transmission
A direct audit tested whether current participation information was failing to displace stale WR hierarchy anchors.

Result: when the M38 anchor and strict-prior participation leader disagreed, the participation leader was **25.47 percentage points worse** at identifying the actual top-target WR and averaged **1.498 fewer actual targets**.

Conclusion: there is no hidden easy fix where "more current participation" should simply replace the current WR anchor.

### Rush pool evidence guard
A research-only rush-pool signal qualified locally but failed exact full-stack integration. It improved some RB yardage metrics but worsened broader rush-attempt/rush-yard science, including QB outcomes.

Conclusion: do not repair the rushing system through ad hoc membership/top-N/evidence-state rules.

## 5. One confirmed production improvement from this audit era

'RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFIED'

For non-Week-1 RB 'rush_rec_yards', final combined draw = final-mean-aligned rushing draw + final-mean-aligned receiving draw.

Historical 2024-2025:
- MAE **27.6853 -> 25.5718**
- RMSE **39.8437 -> 36.4030**
- p90 absolute error **64.7742 -> 57.2604**

This is an example of the systems-audit philosophy succeeding: a structural relationship was enforced without fitting a new predictive feature.

## 6. Active open structural question

The strongest unresolved question is now **how real personnel vacancies propagate through finite opportunity**.

RB Vacancy Opportunity V1 is already frozen prospectively for Week 3:
- DEN: Jonah Coleman unavailable
- PIT: Rico Dowdle unavailable

A separate public-intent capture is also frozen:
- DEN: 'ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION'
- PIT: 'WARREN_LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT'

These must remain untouched until games are final.

## 7. Parallel research rule going forward

We will continue two complementary lanes without conflating them:

1. **Systems integrity lane**
   - search for contradictions between team state, player opportunity, efficiency, market-specific means, specialist layers, and final ensemble outputs;
   - distinguish literal bugs from model assumptions;
   - require a frozen repair before scoring;
   - reject mathematically elegant repairs that degrade blind historical accuracy.

2. **New-information / vacancy lane**
   - complete RB Vacancy V1 exactly as frozen;
   - grade public-intent labels only after Vacancy V1 itself is graded;
   - prospectively accumulate attributable intent evidence across future qualifying events before considering any fitted use.

## 8. Anti-loop rules

Do not:
- launch another broad feature sweep merely because current production improvement is limited;
- retune ensemble weights without a preregistered structural reason;
- repair one market in isolation if it creates a cross-market contradiction;
- force conservation to an upstream authority that is itself empirically biased;
- reopen closed WR participation, Rush Pool, TE Width, old three-market weight-fit, or post-ensemble reconciliation families;
- use Week-3 outcomes to redesign the already-frozen vacancy transfer;
- use sportsbook information upstream.

The goal is no longer "find another correlated feature." The goal is to identify where correct football information is lost, distorted, double-counted, or inconsistently recombined before final projections.


## 9. Post-ensemble routing/order audit — 2026-09-26

Disposition: **NO NEW LIVE ROUTING BUG FOUND / DUPLICATE SEAMS CLOSED**

The current production wrapper chain was read directly from main:
- stable V3 entrypoint -> V6 production wrapper;
- V6 -> V5 -> V4 -> V3 promoted entitlement/C2 stack -> V1 football-universe wrapper -> run_pricing_v2;
- non-Week-1 RB Rush+Receiving Conservation V2 is activated around the V5 parent and stamped after pricing.

Verified:
- Week-3 RB rush+receiving V2 rebuilds the combo draw from final-mean-aligned standalone rush and receiving draws before sportsbook comparison;
- V6 asserts final model_proj equals the V2 target and target equals rush+receiving component sum;
- Week-1-only RB P3/R22/R26 routes explicitly no-op outside Week 1 rather than silently reusing Week-1 science;
- TE-R5P and WR-R15 are applied upstream of joint MC, then the existing ensemble is applied downstream exactly as production documents;
- the possibility that receiving specialists are diluted by downstream ML/State blending was already tested in the frozen PR #549 production-order historical replay, so it is not a new untested seam;
- the football simulation universe intentionally remains full-league / football-only while kickoff eligibility narrows downstream pricing. The visible 32-team requirement is therefore not, by itself, a current modeling defect.

Historical benchmark warning preserved:
- prior research infrastructure did contain a real post-ensemble injection bug for QB synthesis: an early unified historical benchmark wrote QB synthesis into mc_proj and re-blended it, unlike production where synthesis replaces the final ensemble mean. That benchmark bug was repaired before interpretation and is not present in current production run_pricing_v2.

Conclusion:
The current post-ensemble routing/order path does not justify a new repair candidate. Do not reopen this seam without new contradictory evidence.

### Next systems-integrity seam

Move one layer upstream to **availability -> opportunity propagation**:
- does definitive unavailability remove the player from the correct football competition state?
- does vacated target/carry mass become residual, normalize mechanically, or reach plausible successors?
- are different positions/markets handling vacancy consistently?
- diagnostic first, no candidate variants and no Week-3 outcome use.

This directly complements, but must not alter, the already-frozen RB Vacancy Opportunity V1 Week-3 experiment.
