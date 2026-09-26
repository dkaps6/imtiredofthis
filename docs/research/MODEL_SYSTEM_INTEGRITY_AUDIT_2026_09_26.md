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

## 3. Weighting / blending failure investigated

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
- reopen closed WR participation, Rush Pool, TE Width, or post-ensemble reconciliation families;
- use Week-3 outcomes to redesign the already-frozen vacancy transfer;
- use sportsbook information upstream.

The goal is no longer "find another correlated feature." The goal is to identify where correct football information is lost, distorted, double-counted, or inconsistently recombined before final projections.
