# PRISM WR/TE Player Component-Dominance Routing V1 — Frozen Source / Anti-Retest Plan

Date: 2026-10-18

Status: **FROZEN BEFORE PREDICTIVE SCORING / PLAYER-SPECIFIC MECHANISM ROUTING ONLY / NO PROJECTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1`

## 1. Why this is the next lane

The active signed-opportunity frontier is preserved fail-closed:

- WR opportunity signed component bias remains QUALIFIED, but exact WR-R15 authority recovery failed immutable byte identity;
- TE opportunity signed component bias remains QUALIFIED, but the already-frozen conserved bridge requires the shared combined source-recovery `READY_ALL` state and cannot execute independently without rewriting a frozen gate;
- WR/TE named-player opportunity-uncertainty science remains CONFIRMED, but historical exact specialist-distribution integration is source-blocked and may only proceed prospectively under a future pregame lock.

No gate is weakened and no blocked bridge is rescued.

The next distinct PRISM question returns to the core architecture:

> For this named WR/TE, does his own strictly-prior error-component history tell us whether the next receiving-yard miss is more likely to be **opportunity-driven**, **efficiency-driven**, or **mixed**?

This is a **mechanism-routing** question at named-player grain.

It does not ask whether the player is generally trustworthy, and it does not ask for another position-wide coefficient.

## 2. Player-first architecture

PRISM remains:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE -> FINITE OPPORTUNITY -> OPPORTUNITY QUALITY -> EFFICIENCY -> CONTEXT -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This V1 is a historical player-identity / mechanism-routing object.

It is intended to answer:

> Which already-existing projection component tends to be the active source of error for this player in this exact receiving-yard market?

If a player repeatedly exhibits opportunity-dominant misses, that is different football information from another player whose misses are predominantly efficiency-driven.

Position is only the prior/fallback.

The named player must earn deviation from that prior through strictly-prior support.

## 3. Exact source authority

Reuse the immutable historical component-decomposition source:

- run `39061606501` — SUCCESS;
- artifact `12111278777`;
- artifact name `prism-player-mechanism-instability-source-v1-39061606501`;
- digest `sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9`;
- live artifact status verified on 2026-10-18: expired=false.

Source rows already preserve exact identity:

`current_total_error = current_opportunity_error + current_efficiency_error`

with decomposition gap <= 1e-10.

Historical windows remain:
- WR: development 2023 / independent confirmation 2024;
- TE: development 2024 / independent confirmation 2025.

No 2026 outcomes.

## 4. Anti-retest boundary — CLOSED mechanism-instability trust

This V1 must not rescue:

`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_CLOSED`.

That closed test asked:

> Does prior switching/instability between component states predict larger future total projection error?

Primary predictor:
- mechanism switch rate.

Target:
- current absolute total receiving-yard error / projection fragility.

It failed dependence-aware confirmation.

Binding anti-rescue from that test remains:
- no entropy/Gini/majority-state **as a replacement trust predictor**;
- no total-error trust/selectivity rescue;
- no width or confidence penalty from component state.

This V1 uses the same historical decomposition only for a different target:

> predict the **next component state itself**, not total-error magnitude.

It may never be interpreted as a generic trust score.

## 5. Anti-retest boundary — signed component bias

This V1 must also remain distinct from:

`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_PARTIAL`.

That signal predicts the **signed mean error** of each component.

Qualified:
- WR opportunity;
- TE opportunity.

Closed:
- WR efficiency;
- TE efficiency.

This V1 does not predict signed yards of error.

It predicts only which component class dominates in magnitude.

A future routing result may not alter a component mean unless a separate bridge is frozen and source-authorized.

## 6. Frozen component-state definition

Use the already-frozen dominance convention unchanged.

For each target historical row:

- `O = abs(current_opportunity_error)`
- `E = abs(current_efficiency_error)`

Class:
- `OPPORTUNITY_DOMINANT` if `O >= 1.5 * E`;
- `EFFICIENCY_DOMINANT` if `E >= 1.5 * O`;
- otherwise `MIXED`.

The 1.5 ratio is inherited to avoid a new threshold search.

Do not change it after results.

## 7. Strictly-prior player state

For every target row:

- same position;
- same named player;
- same season;
- rows strictly before target week;
- latest up to 8 prior component rows;
- minimum 4 prior rows.

From prior rows compute descriptive counts only:
- opportunity-dominant count/share;
- efficiency-dominant count/share;
- mixed count/share;
- prior state count.

Target row state is excluded from its own history.

No target-week component error enters the predictor.

## 8. Frozen hierarchy

Baseline:

`target-player-excluded position prior state distribution`

using:
- same position;
- same season;
- strictly earlier weeks only;
- exclude the target player's rows;
- exclude same-week outcomes.

Candidate:

`named-player hierarchical state distribution`

with exactly four position-prior pseudo-observations:

`p_candidate(class) = (n * p_player(class) + 4 * p_position(class)) / (n + 4)`

where:
- `n` = exact prior player component-row count, 4-8;
- `p_player(class)` = player's strictly-prior empirical class distribution.

No coefficient is fit.

No class is privileged in advance.

## 9. Frozen target and scoring

Target:
- next-row component class:
  - OPPORTUNITY_DOMINANT;
  - EFFICIENCY_DOMINANT;
  - MIXED.

Primary score:
- multiclass Brier score.

Secondary:
- multiclass log loss;
- probability assigned to realized class;
- top-class accuracy;
- confusion matrix;
- per-player Brier change.

Primary improvement is:

`baseline_Brier - candidate_Brier`

positive = player history helps.

Accuracy is diagnostic only and cannot rescue a Brier failure.

## 10. Frozen chronology

WR:
- development = 2023;
- independent confirmation = 2024.

TE:
- development = 2024;
- independent confirmation = 2025.

WR and TE adjudicated separately.

No pooled-position rescue.

## 11. Frozen qualification gates — per position

A position qualifies only if ALL pass:

Development:
1. candidate multiclass Brier < baseline;
2. candidate multiclass log loss <= baseline;
3. equal-player mean Brier improvement > 0.

Independent confirmation:
4. candidate multiclass Brier < baseline;
5. candidate multiclass log loss <= baseline;
6. player-cluster 95% CI lower bound on Brier improvement > 0;
7. team-week-cluster 95% CI lower bound > 0;
8. equal-player 95% CI lower bound > 0;
9. median named-player Brier improvement >= 0;
10. named-player improved share > worsened share;
11. p10 named-player improvement not worse than the frozen tail-harm floor.

Tail-harm floor:
- fixed before scoring as `-0.010000` multiclass-Brier units.

Integrity:
12. chronology violations = 0;
13. decomposition identity max gap <=1e-10;
14. 2026 outcomes used = 0;
15. sportsbook/OddsAPI = 0;
16. coefficients fit = 0;
17. thresholds searched = 0.

Bootstrap:
- 5,000 replicates;
- player seed = 20261018;
- team-week seed = 20261019.

## 12. What qualification would mean

A pass would establish only:

> this named player's strictly-prior component fingerprint contains incremental information about which component is likely to dominate his next receiving-yard projection error.

That would authorize a **separate frozen routing/action study**.

Possible future routing questions could include whether:
- opportunity-side football state should receive more diagnostic authority for that player;
- signed-opportunity correction eligibility should require opportunity-dominance support;
- efficiency-side downstream research should be deprioritized for players whose error is persistently opportunity-driven.

None of those actions is authorized by this signal itself.

## 13. What failure means

If WR or TE fails, close this exact component-dominance persistence/routing family for that position.

Do not rescue with:
- another dominance ratio;
- entropy/Gini;
- recent-4 vs prior-8 variants;
- WR1/WR2+/TE1 subsets;
- position pooling;
- sign-conditioned classes;
- signed-bias magnitude thresholds;
- high-error-only cohorts;
- sportsbook information.

## 14. Position boundary

V1:
- WR receiving yards;
- TE receiving yards.

RB:
- source not certified for this exact decomposition panel;
- do not proxy from a different RB mechanism.

QB:
- excluded from this V1 by prior QB reliability/anti-retest lineage.

## 15. Integrity at freeze

- predictive component-routing rows scored: **0**;
- projection means changed: **0**;
- distributions changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6/Week-7 outcomes used: **0**;
- sportsbook predictors: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_FROZEN`.
