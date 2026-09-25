# Hierarchical Receiver Reconciliation V1 — Read-Only Diagnostic Plan

Date: 2026-09-24

Status: **FROZEN BEFORE NEW OUTCOME SCORING**

Parent evidence:
- `SHARED_PASS_STATE_COHERENCE_V1_STRUCTURAL_DIVERGENCE_CONFIRMED`
- `ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED`

This phase is read-only and uses **no target-game outcomes**. It does not authorize
production changes or historical scoring.

## 1. What One-Pass V1 taught us

One-Pass V1 proved two things simultaneously:

1. reconciling the QB/team passing state with receiver projections has real mean
   signal;
2. uniformly replacing every named receiver with the C2 completed-pass marginal
   is too blunt.

Frozen full-stack evidence:
- pooled WR/TE/RB receiving-yard macro MAE improved
  `16.234770 -> 16.171014`;
- both 2024 and 2025 macro MAE improved;
- WR, TE and RB pooled MAE each improved;
- RB rush+receiving MAE improved in both seasons;
- but high-entitlement Q4 receiver MAE/p90 worsened;
- pooled receiving-yard p90 worsened;
- receptions mean MAE was essentially flat/slightly worse;
- RB combo p90 worsened.

This closes wholesale receiver replacement. It does not close forecast
reconciliation as a class.

## 2. New question

Can the aggregate QB passing forecast and named receiver forecasts be made coherent
while **protecting stronger player-level marginal forecasts**?

This is a hierarchical forecast-reconciliation problem, not a player carveout.

The hierarchy is:

`QB/team passing yards = sum(named receiver receiving yards) + residual receiving yards`

The residual bucket already exists in the C2 football model and represents
unmodeled receiving mass.

## 3. Frozen Phase-A diagnostic

Use the current 2026 Week-3 football-only stack and the exact current C2-selected
team set.

No sportsbook line/odds and no Week-3 outcomes are allowed.

For each selected team:

1. record current C2 QB mean `Q`;
2. record current canonical MC receiving-yard mean for every named WR/TE/RB/FB;
3. let `S = sum(named receiver means)`;
4. record the existing C2 residual receiving-yard mean;
5. compute the coherence gap `G = Q - S`.

### Residual-first feasibility

Before moving any named player:

- if `G >= 0`, define an implied reconciled residual `R* = G` and leave all
  named receiver means unchanged;
- if `G < 0`, set residual to zero and the named receiver system must absorb
  reduction `-G`.

Report:
- fraction of selected teams requiring **no named-player movement**;
- implied residual share `R*/Q` distribution;
- fraction with implied residual share > 5%, >10%, >15%, >20%;
- teams where named receiver means alone exceed Q;
- required named-system reduction as yards and percentage of S.

This is diagnostic only. No residual-share threshold is being chosen.

## 4. Existing uncertainty authority

For teams where `S > Q`, compute a research-only weighted projection using
only existing Bayesian pregame uncertainty fields:

- `bayes_tgt_share_sd`
- `bayes_ypt_sd`
- current explicit `entitlement_tgt_share`
- current `bayes_ypt`
- current team pass-attempt mean

For player i, define first-order receiving-yard epistemic variance:

`V_i = (A * YPT_i * SD_share_i)^2 + (A * Share_i * SD_ypt_i)^2`

where:
- `A` = current team mean pass attempts;
- `Share_i` = current explicit target entitlement;
- `YPT_i` = current Bayesian YPT mean.

This introduces **no fitted coefficient**. It is ordinary first-order uncertainty
propagation in receiving-yard units.

For required downward reconciliation, solve:

minimize `sum((x_i - b_i)^2 / V_i)`

subject to:
- `sum(x_i) = Q`;
- `x_i >= 0`.

Use an exact active-set solution:
- unconstrained movement is proportional to `V_i`;
- any negative solution is clamped to zero;
- remaining reduction is reallocated among still-active players by the same
  frozen variance rule.

No player name, position, depth label, entitlement quartile or result-derived
threshold enters the optimization.

## 5. Diagnostic outputs

Report current Week-3 selected teams and named receiver rows:

### Team level
- Q, S, G;
- canonical named-receiver sum / QB ratio;
- existing C2 residual mean/share;
- implied residual-first mean/share;
- residual-only feasible flag;
- required named reduction yards / percentage;
- weighted-reconciled sum identity gap.

### Player level
- canonical mean;
- weighted-reconciled mean;
- adjustment yards and percentage;
- entitlement share;
- Bayesian target-share SD/effective-N;
- Bayesian YPT SD/effective-N;
- propagated `V_i`;
- evidence state;
- position family.

### Summaries
By position and entitlement quartile:
- median absolute adjustment yards;
- median absolute percentage adjustment;
- median propagated variance;
- median effective-N.

Also report Spearman relationships between adjustment magnitude and:
- entitlement;
- target-share effective-N;
- YPT effective-N;
- propagated variance.

## 6. Integrity gates

The diagnostic stops unless all pass:

1. exact current C2 selected-team identity is preserved;
2. QB arrays/means are not modified;
3. canonical receiver arrays are not modified;
4. target entitlement is not modified;
5. rushing arrays are not modified;
6. all required Bayesian uncertainty fields are finite for every reconciled
   positive-entitlement named receiver;
7. weighted solution is nonnegative;
8. weighted reconciled named sum equals Q within `1e-10` for teams requiring
   named reduction;
9. sportsbook inputs = 0;
10. target-game outcomes = 0;
11. production changed = false.

## 7. Interpretation rule

This phase does **not** qualify a candidate.

It answers:
- whether the residual bucket can absorb most aggregate incoherence without
  touching named players;
- whether existing Bayesian uncertainty naturally protects higher-authority
  receiver marginals;
- whether a mathematically coherent weighted-reconciliation candidate is worth
  freezing.

No result-derived threshold, Q4 exemption, WR/RB carveout, player exception,
sportsbook routing, or One-Pass V1 rescue is allowed.

## 8. If the diagnostic is structurally sane

Freeze a separate historical scoring candidate **before outcomes are consulted**.

That candidate should initially modify receiving-yard means only:
- receptions stay current production;
- rushing stays current production;
- RB rush+receiving is rebuilt through current RB V2;
- no player/position exemptions;
- no threshold search.

Historical evidence must be described as retrospective because 2024-2025 outcomes
are already known elsewhere in the project. Any historical qualification must
still be followed by prospective 2026 capture before production promotion.
