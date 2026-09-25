# Hierarchical Receiver Reconciliation V1 — Read-Only Diagnostic Result

Date: 2026-09-24

Disposition: **STRUCTURALLY SANE — HISTORICAL CANDIDATE JUSTIFIED**

This result is diagnostic only. It uses no target-game outcomes and does not
authorize production.

## Authority

- branch: `research-hierarchical-receiver-reconciliation-v1`
- authoritative run: `36080565164`
- job: `107901395166`
- head: `11cc88c1c19bb7192bdbfc445f60ff2a077bf958`
- artifact: `10842285131`
- digest: `sha256:2f3a971f2c2923a453a8aed5e9431a1584716844010eb5fc5173833b7bf06d93`
- season/week: 2026 Week 3
- football teams in certified current universe: **30**
- canonical games: **15**
- selected C2 teams: **30**
- receiver rows: **356**
- sportsbook inputs: **0**
- target-game outcomes: **0**
- production changed: **false**
- weighted reconciliation max identity gap:
  `5.684341886080802e-14`

## Mechanical correction before result

The first run `36080048529` stopped before diagnostics because an older
research helper hard-coded a 32-team current universe. Current availability had
certified 30 active teams.

The audit was corrected to use the exact production
`validate_current_team_set` boundary. No missing team/player was imputed and
no reconciliation metric existed before the correction.

## Team-level finding

The current named receiver means do not naturally reconcile to the C2 QB/team
mean on most selected teams.

- residual-only feasible: **9 / 30 teams (30%)**
- named-player reduction required: **21 / 30 teams (70%)**
- median absolute QB-minus-named gap: **11.49 yd**
- median required named-system reduction among required teams:
  **4.95%**
- worst required team reductions were about **9-11%**

The existing C2 residual share is usually around five percent.

When the named receiver system already exceeds the QB/team mean, a residual
bucket cannot solve the inconsistency by itself; named receiver means must move
if exact mean coherence is imposed.

## Weighted-reconciliation behavior

The frozen diagnostic used existing Bayesian posterior uncertainty only:

`V_i = (A * YPT_i * SD_share_i)^2 + (A * Share_i * SD_ypt_i)^2`

and solved the nonnegative weighted least-squares projection exactly.

No fitted coefficient, player label, position carveout, result-derived threshold
or sportsbook input was used.

Across all current selected-team receiver rows:

- median absolute adjustment: **0.344 yd**
- p75 absolute adjustment: **1.222 yd**
- p90 absolute adjustment: **2.375 yd**
- p95 absolute adjustment: **3.116 yd**
- p99 absolute adjustment: **5.712 yd**
- maximum absolute adjustment: **6.759 yd**

This is materially gentler than the wholesale C2 receiver replacement that
failed in One-Pass V1.

## Entitlement / authority behavior

All rows, median absolute adjustment:

- Q1 low: **0.142 yd / 2.11%**
- Q2: **0.368 yd / 3.08%**
- Q3: **0.452 yd / 2.38%**
- Q4 high: **0.533 yd / 1.59%**

Restricting to the 21 teams where named reduction is actually required:

- Q1 low: **0.382 yd / 5.53%**
- Q2: **0.806 yd / 6.52%**
- Q3: **0.730 yd / 3.95%**
- Q4 high: **1.266 yd / 3.08%**

So higher-entitlement players can move more in absolute yards because they own
larger forecasts, but the reconciliation naturally moves them substantially less
as a percentage of their base projection than low/mid-entitlement players.

The current Q4 group also carries stronger historical evidence:
- median target-share effective-N: **11**
- median YPT effective-N: **15**

versus Q1:
- target-share effective-N: **9**
- YPT effective-N: **11**

## Individual diagnostic pattern

The largest percentage reductions are concentrated mainly in low-evidence /
fallback receivers.

Examples among the largest raw adjustments included:
- CJ Williams: ~`-6.76 yd`, `position_prior_only`
- Zavion Thomas: ~`-6.35 yd`, `position_prior_only`
- Kaden Wetjen: ~`-5.59 yd`, `position_prior_only`

Established high-entitlement receivers generally moved less proportionally.

Examples:
- Justin Jefferson: about `-2.62 yd / -5.1%`
- Amon-Ra St. Brown: about `-1.71 yd / -2.6%`
- Puka Nacua: about `-1.67 yd / -2.3%`
- Nico Collins: about `-2.50 yd / -3.7%`
- DK Metcalf: about `-1.96 yd / -4.6%`

These examples are current pregame diagnostics only and are not scored against
Week-3 outcomes.

## Correlation diagnostics

Spearman correlation between absolute adjustment and:

- entitlement share: **0.142**
- propagated variance: **0.359**
- target-share effective-N: **-0.009**
- YPT effective-N: **0.005**

The posterior-variance objective therefore does not mechanically act as an
effective-N ranking. It acts in receiving-yard uncertainty units, which is the
intended geometry.

## Interpretation

The diagnostic supports a separate historical candidate because:

1. the aggregate/component incoherence is real;
2. residual-only handling protects named players on 30% of selected teams;
3. where named movement is mathematically required, the weighted solution is
   nonnegative and exact;
4. adjustments are much smaller than One-Pass V1;
5. high-entitlement receiver movement is materially smaller in percentage terms;
6. no result-derived carveout is needed.

This does **not** prove predictive improvement.

The next candidate must be frozen before historical scoring and should test
**mean-only receiving-yard reconciliation**:
- keep receptions unchanged;
- keep QB distributions/means unchanged;
- keep rushing unchanged;
- adjust only receiving-yard means;
- rebuild dependent RB rush+receiving through current RB V2;
- preserve the residual-first rule;
- use the same uncertainty geometry with no fitted coefficients.

Historical qualification, if any, is retrospective evidence only and still
requires prospective 2026 confirmation before production promotion.
