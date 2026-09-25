# Hierarchical Receiver Mean Reconciliation V1 — Frozen Historical Candidate Plan

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

Parent read-only diagnostic:
- `HIERARCHICAL_RECEIVER_RECONCILIATION_V1_AUDIT_RESULT.md`
- run `36080565164`
- artifact `10842285131`
- no Week-3 outcomes used

This is a new candidate. It is not a rescue of ONE_PASS_STATE_INTEGRATION_V1.

## 1. Production-order correction

The read-only current-slate audit established that the residual-first +
uncertainty-weighted reconciliation geometry is mechanically sane, but it used
the raw C2/canonical QB mean as the aggregate diagnostic anchor.

Production's final QB passing-yards **mean** authority is downstream:

1. canonical MC/ML/State;
2. evidence-weighted ensemble;
3. M89/M90 `QB_PASS_SYNTHESIS_V1`;
4. QB MC distribution rescaled to the M89/M90 synthesis mean.

Therefore historical scientific scoring must reconcile receiver means to the
exact OOS M89/M90 `football_synthesis` authority, not the intermediate C2 mean.

The C2 selector is a mean-neutral **distribution** selector and is not used to
route this mean-only candidate.

## 2. Historical QB authority

Use the exact preserved M89 OOS authority:

- source run: `34122984048`
- artifact: `10018942911`
- artifact name: `qb-pd3-internal-disagreement-reliability`
- digest: `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`
- casebook rows: **884**
- 2024: **444**
- 2025: **440**

Only `football_synthesis` is consumed by this candidate. Postgame columns in
that artifact are prohibited from candidate construction.

The M89 authority row must be joined by:
- season
- week
- team
- opponent
- primary-QB identity where available

No authority row may be invented.

## 3. Receiver base authority

For each historical week, rebuild the current receiver stack using the same
leakage-safe machinery as the latest full-stack experiments:

- historical football context;
- Bayesian baseline;
- empirical football rules;
- explicit team target entitlement;
- fold-safe TE-R5P authority;
- fold-safe WR-R15 authority where historically authorized;
- canonical MC receiver distribution;
- ML;
- state;
- current frozen ensemble weights.

For each named WR/TE/RB/FB receiving-yard row, define:

`b_i = final generic rec_yards ensemble mean`

This is the base named-player mean. No sportsbook field is used.

## 4. Exact candidate mechanism

Candidate:
`HIERARCHICAL_RECEIVER_MEAN_RECONCILIATION_V1`

For each M89-authorized team-game:

- `Q` = M89 `football_synthesis`;
- `S = sum(b_i)` over named WR/TE/RB/FB receiving-yard projections.

### Case A — residual-only feasible

If `S <= Q`:

- every named receiver mean remains exactly `b_i`;
- implied residual receiving mean is `Q - S`;
- no named player is adjusted.

### Case B — named system exceeds QB aggregate

If `S > Q`:

- residual receiving mean = 0;
- named receiver means are projected downward to sum exactly to `Q`.

For each named player:

`V_i = (A * YPT_i * SD_share_i)^2 + (A * Share_i * SD_ypt_i)^2`

where:
- `A` = strict-prior/current-stack team mean pass attempts;
- `Share_i` = explicit target entitlement;
- `YPT_i` = Bayesian YPT;
- `SD_share_i` = Bayesian target-share posterior SD;
- `SD_ypt_i` = Bayesian YPT posterior SD.

Solve exactly:

minimize

`sum((x_i - b_i)^2 / V_i)`

subject to:

- `sum(x_i) = Q`
- `x_i >= 0`

using the frozen active-set weighted projection from the read-only diagnostic.

No fitted coefficient is introduced.

## 5. What changes

Only the final football **mean** of `rec_yards` may change.

For any adjusted receiver row:
- keep the current MC distribution shape;
- rescale that rec_yards draw array to the reconciled mean;
- do not alter receptions;
- do not alter target entitlement;
- do not alter catch rate;
- do not alter YPT inputs;
- do not alter QB means or distributions;
- do not alter rush attempts;
- do not alter rush yards;
- do not alter ATD.

For RB `rush_rec_yards`:
- Week 1 remains current baseline because RB Rush+Receiving Conservation V2 is
  non-Week-1;
- Weeks 2-18 rebuild the final combo target exactly as current RB V2:
  unchanged final rush-yards target + reconciled final rec-yards target.

## 6. Explicitly forbidden

No:
- player-name rules;
- WR/TE/RB carveouts;
- Q4 exemption;
- entitlement threshold;
- effective-N threshold;
- variance exponent;
- variance cap/floor search;
- residual-share threshold;
- C2 selector threshold;
- M89 retune;
- WR-R15/TE-R5P retune;
- sportsbook input;
- target-game outcome in construction;
- candidate variants.

Parameters fit: **0**

Candidate variants scored: **1**

## 7. Historical evaluation

Evaluate 2024 and 2025 separately and pooled.

The experiment is retrospective because those outcomes are already known
elsewhere in the project. No claim of pristine prospective discovery is allowed.

Score primary positions:
- WR
- TE
- RB

Primary market:
- receiving yards

Protected dependent market:
- RB rush+receiving yards

Receptions are an exact mechanical no-op and are not a candidate score target.

## 8. Required mechanical / provenance gates

All must pass:

1. exact M89 authority rows: 444 in 2024 and 440 in 2025;
2. no duplicate M89 team-game authority;
3. every consumed M89 row maps to one historical football team-game;
4. candidate consumes only `football_synthesis` from M89 authority;
5. QB projection means unchanged;
6. QB distribution arrays unchanged;
7. receptions arrays unchanged;
8. rush-att arrays unchanged;
9. rush-yard arrays unchanged;
10. target entitlement unchanged;
11. unadjusted receiver rows bit-identical;
12. reconciled receiver means finite and nonnegative;
13. for every M89-authorized team:
    `sum(named reconciled means) + residual = Q` within `1e-10`;
14. residual mean nonnegative;
15. RB V2 combo identity preserved after using the candidate rec component;
16. sportsbook inputs = 0;
17. target-game outcomes used upstream = 0;
18. parameters fit = 0;
19. candidate variants scored = 1.

Any failure above is mechanical/provenance failure.

## 9. Frozen scientific scorecard

For receiving yards, report by:
- season;
- pooled;
- WR / TE / RB;
- entitlement quartile.

Metrics:
- n;
- MAE;
- RMSE;
- bias;
- absolute bias;
- correlation;
- median AE;
- p75 AE;
- p90 AE;
- 20+ / 30+ / 40+ yard miss rates;
- changed rows;
- candidate closer / baseline closer / tie;
- candidate closer rate among decided rows.

Also report:
- adjustment yards / percentage distribution;
- residual-only feasible rate;
- implied residual share distribution;
- named-reduction-required rate;
- named reduction percentage distribution.

For RB rush+receiving:
- MAE;
- RMSE;
- bias;
- median AE;
- p90 AE;
- 30+ / 40+ miss rates.

## 10. Frozen scientific gates

`HIERARCHICAL_RECEIVER_MEAN_RECONCILIATION_V1_QUALIFIED` requires all:

### Receiving-yards mean accuracy
1. pooled WR/TE/RB macro MAE strictly improves;
2. 2024 macro MAE is nonworse;
3. 2025 macro MAE is nonworse;
4. pooled WR MAE is nonworse;
5. pooled TE MAE is nonworse;
6. pooled RB MAE is nonworse.

### Tail protection
7. pooled macro p90 AE is nonworse;
8. 2024 macro p90 AE is nonworse;
9. 2025 macro p90 AE is nonworse;
10. pooled macro 40+ yard miss rate is nonworse.

### High-authority protection
11. Q4 receiving-yard MAE is nonworse;
12. Q4 receiving-yard p90 AE is nonworse;
13. Q4 40+ yard miss rate is nonworse.

### Bias protection
14. pooled macro absolute bias is nonworse.

### RB dependent market
15. RB rush+receiving MAE is nonworse in 2024;
16. RB rush+receiving MAE is nonworse in 2025;
17. pooled RB rush+receiving p90 AE is nonworse.

### Architecture / integrity
18. every mechanical/provenance gate passes.

If any frozen gate fails:

`HIERARCHICAL_RECEIVER_MEAN_RECONCILIATION_V1_FAILED_CLOSED`

## 11. No-rescue rule

After results are visible, do not try:
- WR-only reconciliation;
- RB-only reconciliation;
- TE-only reconciliation;
- Q4 protection override;
- player-name exceptions;
- top-N receiver rules;
- effective-N cutoffs;
- alternative uncertainty exponents;
- variance winsorization;
- alternate residual caps;
- partial reconciliation fractions;
- QB/receiver blend weights;
- C2-routed variants;
- sportsbook-conditioned routing.

Any such idea requires a separately justified future hypothesis and new
validation design.

## 12. Promotion rule

Historical qualification does not authorize production.

If qualified:
1. freeze a prospective 2026 shadow contract;
2. compute candidate projections pregame with the final deployed M89 mean;
3. preserve those projections before outcomes;
4. score prospectively;
5. only then consider a separate production integration certification.

No paid OddsAPI pull is authorized.
