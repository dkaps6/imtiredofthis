# TE LIVE ENTITLEMENT VS EFFICIENCY V1 — FROZEN PLAN

**Status:** FROZEN BEFORE IMPLEMENTATION  
**Branch:** `research-te-live-entitlement-efficiency-v1`  
**Parent main:** `69e9c6211710c60de9aaf07d06aca6cafeff5ccf`  
**Production changes authorized by this study:** NONE

## Why this exists

PR #626 is closed. The canonical 2026 Week-1/Week-2 production record is now
stable enough to diagnose football-model error rather than scoreboard error.

The clearest live weakness is TE:

- overall: 58-75 / -21.45u;
- receiving yards: 28-41 / -16.09u;
- receptions: 30-34 / -5.37u;
- Week 2 TE: 26-38.

Historical TE-R1 already established a valid mechanism decomposition over
6,371 scoreable 2020-2025 rows:

- TARGETS: 45.2% of absolute error mass;
- YPR: 29.5%;
- CATCH_RATE: 25.3%.

Current-Season State Persistence V1 independently established that TE target
share is one of the strongest early-season state variables, while early
efficiency should remain more heavily shrunk.

PR #627 then prospectively enabled strict-prior 2026 snap participation in
TE-R5P beginning Week 3 without refitting coefficients.

The immediate football question is therefore:

> Are the live 2026 TE receiving-yard misses primarily an entitlement/opportunity
> miss, an efficiency miss, or a downstream distribution/ensemble miss — and
> does the newly available current-season snap state move TE-R5P entitlement in
> the correct direction?

## Immutable inputs

1. Canonical selected/settled W1/W2 ledger merged by PR #626:
   `data/market_track_record/graded/2026_wk01_wk02_graded.csv`.
2. Exact paid Week-2 Full Slate origin artifact:
   - source run `35282021679`
   - source SHA `c6ec55be70d6e05bbd1dbae83d7d5c86ac8aa00a`
   - artifact `10523345092` / `run_35282021679`
   - artifact digest `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`
3. nflverse/nflreadpy final 2026 Week-2 player stats and snap counts.
4. Frozen production TE-R5P model:
   `TE_R5P_PRODUCTION_MODEL_V1`.
5. Historical TE-R1 authority:
   run `34123413402`, artifact `10019112418`,
   disposition `TE_MECHANISM_DECOMPOSITION_ACTIONABLE`.
6. Current-season state authority:
   run `35741758765`, artifact `10699781744`.
7. PR #627 source continuation authority beginning 2026 Week 3.

No new sportsbook request is allowed. Existing archived lines may identify the
live production-selected cohort but are not football-model inputs.

## Cohorts

### A. Week-2 all-TE football cohort
Every Week-2 TE row in the exact origin football universe that can be resolved
to final nflverse targets/receptions/receiving yards.

Purpose: diagnose football projection mechanics without conditioning on whether
a sportsbook bet happened to be selected.

### B. Week-2 canonical TE receiving-yard betting cohort
Canonical PR-626 Week-2 TE `rec_yards` settled rows, joined to the exact origin
trace and final nflverse outcomes.

Purpose: explain the specific live betting failure.

### C. Week-1/Week-2 scoreboard context
Use the canonical ledger only to report W1/W2 TE projection error and W/L.
Do not invent a Week-1 entitlement trace: the original full Week-1 artifact is
expired. Week-1 mechanism attribution is therefore explicitly out of scope
unless independently immutable exact pregame evidence is recovered.

## Diagnostic 1 — live entitlement vs efficiency

For each Week-2 TE with exact production entitlement state:

- production target share = exact TE-R5P entitlement target share;
- actual target share = actual targets / actual team targets;
- actual TE-room share = actual TE targets / actual team TE targets;
- production TE-room share = exact TE-R5P room share;
- production target-count proxy = actual team targets × production target share.

Using the final football receiving-yard projection, define an explicit
**implied football YPT**:

`implied_ypt = model_proj / production_target_count_proxy`

where the denominator is positive. This is a diagnostic factorization of the
final projection, not a claim that YPT is the only downstream model parameter.

Counterfactuals:

- perfect entitlement, frozen implied efficiency:
  `actual_targets × implied_ypt`;
- frozen entitlement, perfect realized efficiency:
  `production_target_count_proxy × actual_ypt`.

Report absolute-error recovery for each. This produces a direct live
opportunity-vs-efficiency attribution that is comparable in interpretation,
but not numerically identical, to historical TE-R1.

Also report MC-vs-final movement (`model_proj - mc_proj`) so a downstream
ensemble overlay that worsens an otherwise reasonable simulation is visible.

## Diagnostic 2 — Week-2 current-snap counterfactual for TE-R5P

Production Week 2 intentionally used the legacy 2020-2025 snap-source contract.
Research-only counterfactual:

- keep the exact Week-2 pregame entitlement frame and frozen TE-R5P coefficients;
- add 2026 Week-1 snap counts, which were available strictly before Week 2;
- rebuild only TE-R5P individual room allocation;
- preserve team TE target-pool mass exactly;
- compare candidate vs production on:
  - all-TE target-share MAE;
  - all-TE room-share MAE;
  - canonical selected TE receiving-yard cohort;
  - top absolute target-share-error quartile.

This is **not** permission to rewrite Week-2 production history. Its value is
prospective: Week 3 already has the equivalent strict-prior current-season snap
source enabled by PR #627.

## Frozen interpretation gates

This study is diagnostic first. No production promotion occurs automatically.

Current-snap entitlement signal is considered prospectively supportive only if:

1. all-TE target-share MAE improves by at least 2% versus exact Week-2 production;
2. all-TE TE-room-share MAE does not worsen;
3. selected TE receiving-yard cohort target-share MAE does not worsen by more than 1%;
4. no team TE-pool conservation violation exceeds 1e-12;
5. zero sportsbook fields enter the candidate football calculation.

If these do not clear, Week-3 snap continuation remains source-correct but is
not claimed as a live accuracy improvement from this test.

## Decision tree after results

- **Entitlement dominates + current snaps help:** Week-3 source continuation is
  directly supported; next work focuses on remaining efficiency/distribution.
- **Entitlement dominates + current snaps do not help:** investigate other
  pregame role information (routes/personnel/TE-room vacancy), not coefficient
  retuning against W1/W2 outcomes.
- **Efficiency dominates:** move immediately to catch-rate/YPR/distribution
  diagnostics using historically justified shrinkage; do not distort target
  entitlement to solve an efficiency problem.
- **Final ensemble movement is the main damage:** isolate TE downstream
  ensemble/distribution authority before inventing new football features.

## Anti-overfit / stopping rules

- No coefficient search against 2026 W1/W2 outcomes.
- No line-informed football feature.
- No global SD multiplier from two live weeks.
- No reopening closed TE/RB/WR historical families merely because live results
  are poor.
- One frozen current-snap counterfactual only.
- Every result and disposition is written to GitHub and Issue #535.
