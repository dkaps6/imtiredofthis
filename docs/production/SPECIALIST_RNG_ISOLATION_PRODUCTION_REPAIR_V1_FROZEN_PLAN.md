# Specialist RNG Isolation Production Repair Candidate V1 — Frozen Plan

**STATUS: FROZEN BEFORE PRODUCTION-CANDIDATE SCORING. NO MERGE AUTHORIZED.**

## Authority

Canonical production main at branch creation:
- `8a965f2754b4ccfa960a2a05517fd40861f49a3d`

Paid Week-3 historical authority:
- run `36293274478`
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- paid head `0982b62276303403e2ca58b16e6f4fc3e041f65d`

Research authorization chain:
- Specialist MC materiality: `SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE`
- Specialist RNG core: `SPECIALIST_RNG_ISOLATION_CORE_PASS`
- QB C2 extension: `QB_C2_RNG_ISOLATION_DOWNSTREAM_PASS`
- Paid-board counterfactual: `RNG_ISOLATION_COUNTERFACTUAL_WITHIN_ORDINARY_MC_ENVELOPE`
- Issue #535 counterfactual checkpoint: `5871358872`

This plan authorizes a **production-code candidate for testing only**. It does not authorize merge, historical-board rewrite, outcome scoring, or new sportsbook acquisition.

## Repair scope

The defect is random-number-path coupling, not football science.

The candidate may change only:
1. how random streams are routed;
2. how already-conserved TE-R5P and WR-R15 target rooms are sampled hierarchically;
3. how QB C2 uses the same semantic random-stream routing;
4. audit/report plumbing required to prove those contracts.

The candidate may not change:
- any football feature;
- any coefficient;
- any Bayesian weight;
- M38 entitlement values;
- TE-R5P values;
- WR-R15 values;
- M89/M90 QB mean authority;
- QB C2 selector logic;
- RB Rush+Receiving Conservation V2;
- Discrete Count Mean Alignment V1;
- sportsbook lines/odds;
- betting thresholds;
- selection rules.

## Production architecture

Add a new semantic-RNG simulator module rather than modifying `simulation_v2.py` in place.

### Base simulation

The production candidate consumes the already-materialized explicit entitlement columns.

Required specialist scope metadata:
- `te_r5p_applied`
- `wr_r15_applied`

Required baseline authority:
- `baseline_entitlement_tgt_share`

Required final stage:
- `entitlement_tgt_share`

The top-level target allocator collapses:
- all `te_r5p_applied == True` rows into `TE_R5P_ROOM`;
- all `wr_r15_applied == True` rows into `WR_R15_ROOM`;
- every other player into an individual category;
- residual target mass into a residual category.

Top-level category masses come from the M38 explicit baseline authority. Because TE-R5P conserves the TE room and WR-R15 conserves the WR2+ room, those top-level probabilities are invariant to specialist redistribution.

Room-internal allocation uses the current stage's `entitlement_tgt_share`.

### Semantic RNG streams

Stable digest-derived substreams are keyed by:
- base seed;
- semantic layer;
- event;
- team;
- optional player/room.

Shared football states remain shared:
- game pace;
- team volume/pass rate;
- pass efficiency;
- rush efficiency.

Independent semantic streams:
- top-level target allocation;
- specialist-room allocation;
- rushing allocation;
- catch process;
- receiving-yard noise;
- rushing-yard noise;
- QB pass noise;
- TD process.

This removes unrelated sequential RNG consumption while preserving intended football dependence.

## Baseline parity authority

Do **not** weaken the existing projection-neutral entitlement proof.

Before using the new candidate simulator:
1. reconstruct the legacy raw-input simulation;
2. reconstruct the current explicit-M38 baseline using the existing production explicit adapter;
3. retain the existing exact finite-sample equality gate between those two current-authority paths.

That proves the M38 explicit entitlement seam is still semantically identical to the legacy authority.

The new semantic-RNG simulator is then applied to:
- M38 baseline explicit entitlement;
- TE-R5P stage;
- final TE-R5P + WR-R15 stage.

The research result already established that its finite sample need not equal the old global-RNG sample; it must instead pass the candidate gates below.

## Candidate base-simulation gates

For M38 -> TE-R5P protected players:
- zero element drift across pass_yards, rush_att, rush_yards, receptions, rec_yards, rush_rec_yards;
- zero mean drift.

For TE-R5P -> WR-R15 protected players:
- same exact zero-drift gates.

Intentional specialist receiving arrays must still change.

For every event/team/iteration:
- modeled targets <= pass attempts;
- modeled carries <= rush attempts;
- no negative counts;
- no non-finite values.

## QB C2 production candidate

Do not alter the frozen selector.

The existing production adapter receives an optional C2 apply function. Default remains the current production C2 implementation so all unrelated callers remain backward compatible.

The RNG-isolation candidate supplies a semantic-stream C2 apply function that:
- reuses preserved pass attempts and pass-efficiency state;
- uses the same explicit specialist-room metadata;
- uses player-keyed catch/yards streams;
- uses team-keyed residual receiving streams;
- changes only primary-QB pass_yards arrays;
- remains mean-neutral to <= 1e-10;
- uses no sportsbook inputs.

## Full candidate integration

Add a new candidate Full Slate entry point rather than replacing the canonical production entry point.

The candidate entry point must:
- reuse the exact current full-roster builder and specialists;
- preserve the existing legacy-vs-explicit M38 exact parity proof;
- use semantic-RNG simulation for baseline/TE/final specialist stage audits;
- use semantic-RNG state capture;
- use the unchanged production QB C2 selector with the semantic C2 apply function;
- preserve all current production audits.

Canonical `.github/workflows/full-slate.yml` remains untouched during candidate validation.

## Research-to-production equivalence gate

Using the preserved Week-3 paid artifact, the production candidate must reproduce the frozen research findings:

### Core isolation
- M38 -> TE-R5P: 1,750 protected keys, 0 drift;
- TE-R5P -> WR-R15: 1,525 protected keys, 0 drift;
- intentional TE receiving arrays changed: 172 / 172;
- intentional WR receiving arrays changed: 262 / 262.

### C2
- exactly 30 primary-QB rows;
- exactly 30 changed simulation keys;
- all changed keys primary-QB pass_yards;
- max raw mean gap <= 1e-10;
- max non-selected element gap = 0.

### Paid-board counterfactual
Recompute candidate-vs-paid on the exact preserved supported board.

Frozen all-supported targets:

SHAPE_ONLY_FIXED_FINAL_MEAN:
- p99 fair-prob delta: `0.01616`
- max fair-prob delta: `0.03588`
- p99 EV delta: `0.031741616000000014`
- Best Snapshot BET/PASS flips: `6`
- Best Snapshot identity changes: `1`
- Best-EV Spearman: `0.9984915688405269`
- top-10 turnover: `0`
- top-25 turnover: `0`

FULL_DOWNSTREAM_PROPAGATION:
- p99 fair-prob delta: `0.010273600000000119`
- max fair-prob delta: `0.015880000000000005`
- p99 EV delta: `0.019283169750603693`
- Best Snapshot BET/PASS flips: `4`
- Best Snapshot identity changes: `1`
- Best-EV Spearman: `0.9989391731762268`
- top-10 turnover: `0`
- top-25 turnover: `0`

Continuous metrics must match within `1e-12`; integer decision counts must match exactly.

If the production port cannot reproduce these frozen shadow results, fail closed.

## Regression gates

Candidate workflow must run:
- focused semantic-RNG tests;
- focused C2 candidate tests;
- the existing simulator/entitlement/C2 tests directly affected by the port;
- repository CI if/when a PR is opened.

No paid/live sportsbook workflow may be triggered for candidate validation.

## Frozen dispositions

### `RNG_ISOLATION_PRODUCTION_CANDIDATE_PASS`
if all semantic, conservation, C2, research-equivalence, and focused regression gates pass.

This means the repair is technically merge-candidate quality. It still does not merge automatically.

### `RNG_ISOLATION_PRODUCTION_CANDIDATE_FAIL`
if any gate fails.

No rescue retuning. Diagnose the exact mechanical contract and either repair the implementation without changing the frozen design or fail closed.

## After candidate PASS

Before merge consideration:
1. inspect full diff;
2. run Repo CI;
3. ensure canonical Full Slate workflow remains unchanged until explicit promotion;
4. document the historical Week-3 paid board as immutable;
5. document that any repaired replay is counterfactual only;
6. only then decide whether to promote the candidate code.

Week-3 realized outcomes remain outside this mechanical repair decision.
