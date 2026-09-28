# QB C2 RNG Isolation Extension V1 — Frozen Shadow Plan

**STATUS: FROZEN BEFORE SCORING. RESEARCH-ONLY. NO PRODUCTION MUTATION.**

## Authority

Antecedent:
- Specialist MC downstream materiality result: `SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE`
- Specialist RNG Isolation V1 core result: `SPECIALIST_RNG_ISOLATION_CORE_PASS`
- core run: `36427943860`
- core artifact: `10971812982`
- core artifact digest: `sha256:b17c6317f3c5815f48955fb2d8bbee9915d97eef7cb8162d16c36ac3ab52eb8d`
- core head: `81825a33c853aefad992c9884205df8242aecec2`
- Issue #535 core checkpoint: `5870771354`

Paid Week-3 source remains:
- run `36293274478`
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`

No Week-3 outcomes. No OddsAPI. No production mutation.

## Why this extension exists

The core shadow isolated the canonical pre-C2 joint simulator exactly:
- 1,750 TE-protected keys: 0 drift;
- 1,525 WR-protected keys: 0 drift;
- intentional TE/WR specialist arrays still changed.

The current QB C2 candidate has its own sequential RNG path:
- target allocation;
- receiver catches;
- receiver yardage noise;
- residual receiving process.

Therefore a production repair cannot be evaluated honestly until C2 is routed through the same semantic-stream discipline.

## Frozen C2 design

Use the core isolated team's preserved shared states:
- pass attempts;
- pass-efficiency shock.

Within C2:
1. restrict the target allocator to pass-catcher rows, matching the current C2 semantic;
2. use the same frozen mutable specialist rooms from Specialist RNG Isolation V1;
3. allocate top-level target counts with a deterministic keyed stream;
4. allocate each mutable room internally with its own keyed stream;
5. use player-keyed C2 catch streams;
6. use player-keyed C2 receiving-yard streams;
7. use a team-keyed residual catch/yards stream;
8. preserve the current C2 YPR construction, volatility rule and residual assumptions;
9. preserve mean neutrality by scaling the C2 team receiving-yard total to the exact canonical raw QB passing-yard anchor.

No C2 selector feature, threshold, mean authority or football coefficient changes.

## Required C2 integrity

For every stage:
- exactly the frozen 30 Week-3 primary QBs are processed;
- only primary-QB `pass_yards` arrays may be replaced by C2;
- every C2 array is finite and has the same length as the canonical base;
- raw QB mean neutrality <= `1e-10`;
- no sportsbook field enters C2 simulation;
- no Week-3 outcome enters the study.

## Downstream pricing replay

Use the exact preserved paid Week-3 sportsbook rows and the same downstream pricing helpers as the completed materiality audit.

Supported markets:
- pass_yards
- rush_yards
- rec_yards
- receptions

Excluded exactly as before:
- anytime_td: no dedicated science certification;
- rush_att: no paid Week-3 sportsbook rows;
- rush_rec_yards: separate RB Rush+Receiving Conservation V2 pathwise semantics.

Replay both surfaces:
1. `SHAPE_ONLY_FIXED_FINAL_MEAN`
2. `FULL_DOWNSTREAM_PROPAGATION`

## Primary repair gate — non-QB protected board

For protected rows excluding `pass_yards`, the specialist transitions must now produce:
- max absolute fair-probability movement = 0;
- max absolute EV movement = 0;
- preferred-side flips = 0;
- HAS EDGE/PASS quote flips = 0;
- Best Snapshot BET/PASS flips = 0;
- Best Snapshot identity changes = 0;
- top-10 turnover = 0;
- top-25 turnover = 0.

This is the direct downstream consequence expected from exact protected-array isolation.

## QB pass-yards interpretation

QB C2 is a receiving-process distribution specialist. Changing TE/WR entitlement can legitimately change C2 passing-yard **shape** even while the QB mean remains fixed.

Therefore pass-yards rows are reported separately and are **not required to be invariant across specialist stages**.

For pass_yards, report:
- fair-probability movement;
- EV movement;
- BET/PASS changes;
- mean neutrality;
- distribution compatibility versus the current C2 implementation.

Do not classify a deterministic C2 response to changed receiver mix as RNG-path contamination.

## Distribution compatibility

Across preregistered seeds:
`1042, 2042, 3042, 4042, 5042, 6042`

compare the final-state current canonical+C2 simulation with the final-state isolated-core+isolated-C2 simulation.

Report QB pass-yards:
- mean absolute mean difference;
- p95 / max mean difference;
- mean absolute SD difference;
- p95 / max SD difference;
- p10 / p50 / p90 absolute quantile differences.

This is descriptive evidence only.

## Frozen dispositions

### `QB_C2_RNG_ISOLATION_DOWNSTREAM_PASS`
if:
- all C2 integrity gates pass;
- non-QB protected downstream movement is exactly zero on both pricing surfaces;
- intentional specialist movement remains live;
- production remains unchanged.

### `QB_C2_RNG_ISOLATION_DOWNSTREAM_FAIL`
if any non-QB protected probability/EV/decision movement remains or any integrity gate fails.

No disposition authorizes a production merge.

## Next action after PASS

Freeze a separate **production repair candidate plan** that ports semantic RNG substreams + specialist-room hierarchical allocation into production code, then:
1. replay the preserved Week-3 board counterfactually;
2. prove production contracts and paid-board population remain intact;
3. compare old-vs-repaired board only as a mechanical counterfactual;
4. do not rewrite the historical Week-3 board;
5. do not use Week-3 outcomes to justify the repair.

Only after that mechanical candidate passes may production integration be considered.
