# Specialist RNG Isolation V1 — Frozen Shadow Design

**STATUS: FROZEN BEFORE CANDIDATE SCORING. RESEARCH-ONLY. NO PRODUCTION MUTATION.**

## Authority

Canonical repository: `dkaps6/imtiredofthis`

Antecedent materiality authority:
- paid Week-3 Full Slate run: `36293274478`
- paid artifact: `10923570170`
- paid artifact digest: `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- Specialist MC materiality run: `36362165079` = SUCCESS
- result artifact: `10946047896`
- result digest: `sha256:ed4b129463316ac5fb8f7f5727cddf10ff7e1b22866ee5b6c224e5dce48e5cd0`
- result head: `a651c875c25f5f3b9eec48cea75eebf137f57788`
- Issue #535 continuity comment: `5870589884`

Frozen antecedent disposition:

`SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE`

That audit authorized **RNG-isolation design only**. It did not authorize a production repair.

No Week-3 outcome may enter this study. No OddsAPI acquisition is allowed. No football coefficient, specialist entitlement, sportsbook threshold, ensemble weight, QB M89/M90 authority, QB C2 selector rule, RB conservation rule, or discrete-count rule may be retuned.

## Problem

The current canonical joint simulator consumes one global NumPy generator in sequence.

A specialist can preserve another player's football input exactly while changing a multinomial probability vector elsewhere. NumPy's finite draw path then changes, and all later consumers of the same global generator may receive different random variates. V1 proved this changes protected simulation outputs. The downstream materiality audit proved the movement can change betting decisions beyond the preregistered ordinary-resampling envelope.

The repair target is therefore **randomness routing**, not football science.

## Design hypothesis

Replace order-coupled global RNG consumption in the shadow candidate with deterministic semantic substreams.

The seed for a substream is a stable digest of:
- base seed,
- semantic layer,
- event,
- team,
- optional player / room.

The candidate also replaces the flat target multinomial with a mathematically equivalent hierarchical multinomial at the two already-conserved specialist seams:

1. changed TE specialist rows form one `TE_CHANGED_ROOM`;
2. changed WR-R15 rows form one `WR_CHANGED_ROOM`;
3. every player outside those mutable rooms remains an individual top-level category;
4. residual target mass remains its own top-level category;
5. room totals are allocated at the top level;
6. conditional multinomials allocate each mutable room internally.

Because the specialist contracts conserve the mutable room's total probability, the top-level probability vector is identical across the corresponding before/after specialist states. With the same keyed top-level RNG, protected players therefore receive exactly the same target counts. The changed room still redistributes internally as intended.

Hierarchical multinomial factorization is distributionally equivalent to the original flat multinomial for a fixed probability vector. The study must nevertheless verify empirical distribution compatibility because the finite sample path changes.

## Shared-state contract

The candidate must preserve intended football dependence.

These remain shared within the same game/team exactly as concepts:
- game pace shock;
- team play volume;
- team pass-rate shock;
- team pass-attempt / rush-attempt totals;
- team pass-efficiency shock;
- team rush-efficiency shock.

These receive stable game/team keyed streams.

The following receive independent semantic streams so unrelated upstream draw consumption cannot perturb them:
- target top-level allocation;
- TE mutable-room allocation;
- WR mutable-room allocation;
- rushing allocation;
- player catch process;
- player receiving-yard noise;
- player rushing-yard noise;
- player QB passing noise;
- player TD process.

The candidate must not create independent team states that were shared in production.

## Frozen candidate populations

Use the exact preserved Week-3 football universe and entitlement traces.

For research proof only, mutable room membership is defined from the already-frozen exact specialist deltas:
- TE mutable room: `abs(te_entitlement - m38_entitlement) > 1e-12`;
- WR mutable room: `abs(final_entitlement - te_entitlement) > 1e-12`.

This is not permission for a production implementation to infer groups post hoc. A production candidate, if later authorized, must obtain room membership from explicit specialist scope metadata.

## Candidate stages

Replay:
1. M38 explicit entitlement;
2. TE-R5P entitlement;
3. final TE-R5P + WR-R15 entitlement.

Use 25,000 iterations and production base seed 42.

## Primary exact-isolation gates

Before QB C2 substitution, for every player protected by the corresponding specialist contract:

### M38 -> TE-R5P
All protected:
- pass_yards,
- rush_att,
- rush_yards,
- receptions,
- rec_yards,
- rush_rec_yards

must have:
- identical array shape;
- `max_element_gap <= 1e-12`;
- `abs(mean_delta) <= 1e-12`.

### TE-R5P -> WR-R15
The same exact gates apply to WR-R15 protected players.

A single protected-key failure means the candidate has not solved the path-dependence defect.

## Intentional-change gate

The isolation candidate must not freeze the specialist itself.

For each stage, at least one specialist-changed player must have a changed receiving simulation array. If all targeted specialist rows become identical, the candidate is invalid.

## Conservation gates

For every event/team/iteration:
- target allocation total across modeled players cannot exceed pass attempts;
- rushing allocation total cannot exceed rush attempts;
- no negative counts;
- no non-finite simulated values.

Specialist room totals remain the already-certified football authorities. This study does not alter entitlement values.

## Distribution compatibility gates

Exact equality to the old finite sample is neither expected nor desired.

For the final football state, compare the isolated candidate with the canonical simulator across preregistered seeds:
`1042, 2042, 3042, 4042, 5042, 6042`.

Report by market:
- mean absolute mean difference;
- p95 and max absolute mean difference;
- mean absolute SD difference;
- p95 and max absolute SD difference.

This is descriptive V1 evidence. It does not authorize a repair by itself.

## QB C2

V1 first proves isolation at the canonical pre-C2 simulation seam.

The current C2 receiving-process generator has its own RNG path and may require the same semantic-stream treatment. V1 must report this honestly rather than masking it.

If pre-C2 exact isolation passes, the next authorized substep is:
`QB_C2_RNG_ISOLATION_EXTENSION_V1`

unless existing C2 replay is already invariant under the candidate by construction.

## Downstream board gate

No claim that the betting-board defect is repaired is allowed until an isolated candidate is propagated through the exact preserved downstream pricing path and the previous:
- fair-probability movement,
- EV movement,
- HAS EDGE/PASS flips,
- Best Snapshot BET/PASS flips,
- identity changes

are remeasured.

## Frozen dispositions

### `SPECIALIST_RNG_ISOLATION_CORE_PASS`
if both specialist stage comparisons have zero protected-key drift pre-C2, intentional specialist rows still move, and all conservation/integrity gates pass.

### `SPECIALIST_RNG_ISOLATION_CORE_FAIL`
if any protected-key drift remains or any integrity/conservation gate fails.

### `SPECIALIST_RNG_ISOLATION_INVALID_FREEZE`
if the candidate accidentally suppresses intended specialist changes.

No disposition in this study authorizes a production merge.

## What remains forbidden

Do not:
- merge the shadow simulator into production;
- change `scripts/simulation_v2.py`;
- change `scripts/simulation_c2_qb_candidate.py`;
- increase MC iterations as a substitute for isolation;
- tune a betting threshold around Week 3;
- use Week-3 outcomes;
- use sportsbook lines to construct RNG groups;
- alter TE-R5P / WR-R15 entitlement science;
- reopen closed RB/TE/QB feature hunts.

## Exact next action after a core PASS

Extend the same semantic-stream architecture to the C2 receiving-process candidate, then rerun the frozen downstream materiality surface on the preserved paid board. Only if that eliminates the excess decision-boundary movement without violating distribution/conservation contracts may a separately frozen production repair candidate be designed.
