# Offensive Regime Boundary Room History V1 — Stage A Result

Status: **FAILED CLOSED**

Disposition:

`OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_FAILED_CLOSED`

The parent Receiver Room Targets-Per-Play V1 remains failed closed. No Stage-B run is authorized. No production integration is authorized.

## Provenance

Frozen plan:
`docs/research/OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_PLAN.md`

Research branch:
`research-offensive-regime-boundary-room-history-v1`

Frozen plan commit:
`9e021673e6f5eabaaa5015346bf68c8c3890dac5`

Initial evaluator commit:
`a7307dac4884fc65dce0361805cc3e690fd8150e`

Initial workflow head:
`444883f526a37c5fb2cd8419eaddd1b9f0c5f67e`

First run:
`36177363204`
job:
`108211110926`

The first run completed the expensive walk-forward calculations but stopped during descriptive CSV serialization because the runner's pandas version did not accept `include_groups=False` in `GroupBy.apply`.

That run produced no scorecard artifact and has no scientific disposition.

Bounded compatibility repair:
`106013927ba916f6456f8b694445684daae9b636`

The repair changed only the post-score descriptive group aggregation from `groupby.apply` to an explicit deterministic loop.

Unchanged:
- candidate formula;
- QB regime-boundary definition;
- 2020/2021 cohort;
- room history semantics;
- zero-history league fallback;
- baseline and parent comparators;
- all frozen gates;
- candidate count.

Authoritative Stage-A rerun:

- run: `36177811260`
- head: `106013927ba916f6456f8b694445684daae9b636`
- artifact: `10882674429`
- artifact name: `offensive-regime-boundary-room-history-v1-stage-a`
- artifact digest: `sha256:a981657eedea2fb248dd004202bfcd2830ab669d8b85ef2945ffcdec311c5b77`

Controls:

- candidate variants scored: **1**
- parameters fit: **0**
- sportsbook inputs: **0**
- target-game outcomes used upstream: **0**
- 2026 outcomes used for fitting: **0**
- production mutations: **0**

## Frozen candidate

The candidate applied one hard, pregame regime boundary to WR / TE / RB_FB alike.

If the prior-season primary QB was still on the target-week pregame roster, the candidate was numerically identical to the unchanged parent room-history construction.

If the prior-season primary QB was absent, prior-season team room history was discarded and the candidate used strict-prior current-season team games only.

At zero current-season prior games, the candidate used the prior-season league room targets-per-play rate as the frozen zero-history fallback.

No window, minimum-games threshold, blend, shrinkage, recency weight, WR-only route, play-caller variant, bias offset, sportsbook input or 2026 outcome was used.

## Stage-A scorecard

Stage A was the untouched reverse-time falsification cohort:

- 2020 regular season Weeks 1-17;
- 2021 regular season Weeks 1-18.

These seasons were not used in the 2022-2025 regime-instability audit that generated the hypothesis.

### 2020

Macro MAE:

`3.576432 -> 3.289737 -> 3.285100`

for:

`production baseline -> unchanged parent targets-per-play -> regime-boundary candidate`

WR MAE:

`5.141964 -> 4.819280 -> 4.823976`

The candidate was slightly better than the parent on macro MAE by about `0.004637`, but slightly worse than the parent on WR MAE by about `0.004697`.

### 2021

Macro MAE:

`3.435789 -> 3.278513 -> 3.341461`

WR MAE:

`4.947699 -> 4.812740 -> 4.929359`

The hard boundary materially degraded the unchanged parent in 2021.

### Pooled 2020-2021

Macro MAE:

- baseline: `3.503980`
- parent: `3.283955`
- boundary candidate: `3.314134`

WR MAE:

- baseline: `5.041888`
- parent: `4.815911`
- boundary candidate: `4.878264`

TE MAE:

- baseline: `2.731921`
- parent: `2.580608`
- boundary candidate: `2.554283`

RB/FB MAE:

- baseline: `2.738130`
- parent: `2.455346`
- boundary candidate: `2.509856`

Candidate-vs-baseline closer rate:

`54.10%`

Macro p90:

- baseline: `7.091715`
- parent: `6.641750`
- boundary candidate: `6.822284`

Macro absolute bias:

- baseline: `0.545323`
- parent: `0.351313`
- boundary candidate: `0.366672`

The candidate retained much of the original parent architecture's aggregate benefit over production baseline, but it did not improve the parent architecture and did not satisfy the frozen structural mechanism gates.

## Critical boundary-cohort result

The direct mechanism test failed.

On rows where the prior-season primary QB was absent from the target-week pregame roster:

### WR only

Parent MAE:

`5.138451`

Boundary candidate MAE:

`5.323931`

Parent p90:

`10.371903`

Boundary candidate p90:

`10.975617`

Candidate closer rate versus parent:

`47.04%`

### All rooms

Parent MAE:

`3.514907`

Boundary candidate MAE:

`3.604680`

Parent p90:

`7.510626`

Boundary candidate p90:

`7.964502`

Candidate closer rate versus parent:

`49.77%`

Therefore the exact population the intervention was designed to repair became **worse**, not better.

## Summed-room result

Summed-room MAE:

- baseline: `6.574364`
- parent: `6.580113`
- boundary candidate: `6.659841`

Summed-room p90:

- baseline: `13.172576`
- parent: `13.417463`
- boundary candidate: `13.785218`

The candidate also produced a large summed-room absolute bias:

- baseline: `0.184143`
- candidate: `1.100015`

The frozen gate set did not separately gate summed-room absolute bias, but the magnitude reinforces the failure rather than rescuing it.

## Frozen gates

### Passed

- pooled macro MAE improves vs production baseline;
- 2020 macro MAE improves vs baseline;
- 2021 macro MAE improves vs baseline;
- pooled WR MAE improves vs baseline;
- 2020 WR MAE improves vs baseline;
- 2021 WR MAE improves vs baseline;
- pooled TE MAE nonworse vs baseline;
- pooled RB/FB MAE nonworse vs baseline;
- pooled macro p90 nonworse vs baseline;
- pooled macro absolute bias nonworse vs baseline;
- candidate closer rate > 50% vs baseline;
- stable-regime rows identical to unchanged parent;
- target-game outcomes upstream = 0;
- sportsbook inputs = 0;
- parameters fit = 0;
- candidate variants scored = 1;
- room rates finite and bounded;
- summed room rate <= 1;
- QB-boundary source coverage = 100%.

### Failed

- **summed-room MAE improves vs production baseline**
- **summed-room p90 nonworse vs production baseline**
- **QB-boundary WR MAE better than unchanged parent**
- **QB-boundary all-room MAE better than unchanged parent**

All frozen gates were mandatory.

## Scientific interpretation

The 2022-2025 diagnostic finding and the 2020-2021 candidate failure are compatible.

The diagnostic established that primary-QB and verified play-caller transitions are useful **instability markers**: the unchanged team room-history forecast was more fragile around those transitions.

Stage A tested a different claim:

> if a QB transition marks instability, discarding the prior team's room history at that boundary should improve prediction.

That causal intervention did **not** survive untouched falsification.

Therefore:

- primary-QB transition remains informative as a description of where room-history error is elevated;
- it does **not** justify throwing away prior-season team room history;
- no hard reset architecture is supported;
- no minimum-games rescue, crossover week, recency weight, shrinkage factor, fallback change, WR-only route or blended reset may be searched after this result.

This is a useful negative result: an instability marker is not automatically the correct correction.

## Disposition and frontier closure

**OFFENSIVE_REGIME_BOUNDARY_ROOM_HISTORY_V1_STAGE_A_FAILED_CLOSED**

Stage B on 2018-2019 is **not authorized** and must not run.

The same-team receiver-room-history family is now closed under the user's standing research rule:

- Receiver Room Targetable-Rate V1: closed;
- Active-Roster Receiver Room State V1: closed;
- WR1-only current-state room rescue: closed;
- Receiver Room Targets-Per-Play V1: failed confirmation;
- Offensive Regime Boundary Room History V1: failed untouched Stage A.

Do not search room-history windows, weights, recency, shrinkage, resets, roster filters or subgroup routers.

The next research lane must use genuinely new pregame information or a materially different architecture rather than another transformation of same-team historical room rates.
