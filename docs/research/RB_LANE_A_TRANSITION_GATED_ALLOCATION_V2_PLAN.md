# RB Lane A — Transition-Gated Carry-Share Reallocation V2

**FROZEN AFTER V1 CONSTRUCTIBILITY FAILURE, BEFORE ANY OUTCOME SCORING. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Why V2 exists

Lane A V1 is terminal and is not being rescued.

Authoritative V1 pre-outcome run:
- run `35167213629`
- head `ea432a2686639c6064c23aeead084944bbe7bb25`
- artifact `10474579202`
- digest `sha256:8c92c92ae713f4eedeb73acfdb94f3f0191c9fb7d5a79794581107eaa17e96ea`
- final disposition: `RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE`

V1 passed Gate 0, same-job promotion-comparator authority for both rotations, and Rotation-2 mechanism-comparator authority, then failed Amendment-8 constructibility before outcomes were opened. No candidate-vs-actual MAE, bootstrap, p90/catastrophic, protected-cohort, per-season, or whole-season performance result exists for V1.

The exact V1 failure was structural: the roster-derived post-transition active room contained RB rows that did not exist in the live-production-style promotion comparator universe for that same `(season, week, team, player_clean_key)`. Because Amendment 8 explicitly prohibited repairing V1 after seeing its constructibility coverage, V1 remains failed exactly as run.

V2 is a new prospective experiment that tests the same allocation hypothesis in the universe that can actually be deployed by production.

## Frozen V2 scientific question

On the same leakage-safe loss/vacancy transition team-weeks, does the already-frozen HHI-dampened carry reallocation improve rushing-yard projection accuracy versus the real Weeks-2-18 promotion comparator when allocation is performed only among backs that the production projection universe can actually score that week?

No V1 outcome was opened before this V2 plan was frozen.

## What remains unchanged from V1

The following are inherited byte-for-byte / logically unchanged unless this document explicitly says otherwise:

- Gate 0 and all source contracts.
- Broad detected-transition disclosure population.
- Narrow scored-V1/V2 transition definition: loss/vacancy only.
- Both temporal OOS rotations: 2023-fit -> 2024 test and 2024-fit -> 2025 test.
- Same-job Build-A / Build-B authority reconstruction.
- Per-rotation production ensemble weights and `calibration_season < test_season` assertions.
- Team-level predicted pregame conservation pool.
- Prior-3 RB carry-share role weight.
- Pre-transition HHI calculation.
- Frozen exponent `p = 1 + 2*H`.
- No sportsbook input.
- Amendment-8 held-incumbent-efficiency rush-yard translation.
- Stable/detected-but-not-scored weeks remain exactly equal to the promotion comparator.
- Rotation-2 mechanism diagnostic remains diagnostic-only; Rotation 1 remains `NOT_CONSTRUCTIBLE_NO_CASEBOOK` for that diagnostic.
- All adequacy, protected-cohort, bootstrap, crossed-bootstrap, per-season, p90/catastrophic, stable-identity, conservation, and whole-season safety gates.
- No rescue tuning after outcome exposure.

## V2 recipient universe — the only scientific change

For each scored transition team-week, begin from the same leakage-safe post-transition active RB-room roster used by V1.

Construct the dual-market promotion comparator **first**, from the same Amendment-6 Build-A source and frozen per-rotation weights.

A player is a V2 eligible recipient if and only if all of the following are true:

1. The player is in the post-transition active RB room under the frozen Gate-0 roster/status contract.
2. Exact identity match exists on `(season, week, team, player_clean_key)` in the dual-market promotion comparator.
3. `promotion_rush_yards_i` is finite.
4. `promotion_rush_att_i` is finite and `promotion_rush_att_i > 0.20`.

No fuzzy identity join, imputation, fallback efficiency, clipping, row synthesis, or projection fabrication is permitted.

The V2 eligible-recipient set is therefore the intersection of:

`post_transition_active_rb_room ∩ production_scoreable_dual_market_rb_universe`.

This is not a post-result performance filter. It is a deployability contract: V2 may only redistribute the production carry pool among players for whom the incumbent production route itself can emit both a carry and rushing-yard mean.

## Team-week coverage rule

No scored transition team-week may be dropped merely because some active-room players are not production-scoreable.

For every scored transition team-week:

- if at least one V2 eligible recipient exists, reallocate the **entire unchanged team-level pool** among the eligible recipients only;
- if zero V2 eligible recipients exist, fail closed with `V2_RECIPIENT_UNIVERSE_FAILURE` before any outcome is opened.

The pre-outcome V1 evidence has already established that the currently frozen two rotations contain at least one production-scoreable recipient in every scored transition event (112/112 in 2024 and 108/108 in 2025). This fact is disclosed because it was learned before V2 outcome scoring; it is not an outcome metric and no threshold was tuned from it.

## V2 allocation formula

For eligible recipients only:

- `raw_w_i = prior3_rb_share_i`
- `w_i = raw_w_i / sum(raw_w_j over eligible recipients)`
- retain the same pre-transition-room HHI `H` from V1 (computed over the full pre-transition room, not the filtered recipient set)
- `p = 1 + 2*H`
- `v_i = w_i^p`
- `recipient_share_i = v_i / sum(v_j)`
- `candidate_att_i = recipient_share_i * pool`

Hard assertions:

- `sum(candidate_att_i over eligible recipients) == pool` within the already-frozen conservation tolerance;
- eligible-recipient `raw_w` sum must be positive, otherwise fail closed with `V2_RECIPIENT_WEIGHT_FAILURE` before outcomes.

Players outside the production-scoreable recipient set receive no synthetic candidate row. V2 never manufactures a projection for a player absent from the promotion comparator universe.

## Rush-yard translation

Unchanged Amendment-8 design:

`incumbent_ypc_i = promotion_rush_yards_i / promotion_rush_att_i`

`candidate_rush_yards_i = candidate_att_i * incumbent_ypc_i`

Only carry allocation changes. Per-player incumbent efficiency remains fixed.

## Scoring universe

The decisive transition comparison is performed on the same promotion-comparator player rows for the scored transition team-weeks. V2 candidate values replace the promotion comparator only for eligible recipients on scored transitions; every other production row remains exactly the promotion comparator.

No actual/postgame field may be touched until all of the following pass for both rotations:

1. Gate 0.
2. Same-job authority reconstruction.
3. Rotation-2 mechanism authority (diagnostic lane only).
4. V2 recipient-universe integrity.
5. V2 eligible-recipient positive-weight integrity.
6. Rush-yard translation constructibility.
7. Conservation.

Only after those pre-outcome checks pass may the previously frozen outcome gates run.

## Frozen disposition rules

Pre-outcome failures retain their exact fail-closed labels and stop the run before outcome scoring.

If all pre-outcome checks pass, use the unchanged V1 decisive qualification gates. No threshold, cohort, bootstrap rule, exponent, trailing window, pool formula, comparator, or safety gate may be changed after outcomes are exposed.

A V2 failure is preserved. There is no V2 rescue/tuning loop.
