# Specialist RNG Isolation Production-Repair Counterfactual V1 — Frozen Plan

**STATUS: FROZEN BEFORE SCORING. MECHANICAL COUNTERFACTUAL ONLY. NO PRODUCTION MUTATION.**

## Authority

Canonical paid Week-3 board:
- Full Slate run `36293274478`
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- paid head `0982b62276303403e2ca58b16e6f4fc3e041f65d`

Research antecedents:
- `SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE`
  - run `36362165079`
  - artifact `10946047896`
  - Issue #535 comment `5870589884`
- `SPECIALIST_RNG_ISOLATION_CORE_PASS`
  - run `36427943860`
  - artifact `10971812982`
  - Issue #535 comment `5870771354`
- `QB_C2_RNG_ISOLATION_DOWNSTREAM_PASS`
  - run `36429000177`
  - artifact `10972977687`
  - Issue #535 comment `5870982920`

No Week-3 outcomes. No OddsAPI acquisition. No production mutation.

## Question

> If the exact paid Week-3 football state were simulated with the isolated semantic-RNG architecture instead of the order-coupled global RNG, how much would the actual production betting board change, and is that movement larger than ordinary finite-MC resampling of the current simulator?

This is a mechanical counterfactual. It does not ask whether the changed board would have won more bets.

## Frozen populations

Use the exact preserved paid artifact and exact supported paid rows for:
- pass_yards
- rush_yards
- rec_yards
- receptions

As in the completed materiality study:
- anytime_td remains excluded from the primary gate because dedicated ATD science is not certified;
- rush_att has no paid sportsbook board;
- rush_rec_yards remains outside this candidate because its production path includes separate RB Rush+Receiving Conservation V2 draw semantics that require an independent repair integration gate.

The paid historical board itself must never be overwritten or re-labeled.

## Counterfactual candidate

Use:
- final TE-R5P + WR-R15 football entitlement state;
- Specialist RNG Isolation V1 core semantic streams and hierarchical specialist-room allocation;
- QB C2 RNG Isolation Extension V1;
- exact paid downstream football mean authorities and sportsbook offers.

No football coefficient, specialist entitlement, ensemble weight, QB selector feature, M89/M90 mean authority, line, odds or publication threshold changes.

## Two pricing surfaces

### A. SHAPE_ONLY_FIXED_FINAL_MEAN
Hold every paid row's final `model_proj` fixed to the actual historical paid value.

This isolates how much the repaired simulation **shape/probability surface alone** would have changed the board.

### B. FULL_DOWNSTREAM_PROPAGATION
Allow the repaired finite-MC `mc_proj` to flow through the existing frozen ensemble and QB synthesis machinery exactly as production does.

This measures the complete mechanical counterfactual without changing football science.

## Paid-board comparison metrics

For each surface report:
- supported side rows;
- quotes;
- player-markets;
- mean / p95 / p99 / max absolute fair-probability movement;
- mean / p95 / p99 / max absolute EV movement;
- preferred-side flips;
- HAS EDGE/PASS quote flips;
- Best Snapshot BET/PASS flips;
- Best Snapshot side flips;
- Best Snapshot identity changes;
- mean / max Best-EV movement;
- Best-EV Spearman;
- top-10 turnover;
- top-25 turnover.

Also report by market.

## Ordinary-resampling benchmark

Using the **current production simulator + current C2 implementation**, replay the same exact final football state at:
`1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042, 10042, 11042, 12042`.

Each alternate-seed board is compared to the reconstructed production-seed-42 current board on the same two surfaces.

The ordinary-resampling envelope is the maximum observed alternate-seed value for each primary metric.

This benchmark uses the same football state and sportsbook rows. It does not use outcomes.

## Integrity gates

Before candidate scoring:
1. current production-seed-42 reconstruction must match paid `mc_proj`, `model_proj` and fair probability within the already-proven replay tolerances;
2. paid board population and provider/player identities must match exactly;
3. production source files remain byte-identical to paid-head authority;
4. isolated candidate uses no sportsbook fields to create football inputs or RNG groups;
5. Week-3 outcomes loaded = false;
6. new OddsAPI request = false;
7. production mutation = false.

Any failure => `RNG_ISOLATION_COUNTERFACTUAL_INTEGRITY_FAILURE`.

## Primary decision rule

The isolated candidate is mechanically acceptable for **production-repair design** only if:

1. all integrity gates pass;
2. candidate Best-EV rank correlation remains >= 0.99 on both surfaces;
3. candidate top-10 turnover <= ordinary-resampling max on the matching surface;
4. candidate top-25 turnover <= ordinary-resampling max;
5. candidate Best Snapshot BET/PASS flips <= ordinary-resampling max;
6. candidate Best Snapshot identity changes <= ordinary-resampling max;
7. candidate p99 fair-probability movement <= ordinary-resampling max p99 fair-probability movement;
8. candidate p99 EV movement <= ordinary-resampling max p99 EV movement.

The comparison is deliberately against normal finite-MC uncertainty rather than zero movement, because changing RNG architecture necessarily changes the finite sample.

## Additional strict sanity gate

No individual candidate probability may move by more than 5 percentage points from the paid board on supported rows.

This is a fixed engineering sanity bound, not a betting threshold and not fitted from outcomes.

## Frozen dispositions

### `RNG_ISOLATION_COUNTERFACTUAL_WITHIN_ORDINARY_MC_ENVELOPE`
if every integrity and primary gate passes.

This authorizes a separately frozen production-code repair candidate. It does **not** authorize a merge.

### `RNG_ISOLATION_COUNTERFACTUAL_EXCEEDS_ORDINARY_MC_ENVELOPE`
if integrity passes but any primary candidate metric exceeds the ordinary-resampling envelope or strict probability sanity bound.

No production port is authorized.

### `RNG_ISOLATION_COUNTERFACTUAL_INTEGRITY_FAILURE`
if any source, reconstruction, identity or no-outcome/no-Odds integrity gate fails.

## What this study cannot claim

Do not use this audit to claim:
- improved Week-3 predictive accuracy;
- better historical win rate;
- better calibration;
- a new betting edge;
- a new confidence threshold;
- that the football model is healthy.

Those questions remain reserved for the frozen Week-3 postmortem after complete outcomes.

## Exact next action after a PASS

Freeze a production-code repair candidate that:
1. adds semantic keyed RNG routing without changing football inputs;
2. uses explicit TE-R5P / WR-R15 scope metadata already present in production, not post-hoc outcome/delta inference;
3. updates the legacy-vs-explicit baseline parity contract from bitwise finite-sample equality to a mathematically appropriate semantic/distributional invariant without weakening football-mean/conservation protections;
4. preserves the old paid Week-3 board as historical truth;
5. generates only a labeled counterfactual repaired board for comparison;
6. passes the full production test/CI suite before any merge consideration.
