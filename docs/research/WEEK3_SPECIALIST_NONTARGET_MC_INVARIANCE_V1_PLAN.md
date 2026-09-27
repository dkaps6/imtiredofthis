# Week-3 Specialist Non-Target Monte Carlo Invariance V1 — Frozen Audit Plan

**STATUS: FROZEN BEFORE FORMAL AUDIT SCORING. READ-ONLY SYSTEMS AUDIT. NO PRODUCTION CHANGE.**

## Authority / source

Canonical repository: `dkaps6/imtiredofthis`

Frozen Week-3 source:
- paid Full Slate run: `36293274478`
- artifact: `10923570170`
- artifact digest: `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- paid-run head: `0982b62276303403e2ca58b16e6f4fc3e041f65d`
- current research base: `37deff5b51b5ec48117556b91d12dba1d4ca1fad`

No Week-3 outcomes are used. No new OddsAPI request is allowed. Sportsbook data may not define football inputs or the audited protection sets.

## Why this audit exists

The promoted receiving stack is intentionally hierarchical:

`M38 / explicit finite target pool -> TE-R5P inside the conserved TE room -> WR-R15 inside the conserved WR2+ room -> joint MC`

The specialist contracts explicitly certify:
- TE-R5P preserves non-TE entitlement and total team opportunity;
- WR-R15 preserves non-WR entitlement, the M38 WR1 anchor, the WR2+ room total, and total team opportunity.

Production then re-runs the entire explicit-entitlement simulator after each specialist using the same nominal seed.

The existing Full Slate audit records how many simulation keys change at each stage, but it does not distinguish:
1. outputs whose football entitlement/input was intentionally changed; from
2. outputs whose football inputs were explicitly protected but whose finite Monte Carlo sample changed because a different probability vector elsewhere changed the shared RNG path.

This is a systems/data-lineage question, not a new football hypothesis.

## Frozen question

> When a specialist leaves a player's football entitlement exactly unchanged, does re-simulating the full joint state change that protected player's empirical Monte Carlo output anyway?

A positive answer establishes **path-dependent Monte Carlo drift**. It does **not** by itself establish that the underlying probability law is wrong or that the drift is materially large enough to affect betting decisions.

## Frozen stage definitions

### Stage A — TE-R5P

Baseline entitlement for every player:
`m38_explicit_entitlement_tgt_share` from `data/target_entitlement_v1_trace.csv`.

TE-only entitlement:
- for TE rows present in `data/te_r5p_full_slate_entitlement_trace.csv`, use `te_r5p_entitlement_tgt_share`;
- for every other player, TE-only entitlement equals the M38 explicit baseline exactly.

Simulation comparison:
`data/te_r5p_full_slate_simulation_delta.csv`
(`m38_baseline` -> `te_r5p`).

A player is **TE-protected** iff its Stage-A entitlement delta is <= `1e-12` in absolute value.

### Stage B — WR-R15

Stage-B baseline is the TE-only entitlement above.

Final entitlement is `entitlement_tgt_share` from `data/target_entitlement_v1_trace.csv`.

Simulation comparison:
`data/wr_r15_full_slate_simulation_delta.csv`
(`te_r5p` -> `wr_r15`).

A player is **WR-protected** iff its Stage-B entitlement delta is <= `1e-12` in absolute value.

This definition is numeric and authority-exact. It does not infer protection from player name, depth chart, sportsbook coverage, or postgame outcome.

## Primary protected-output family

The primary contradiction check uses markets whose football inputs are not supposed to be changed by receiver-room entitlement redistribution:

- `pass_yards`
- `rush_att`
- `rush_yards`

These are the **SEMANTICALLY_UNRELATED** markets.

Receiving-linked markets are still reported descriptively because an unchanged player's own target probability implies an unchanged marginal receiving law, but they are not needed to confirm the primary systems contradiction:

- `receptions`
- `rec_yards`
- `rush_rec_yards`

`anytime_td` is reported separately and is not used as a primary gate because current ATD science is not dedicated-certified.

## Frozen integrity gates

Before any result is interpreted:

1. source run/artifact identifiers match the authority above;
2. target-entitlement trace is nonempty and unique by `(event_id, player_clean_key)`;
3. TE and WR specialist traces are nonempty and unique at their player grain;
4. simulation-delta files have identical key universes across the two compared stages;
5. all compared means/deltas are finite;
6. TE-R5P audit still certifies non-TE/team-room conservation;
7. WR-R15 audit still certifies WR1/WR2+ room/non-WR/team conservation;
8. zero sportsbook fields are used to define protected membership;
9. zero Week-3 outcomes are loaded;
10. production mutation = false.

Any integrity failure => `SPECIALIST_NONTARGET_MC_INVARIANCE_INTEGRITY_FAILURE` and stop.

## Frozen reported metrics

For each specialist stage and for:
- all protected keys;
- protected SEMANTICALLY_UNRELATED keys;
- protected receiving-linked keys;
- each market;
- each position;

report:

- protected player count;
- protected simulation-key count;
- count and rate with `abs_mean_delta > 1e-12`;
- count and rate with `max_element_gap > 1e-12`;
- mean / median / p90 / p95 / p99 / max `abs_mean_delta`;
- max `max_element_gap`;
- signed mean delta;
- top 25 protected drifts by absolute mean delta.

Also report specialist-target rows separately so intentional football changes are never mixed with protected drift.

## Frozen disposition

After integrity gates pass:

### `SPECIALIST_NONTARGET_MC_PATH_DRIFT_CONFIRMED`

if **either** TE-R5P or WR-R15 has at least one protected
`SEMANTICALLY_UNRELATED` simulation key with:
- `abs_mean_delta > 1e-12`, or
- `max_element_gap > 1e-12`.

### `SPECIALIST_NONTARGET_MC_PATH_INVARIANCE_HOLDS`

if both stages have zero such protected unrelated-key drift.

This is deliberately an exact systems invariant, not a tuned magnitude threshold.

## What a positive result DOES NOT authorize

Do **not** from this audit alone:
- change seeds;
- increase MC iterations;
- split RNG streams;
- introduce common-random-number routing;
- cache protected arrays;
- splice arrays position-by-position;
- change TE-R5P or WR-R15 football entitlements;
- change M38;
- alter QB C2 or M89/M90;
- change RB rushing / RB Rush+Receiving V2;
- change sportsbook pricing thresholds;
- claim realized predictive improvement.

If drift is confirmed, the only authorized next step is a separately frozen **downstream materiality audit** that measures how much protected fair probabilities / sides / raw EV / publishability can move solely from this path dependence, using the same frozen football inputs and sportsbook lines strictly downstream.

Only that second-stage evidence can justify considering a mechanical RNG-isolation repair.

## Closed-family protection

This audit must not be used to reopen:
- Rush Pool Evidence Guard / top-five rushing rescue;
- zero-MC rescue variants;
- TE Width V2;
- old C1/C3 receiver formulations;
- Receiver Room Targets-per-Play;
- WR Anchor / Role-Transmission;
- generic copula rescue;
- retrospective M96 routing;
- frozen Week-3 RB Vacancy/Public Intent or Receiving Rule Semantics.

The audit is solely about whether protected football inputs remain protected at the finite-Monte-Carlo output surface.
