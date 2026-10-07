# RB RECEIVING ROOM SHARE IMPACT V1 — FROZEN CONTRACT

Date: 2026-10-07
Branch: `research-rb-receiving-room-share-impact-v1`

## Purpose

Answer two separate questions without conflating them:

1. **Retrospective impact question:** If the now-identified RB receiving-room share
   mechanism had been applied to 2026 Weeks 1-4, would individual RB receiving
   projections have moved closer to the realized outcomes?
2. **Prospective validation question:** Does the same no-fit mechanism continue
   to work out of sample from Week 5 forward?

Weeks 1-4 are valid for the first question. They are **not** clean promotion
evidence because Weeks 1-4 were used to discover/localize this mechanism.

No production promotion is authorized by this contract.

## Parent evidence

The frozen player-level decomposition established:

- RB targets are **player-share dominant**, not team-volume dominant.
- Actual player-share diagnostics remove about **76.5%** of RB target MAE.
- Team-volume diagnostics remove only about **2.8%**.

The frozen RB receiving-state audit established:

- current within-RB-room share MAE: **0.22336**
- raw strict-prior `prior_rb_room_share` MAE: **0.18807**
- no-fit improvement: about **15.8%**
- current and prior-state leader identification are nearly identical;
  the gain is **share concentration**, not leader selection.

## Frozen mechanism

Use only:

`prior_rb_room_share`

Reason for freezing:
- it is the simplest full-history strict-prior room-share state;
- it has the broadest usable coverage among the predeclared room-state family;
- it requires no fit, coefficient, threshold, or outcome-derived window.

### Room redistribution rule

Apply after the protected receiving entitlement sequence:

`M38 -> TE-R5P -> WR-R15 -> RB receiving-room redistribution -> simulation`

For every team/game:

1. Identify current active RB/FB rows.
2. Preserve the exact total RB/FB `entitlement_tgt_share` mass.
3. Convert current RB/FB entitlement to normalized current room shares.
4. Let **A** be RB/FB rows with finite, nonnegative strict-prior
   `prior_rb_room_share`.
5. Let **M** be RB/FB rows without usable history.
6. Preserve every player in **M** at the exact current canonical room share.
7. Remaining room mass =
   `1 - sum(current_room_share[M])`.
8. If at least two players are in **A** and `sum(prior_rb_room_share[A]) > 0`,
   allocate remaining room mass across **A** proportional to their
   `prior_rb_room_share`.
9. Otherwise the entire room is an exact no-op.
10. Multiply the new room shares by the exact preserved total RB/FB entitlement
    mass.

This rule is structural, not fitted.

## Protected invariants

The transform must preserve within tolerance:

- total team target entitlement
- total RB/FB target entitlement
- every WR entitlement
- every TE entitlement
- every QB/non-RB entitlement
- team pass/dropback volume
- RB rush allocation
- receiving efficiency inputs
- R22 tail/distribution behavior
- sportsbook separation

No target/future-week outcomes may enter the transform.

## Weeks 1-4 retrospective impact replay

Run both baseline and candidate using the same:
- **ACT-only historical availability-parity pregame universes**
- schedule/team history/player history
- seeds and iterations
- M38
- TE-R5P
- WR-R15
- ensemble weights
- QB authority
- RB rushing authorities

Only the RB receiving-room redistribution differs.

Score individual RB/FB player-game results for:
- targets
- receptions
- rec_yards
- rush_rec_yards

Also verify:
- rush_yards point projections are unchanged
- non-RB receiving projections are unchanged except negligible MC sampling noise;
  where exact same RNG path can be preserved, require exact parity.

Required metrics:
- MAE
- median absolute error
- signed bias
- RMSE
- candidate closer / baseline closer / tie counts
- per-week results
- high-target/high-receiving-workload descriptive bins
- canonical live-board overlap where available

This replay is explicitly labeled:

`RETROSPECTIVE_MECHANISM_IMPACT__NOT_OOS_PROMOTION_EVIDENCE`

## Week-5 prospective freeze

After the mechanism code is certified, create a Week-5 pregame lock using only
strict-prior information available before Week-5 outcomes.

The Week-5 lock must contain:
- player identity
- current canonical RB room share
- `prior_rb_room_share`
- candidate RB room share
- total RB entitlement mass
- history availability/fallback state
- exact digests
- zero sportsbook inputs
- zero Week-5 outcomes

This prospective lock is the clean validation path.

## Interpretation

Weeks 1-4 answer:

> Would this mechanism have improved the individual player projections we already
> made?

Week 5+ answers:

> Does it still improve them after the mechanism is frozen?

Both questions matter. The first gives us immediate impact information; the
second prevents us from promoting a mechanism because we discovered it on the
same outcomes used to judge it.

## Prohibited

- no coefficient fit
- no threshold search
- no selecting a different history window after seeing impact results
- no automatic promotion
- no paid OddsAPI
- no sportsbook inputs upstream
- no reopening RB receiving-yard mean science
- no back-applying the separate Week-5 carry/snap shadow
