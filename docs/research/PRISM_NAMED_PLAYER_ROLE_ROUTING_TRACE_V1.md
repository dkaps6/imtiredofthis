# PRISM — Named-Player Role-Routing Trace V1 (JSN / Kupp / Barner)

**Status:** read-only role-routing trace. **No fitting, no scoring, no promotion.**
**Authorized by:** GPT audit `6441088661`
(`OPPONENT_TRANSMISSION_GENERALIZATION_V2_CLEARED__COMPACT_NAMED_PLAYER_ROLE_ROUTING_TRACE_AUTHORIZED`),
reconfirmed `6443421541`.
**Reproduce:** `python scripts/research/trace_named_player_role_routing_v1.py`
**Artifacts:** `data/research/named_player_routing/named_player_role_routing_v1{.csv,_status.json}`

Sportsbook inputs: **0**. Outcomes inspected: **none**. Week 7 used to score
anything: **no**. Walker redone: **no**.

## The question

> How does canonical role assignment determine which opponent-transmission path
> a named player is actually eligible to receive?

## Population

In 2026 Week 7 all three authorized names are **Seattle** players facing
**Kansas City** — the same-offense / same-opponent contrast the audit asked for,
holding the defensive environment fixed while identity and role vary. Kenneth
Walker III is the *Kansas City* back facing the *Seattle* defense in that same
game, so the cleared Walker case and this trace are opposite sides of one
matchup.

## Answer: WR transmission is rank-gated; TE and RB transmission is position-gated

`simulation_rules.py:201-210` resolves the target multiplier through one
`if/elif` chain:

```python
tgt_mult = 1.0
if   role == "WR1":            tgt_mult *= mods.wr1_target_mult
elif role == "WR1_5":          tgt_mult *= mods.wr1_5_target_mult
elif role == "SLOT":           tgt_mult *= mods.slot_target_mult
elif pos  == "TE":             tgt_mult *= mods.te_target_mult
elif pos  in {"RB", "FB"}:     tgt_mult *= mods.rb_rec_target_mult
base_tgt *= tgt_mult
```

The three WR branches key on the **computed role label**. The TE and RB branches
key on **position**. `_wr_role_labels()` assigns a label to slot receivers plus
only the top two perimeter WRs by target share. So:

- every TE and every RB/FB receives an opponent-coverage target channel
  automatically, by position;
- a WR receives one **only** if he is the slot or one of the top two perimeter
  WRs;
- any other WR falls through the entire chain to `tgt_mult = 1.0`, and **no
  diagnostic records the fall-through.**

An unlabeled WR is the only offensive skill position in production with no
opponent-coverage target channel at all.

## The Seattle receiving room, Week 7

| player | pos | depth | `rules_role` | routed to | channel? | ranking key (raw `tgt_share`) | allocated `bayes_tgt_share` |
|---|---|---|---|---|---|---|---|
| Jaxon Smith-Njigba | WR | WR1 | **WR1** | `wr1_target_mult` | yes | 0.353935 | 0.316891 |
| Rashid Shaheed | WR | WR2 | **WR1_5** | `wr1_5_target_mult` | yes | 0.152958 | 0.157403 |
| **Cooper Kupp** | WR | WR3 | **none** | — | **no** | 0.144752 | 0.151325 |
| Tory Horton | WR | WR4 | none | — | **no** | 0.102703 | 0.121190 |
| Montorie Foster | WR | WR5 | none | — | **no** | **NaN** | 0.165842 |
| AJ Barner | TE | TE1 | none | `te_target_mult` | yes | 0.136376 | 0.136722 |
| Nick Kallerup | TE | TE2 | none | `te_target_mult` | yes | 0.083333 | 0.117876 |
| Eric Saubert | TE | TE3 | none | `te_target_mult` | yes | 0.052747 | 0.075740 |
| Brady Russell | FB | FB1 | none | `rb_rec_target_mult` | yes | 0.034695 | 0.059130 |

**Cooper Kupp misses the WR1_5 label by 0.008206 of raw target share** and is
therefore structurally ineligible for every opponent-coverage target gate —
while Seattle's **third-string tight end** and its **fullback** are both
eligible, purely by position.

## The cutoff is discontinuous, and the key is inconsistent

**Discontinuity.** Rank 2 and rank 3 are separated by 0.008206 of target share
and by the entire difference between a potential ±15% coverage multiplier and
structurally none. Nothing interpolates.

**Key mismatch.** `_wr_role_labels()` sorts perimeter WRs by the **raw**
`tgt_share` with a `0.0` default, while the quantity production actually scales
is `bayes_tgt_share`. These disagree, and Montorie Foster is the clean case: his
raw `tgt_share` is `NaN → 0.0`, so he sorts last and receives no label, yet his
posterior of **0.165842** is the **second-largest in the room** — larger than the
labeled WR1_5 and larger than Cooper Kupp. The role ranking and the allocated
share are keyed on different quantities.

## Structural zero vs circumstantial zero

Opponent-attributable movement for every Seattle WR in Week 7 was **0.0**, but
for two different reasons that must not be pooled:

- **Kupp's zero is STRUCTURAL.** He is ineligible whatever the defense does.
- **JSN's zero is CIRCUMSTANTIAL.** He was eligible and the gate did not fire.

The man-coverage gate is **certified not fired** for Kansas City by an exact
argument: WR1 takes ×0.95 and WR1_5 takes ×1.15, so a fired gate would split
their `rules`/`bayes` ratios. Both are exactly **1.000**, so it did not fire.

## Role-specific efficiency signals, present and stranded

Each role has a matching opponent conversion signal built upstream and
referenced by **no production module at all**: `ypt_allowed_wr`,
`ypt_allowed_slot`, `ypt_allowed_te`, `ypt_allowed_rb`, `ypt_allowed_outside`.
So even a labeled WR has no role-specific conversion channel — only the shared
pressure-driven `pass_eff_mult`.

## Finite room / normalization

The `rules`/`bayes` ratio is uniform **within** each position group: WR 1.000,
RB/FB 1.200, TE 1.265. This is the per-group room normalization.

## Not established

- **Whether the TE and RB coverage/pressure gates fired.** A uniform
  within-group multiplier is indistinguishable from that group's normalization
  scale in the preserved artifact, and `rules_*_mult` was not preserved.
  Reported unresolved rather than guessed.
- **Shared `pass_eff_mult` movement** — `rules_ypt` was not preserved.
- **How often any gate fires league-wide.**
- **Whether consuming the stranded `ypt_allowed_*` family would improve any
  projection.** Untested; must not be scored on Week 7.

## Explicitly separate

Montorie Foster is `position_prior_only` (zero history) and carries a larger
`rules_tgt_share` than Cooper Kupp. That is the **zero-history entitlement
defect**, which the audit requires be kept a distinct cause. It is recorded here
as an observation only and is not merged into the routing finding.

## Method validity

`assert_routing_matches_source()` fails closed if the WR branches stop keying on
the role label, if the TE/RB branches stop keying on position, if any multiplier
leaves the chain, or if a third perimeter WR label appears — so this document
cannot go stale silently.
