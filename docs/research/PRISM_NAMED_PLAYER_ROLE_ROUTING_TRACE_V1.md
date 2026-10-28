# PRISM — Named-Player Role-Routing Trace V1 (JSN / Kupp / Barner)

**Status:** **CLEARED (read-only scientific scope)** by GPT final re-audit
`6446595260`, disposition
`NAMED_PLAYER_ROLE_ROUTING_V1_CLEARED_READ_ONLY__PREDICTIVE_AND_FINAL_MEAN_ATTRIBUTION_NOT_ESTABLISHED`,
at exact SHA `4f95723154d5473b362f33c450b6f2d5fd8f6b60`.

**Cleared:** the compact JSN/Kupp/Barner role-routing *mechanism*, the matching
pregame rule-layer multipliers, the conditional gate decode, the true archived
target-share allocator chain, and the documented stranded role-specific
efficiency signals.

**NOT cleared:** fully measured opponent attribution to final receiving-yard
means, predictive value of any new channel, league-wide gate prevalence, Stage-2
Week 7, and any fitting or promotion. No coefficient/threshold selection, no
outcome scoring, no sportsbook upstream, no paid OddsAPI, no Walker revisit.

**No fitting, no scoring, no promotion.**
**Revised per:** GPT audits `6445134492` and `6445801491`
(`NAMED_PLAYER_ROLE_ROUTING_MECHANISM_AND_ALLOCATOR_SUPPORTED__V2_FINAL_WORDING_AND_SIDECAR_CONSISTENCY_FIX_REQUIRED`).
**Earlier revision per:** GPT audit `6445134492`
(`NAMED_PLAYER_ROLE_ROUTING_CORE_SUPPORTED__FULL_OPPONENT_ATTRIBUTION_TRACE_REVISION_REQUIRED`)
— core role-routing finding SUPPORTED, three blockers fixed below.
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

The **direct WR coverage-target rule-layer delta** for every Seattle WR in Week 7
was **0.0**. That is a statement about one channel only: it is **not** total
opponent attribution, and it does **not** make any WR globally
opponent-insensitive. Shared `pass_eff_mult` can still move YPT, and the
M38/entitlement stages move a WR's share for teammate-structural reasons, which
is indirect opponent exposure. Within that scope there are two different causes
which must not be pooled:

- **Kupp's zero is STRUCTURAL.** He is ineligible whatever the defense does.
- **JSN's zero is CIRCUMSTANTIAL.** He was eligible and the gate did not fire.

The man-coverage gate is **certified not fired** for Kansas City by an exact
argument: WR1 takes ×0.95 and WR1_5 takes ×1.15, so a fired gate would split
their `rules`/`bayes` ratios. Both are exactly **1.000**, so it did not fire.
But the **zone** gate did fire, and no WR role reads zone at all — see above.

## Role-specific efficiency signals, present and stranded

Each role has a matching opponent conversion signal built upstream and
referenced by **no production module at all**: `ypt_allowed_wr`,
`ypt_allowed_slot`, `ypt_allowed_te`, `ypt_allowed_rb`, `ypt_allowed_outside`.
So even a labeled WR has no role-specific conversion channel — only the shared
pressure-driven `pass_eff_mult`.

## The rule-layer ratio is the multiplier, not a normalization

`apply_rules_to_metrics()` sets `base_tgt` from the posterior, applies the
matchup multiplier and any injury limitation, and emits `rules_tgt_share`. It
performs **no team-room normalization** — V1 of this document called the
observed ratios a "per-position-group room normalization," which was wrong.

Because no normalization intervenes, **the ratio IS the multiplier.** That makes
the gate state recoverable:

| role multiplier | achievable products | observed | decoded |
|---|---|---|---|
| `te_target_mult` | 1.000, 1.100, 1.150, **1.265** | **1.265** | zone **and** middle-open fired |
| `rb_rec_target_mult` | 1.000, 1.100, **1.200**, 1.320 | **1.200** | zone fired, pressure did **not** |
| `wr1_target_mult` | 0.950, **1.000** | **1.000** | man did **not** fire |
| `wr1_5_target_mult` | **1.000**, 1.150 | **1.000** | man did **not** fire |

Every role's products are distinct, so an exact match identifies the state
uniquely. **This is a source-conditioned inverse reconstruction, not a reading
of preserved `coverage_*_rate` values** — those were not preserved. The shared booleans are decoded **independently by separate position
groups** — `coverage_zone_rate` by both TE and RB, `coverage_man_rate` by both
WR1 and WR1_5 — and they agree, which is corroboration rather than a
single-group inference.

**Decoded gate state for the Kansas City defense, 2026 Week 7:**

| gate | state |
|---|---|
| `coverage_zone_rate >= 0.60` | **FIRED** |
| `middle_open_rate >= 0.50` | **FIRED** |
| `coverage_man_rate >= 0.50` | not fired |
| `pressure_mismatch > 0.075` | not fired |

**Residual condition, stated rather than assumed away:** injury redistribution
replaces `base_tgt` outright and injury limitation halves it, so either would
break the exact match. The decode holds for rows carrying neither, which the
uniform exact ratios within each group evidence but do not prove.

**Derived consequences.** Pressure did not fire, so `volatility_mult = 1.0` —
the explicit sigma channel did not move. And `pass_eff_mult` is **not** 0.98; it
is 1.00, or 1.01 if pressure `< -0.075`. Field 8 is therefore **bounded to
{1.00, 1.01}, not resolved**: `apply_rules_to_metrics` does emit
`rules_pass_eff_mult`, but it was not preserved.

## No WR role has a zone route at all

This is the finding V1 obscured. `wr1_target_mult` and `wr1_5_target_mult` key
**only** on `coverage_man_rate`; `slot_target_mult` adds `middle_open_rate`. None
of them reads `coverage_zone_rate`. Only `te_target_mult` and
`rb_rec_target_mult` do.

In Week 7 Kansas City's zone gate **fired**. So zone coverage — the most common
coverage family in the league — moved every Seattle tight end by **×1.265** and
every running back and the fullback by **×1.200**, and could not move a single
wide receiver. That is an **architectural** asymmetry between the WR route and
the TE/RB routes.

**It is not a Kupp-specific penalty, and an earlier version of this document
wrongly implied it was.** Both labeled WR multipliers were **1.000** because the
man gate did not fire. Had Kupp been labeled WR1 or WR1_5 in this same frozen
gate state, his direct coverage-target multiplier would **also** have been
1.000. His unlabeled status is therefore **structural eligibility exposure with
zero realized direct coverage effect in this matchup** — a missing potential
channel, not an additional numeric penalty. The claim that his exclusion was
"not academic this week" is withdrawn.

## Fields 7 and 10 — the real allocator chain

Not inferred from a ratio. The preserved chain is
`rules_tgt_share → post_m38_tgt_share → entitlement_tgt_share` plus a residual.

| player | role | `rules` | `post_m38` | M38 Δ | `entitlement` | net `rules`→`entitlement` |
|---|---|---|---|---|---|---|
| Jaxon Smith-Njigba | WR1 | 0.316891 | **0.409603** | **+29.3%** | 0.243027 | ×0.767 |
| AJ Barner | TE (position) | 0.172954 | 0.172954 | **0.0%** | 0.135849 | ×0.785 |
| **Cooper Kupp** | **unlabeled** | 0.151325 | **0.108976** | **−28.0%** | 0.106951 | **×0.707** |
| Rashid Shaheed | WR1_5 | 0.157403 | 0.132245 | −16.0% | 0.079959 | ×0.508 |
| Tory Horton | unlabeled | 0.121190 | 0.087275 | −28.0% | 0.059026 | ×0.487 |
| Montorie Foster | unlabeled | 0.165842 | 0.174552 | +5.3% | 0.052536 | ×0.317 |

**The WR hierarchy stage touches only WRs** — every TE and RB value is identical
pre- and post-M38 — and it moves share away from the non-top receivers toward
JSN.

**This is NOT an opponent effect and must not be attributed to the zone gate.**
The sharpening stage is keyed on WR rank, not on the defense. Kupp's **−28.0% is
the only realized reduction** in this matchup; his coverage-route exclusion
produced nothing numerically here, because the labeled WRs also received 1.000.

Both consequences do follow from his rank — rank decides coverage eligibility
*and* sharpening position — but only one of them is an opponent channel, and
that one was inert this week. An earlier version said "reduced twice over",
which conflated a missing potential channel with an actual adjustment.

Team mass is conserved exactly: `rules` and `post_m38` both sum to
**1.601148605998**, and the entitlement room closes at **0.950000** with a
**0.05** residual and an `entitlement_gap` of **5.55e-16** across 15 players.

## Not established

- **`pass_eff_mult` exactly** — bounded to {1.00, 1.01}; `rules_pass_eff_mult`
  is emitted by production but was not preserved.
- **The decode's residual condition** — that no traced row carried injury
  redistribution or limitation is evidenced by the exact uniform ratios, not
  proven.
- **Total opponent attribution to the final mean.** The chain above is the
  target-share path; YPT, game script and downstream mean construction are not
  accounted for here.
- **How often any gate fires league-wide.**
- **CI verification, precisely.** `audit-only.yml` run
  [`40530603310`](https://github.com/dkaps6/imtiredofthis/actions/runs/40530603310)
  is **green on this exact SHA**, so compileall and the strict repo audit **are**
  independently verified on the runner. That workflow does **not** run pytest,
  and `repo-ci.yml` triggers only on `pull_request` and pushes to `main`, so the
  quoted `803 passed, 1 pre-existing failure` remains an **executor-reported
  local result**, not independently CI-proven. No PR was opened.
- **Whether consuming the stranded `ypt_allowed_*` family would improve any
  projection.** Untested; must not be scored on Week 7.

## Explicitly separate

Montorie Foster is `position_prior_only` (zero history) and carries a larger
`rules_tgt_share` than Cooper Kupp. That is the **zero-history entitlement
defect**, which the audit requires be kept a distinct cause. It is recorded here
as an observation only and is not merged into the routing finding.

## Method validity

`assert_gate_specs_match_source()` **derives** the gate table by parsing
`rules_v2.matchup_multipliers` — the `_num(defense.attr)` bindings, each
`if var OP threshold:` block and the `values["mult"] *= factor` lines inside it
— and raises if the hand-authored `GATE_SPECS` mirror disagrees semantically
(thresholds compared as numbers, so `0.60` and `0.6` are the same gate). Audit
`6445801491` correctly objected that the zone-route test checked the mirror
rather than production; it now checks a mirror that is itself verified against
production. The derivation independently recovers all ten multipliers, including
`rb_rush_eff_mult`'s light/heavy-box gates.

The status sidecar's `not_established` list is **generated from** the gate
decode rather than written beside it, and `_self_consistency()` raises before
the file is written if the two disagree. Audit `6445801491` found the committed
V2 sidecar calling the TE/RB gates unknown and `pass_eff_mult` unbounded while
`gate_decode` had already resolved both — that contradiction can no longer be
committed.

`assert_routing_matches_source()` **raises** if the WR branches stop keying on
the role label, if the TE/RB branches stop keying on position, if any multiplier
leaves the chain, if an `else` arm appears after the RB/FB branch (which would
end the unlabeled-WR fall-through), or if `_wr_role_labels` assigns anything
other than exactly `{SLOT, WR1, WR1_5}` or indexes beyond `perim[0..1]`.

V1 recorded the fall-through check as a boolean **without raising**, and proved
the two-label claim only by the absence of a `perim[2]` string — both of which
audit `6445134492` correctly rejected as non-guards. The check now parses the
assignment statements themselves.
