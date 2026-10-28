# PRISM — `slot_target_mult` Is Unreachable Dead Code

**Status:** read-only source + artifact finding. **No fix proposed here, no
threshold change, no scoring.**
**Prompted by:** GPT artifact replay `6455253348`, which observed that the saved
`rules_role` field contains **zero SLOT-assigned WRs** and therefore that the
rate-scale patch's `slot_target_mult` changes do not propagate "in this frozen
slate."
**This finding is stronger than that:** it is **not** a property of one slate.
No slate can produce a SLOT label as the code currently stands.

## The mismatch

Two modules disagree about what `position` means, and neither is obviously
wrong on its own.

**Upstream deliberately erases alignment.** `scripts/run_player_form_v2.py`:

```python
def _position_family(value) -> str:
    pos = ...
    if pos in {"LWR", "RWR", "SWR", "WR", "WIDE RECEIVER"} or pos.startswith("WR"):
        return "WR"
...
# Consumers should reason about WR/RB/TE/QB family, not LWR/RWR alignment.
out["position"] = out["position_group"]
```

That is a documented design decision, stated in the code.

**The SLOT branch requires alignment.**
`scripts/modeling/simulation_rules._wr_role_labels()`:

```python
slots = [p for p in group
         if str(p.position).upper() == "SWR" or "SLOT" in str(p.role).upper()]
for p in slots:
    out[(team, _key(p.player))] = "SLOT"
```

Both of its conditions fail in production:

- `position == "SWR"` — impossible, because `position` is the family `WR` by then.
- `"SLOT" in role` — Ourlads depth roles are `WR1`/`WR2`/`WR3`
  (`ourlads_depth.WR_ROLE_BY_COL = {"LWR": "WR1", "SWR": "WR2", "RWR": "WR3"}`),
  and **0 of 458** roster rows contain the substring `SLOT`.

So `slot_target_mult` is computed by `matchup_multipliers()` and applied by an
`elif` branch no player can enter.

## Evidence

| | Week 7 (recovered artifacts) | Week 8 (GPT's artifact) |
|---|---|---|
| `rules_role` = SLOT | **0** of 396 | **0** of 404 |
| `rules_role` = WR1 | 28 | 28 |
| `rules_role` = WR1_5 | 28 | 28 |
| blank / other | 340 | 348 |

And the information does exist upstream — it is destroyed in transit:

| | roster (`roles_ourlads.csv`, 458 rows) | at the rule layer |
|---|---|---|
| `SWR` | **62** | **0** |
| `LWR` | 56 | 0 |
| `RWR` | 54 | 0 |
| `WR` | 0 | **145** |

Of the 62 roster-`SWR` players, **48 appear in the Week-7 diagnostics, all with
`position = 'WR'`**. Their assigned roles: **35 blank, 8 WR1_5, 5 WR1** — so slot
receivers are thrown into the perimeter pool and rank-gated against outside
receivers on `tgt_share`, which is the same rank-gating defect already recorded,
reached by a different route.

## Why this matters for the pending rate-scale fix

GPT's replay found the patch changes `slot_target_mult` on 217 Week-8 rows
(1.10 → 1.00). **None of those propagate**, because no row is SLOT-labelled —
and that holds for every slate, not just this one. The patch's only live target
path is the **45 affected TE rows**.

That does not weaken the case for the fix: a gate evaluating a percentage
against a fraction threshold is wrong regardless of how many rows it currently
reaches. But it bounds the expected effect correctly and removes a false
impression that 217 receiver rows would move.

## What this does NOT establish

- **That restoring a slot channel would improve any projection.** Untested, and
  the scoring window is closed.
- **Which module should change.** Upstream's family collapse is deliberate and
  defensible; the SLOT branch's expectation is also defensible in isolation. A
  fix could preserve alignment in a separate field, relabel via depth role, or
  retire the SLOT branch — all are design choices requiring authorization, and
  none is proposed here.
- **That slot receivers are mis-projected.** They are rank-gated like any other
  unlabeled WR; whether that is wrong is a modelling question, not a source
  fact.

## Method

`rules_role`, `position`, roster positions and role strings read from committed
artifacts; the two code paths read from production source. No outcome, market,
line or price was read.
