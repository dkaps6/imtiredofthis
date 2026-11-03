# PRISM — THE MONTE CARLO OPPORTUNITY LEAK V1

**Date:** 2026-11-03
**Status:** DIAGNOSIS, code-level and quantified. **No fix proposed, nothing promoted.**
Any correction requires its own preregistration.
**Follows:** `PRISM_RECEIVING_MASS_RESULTS_V2.md`, which localised the receiving bias to
`mc_proj` (−20.2% team reception mass) after rejecting prior weighting, ensemble reweighting
and post-hoc offsets.

---

## 1. The mechanism, in one table

W7 production universe (`football_simulation_universe`, 396 rows, 28 teams), matched to actual
Week 7 targets:

| | value |
|---|---|
| universe players per team | **14.1** |
| players per team who actually drew ≥1 target | **7.0 (49%)** |
| allocated share reaching players who *were* targeted | **0.6473** |
| **allocated share reaching players with ZERO targets** | **0.3027** |
| total allocated (the 0.95 cap) | 0.9500 |
| **shortfall to the players who actually played** | **−29.3%** |

**Roughly 30% of every team's modelled target mass is allocated to players who record no
targets at all.**

---

## 2. Why it happens — `scripts/simulation_v2.py`

```python
# line 188
targets = _allocate_counts(rng, pass_att, target_shares)

# lines 93-98, _allocate_counts
clean = np.clip(shares, 0.0, 0.95)
total_share = clean.sum()
if total_share > 0.95: clean *= 0.95 / total_share   # scales DOWN only
residual = max(0.0, 1.0 - clean.sum())
allocations = rng.multinomial(total, probs)[:n_players]
```

Two independent faults compound:

**(a) Shares carry no team-sum constraint.** Measured on the W7 universe, per-team
`tgt_share` sums to **1.5097** against a true matched-player total of **0.9436**. Each estimate
is individually **+25.5%** high. All 28 teams exceed the cap, so all 28 are divided by
1.5097/0.95 = **1.589**.

**(b) The universe is ~2× the set that will be targeted.** The division in (a) is taken over
*all* 14.1 modelled players, including the 7.1 who will see nothing. Their share of the pool is
not redistributed — it is simply allocated to them and realised as zeros.

The two faults do not cancel. Over-estimating each player by +25.5% and then dividing the pool
by 1.589 lands every genuine receiver at **−20.5%**, which matches the independently measured
`mc_proj` reception bias of **−20.2%**.

---

## 3. The code asymmetry that explains why rushing differs

Line 187, the two paths side by side:

```python
target_shares = _sharpen_wr_target_shares(team_df, raw_target_shares)
rush_shares   = _sharpen_rb_rush_shares(team_df, _top_n_shares(raw_rush_shares, 5))
```

**Rushing passes through `_top_n_shares(..., 5)`, which zeroes everyone outside the team's top
five. Receiving has no equivalent filter.**

That filter is why the RB lane measured carries at 19.77 vs 23.15 (**−14.6%**) while receiving
lands at **−29.3%**: rushing partially mitigates the same leak by discarding the tail before
normalising; receiving carries its full tail into the pool.

Both lanes have been chasing one defect from opposite ends. `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`
found rush shares summing to 1.2729 against the same 0.95 cap with a 0.7574 haircut. **Same
function, same missing constraint, different mitigation.**

---

## 4. This also closes the loop on the roster lane

V20/V21 **AUDIT TARGET 6** asked whether the Ourlads existence ceiling contaminates WR/TE
target shares, and recorded it as *"a diagnostic question only; do not assume the answer."*

It does, and the channel is now specific: the universe admits players on depth-chart presence
rather than expected availability, and `_allocate_counts` hands them opportunity mass
proportional to their estimated share. **A player who will not play does not merely project
zero — he removes ~30% of his team's opportunity from the players who will.**

This is the quantified version of the existence question, measured rather than assumed.

---

## 5. What is established, and what is not

**Established:**
1. 30.27% of allocated team target mass goes to players with zero actual targets (W7, 28 teams).
2. Per-team `tgt_share` sums to 1.5097 against a true 0.9436 — no team-sum constraint exists.
3. The resulting per-player allocation is −20.5%, matching the measured `mc_proj` bias of −20.2%.
4. Rushing applies `_top_n_shares(..., 5)`; receiving applies no equivalent filter.

**NOT established:**
- That any particular fix works. **Nothing here is a proposed change.** Three candidate
  directions exist (availability-weighted universe, a team-sum constraint on shares, a receiving
  analogue of the top-N filter) and all three are untested.
- That `_top_n_shares` is the *right* mitigation for rushing — it is merely why rushing leaks
  less. Truncating a tail is not the same as modelling it.
- That this generalises beyond W7. One week, 28 teams, one preserved trace. The W7 trace is the
  only committed production universe; weeks 3–5 are permanently lost (AGENTS.md §5a).
- Any claim about downstream pricing. The leak is measured at the allocation step only.

---

## 6. Next

The next step is **not** a fix. It is to confirm the leak reproduces on a second week before
anything is built on it — the W8 board exists and a universe trace could be reconstructed.

Then preregister, with the candidate directions as arms and the **player-level** gates that
caught D0 in the receiving-mass study: an availability-aware allocation that improves pooled
mass while harming a third of named players is not an improvement.

Note for whoever picks this up: three separate hypotheses were tested and rejected before
reaching this (prior weighting, ensemble reweighting, additive offset). All three were attempts
to correct this leak from downstream of it.
