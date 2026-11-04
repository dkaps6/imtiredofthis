# PRISM — CODE AUDIT: rules / bayes / simulation LAYERS

**Date:** 2026-11-03
**Scope:** `scripts/modeling/simulation_rules.py`, `scripts/modeling/rules_v2.py`,
`scripts/modeling/bayesian_v2.py`, `scripts/simulation_v2.py`.
**Status:** AUDIT. No code changed. Each finding needs its own preregistration before any fix.
**Method:** read the live path, then verify each claim numerically or against the committed
production universes (W7+W8, 799 rows). Unverified suspicions are labelled as such.

---

## D1 — A player ruled OUT keeps 50% of his opportunity · **SEVERITY: HIGH**

```python
def _injury_limited(ctx) -> bool:
    return any(t in text for t in ("OUT","DOUBTFUL","IR","PUP"))
...
if _injury_limited(ctx) and (team,pkey) not in injury_overrides:
    base_tgt *= 0.50
    base_rush *= 0.50
# and for the alpha:
give = alpha_share * 0.50
overrides[alpha] = alpha_share - give      # keeps the other 50%
```

**A player on IR, or ruled OUT, retains half his target and rush share.** An OUT player takes
zero snaps. Worse, **OUT, DOUBTFUL, IR and PUP are all treated identically** — a doubtful player
(plays sometimes) and an IR player (cannot play) receive the same 0.50.

Two consequences, both measured elsewhere in this program: it **inflates team opportunity sums**
(see D2's 1.5074) and it **starves the actual replacement**, which is exactly the universe-
composition fault `PRISM_MC_OPPORTUNITY_LEAK_V1` identified and the participation defect
`PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §6 traced the alpha haircut to.

This also violates the **PLAYER-STATE ANOMALY GATE** in AGENTS.md, which requires
`DNP_INACTIVE` to be a distinct state — here it is collapsed into "limited".

## D2 — The matchup multipliers break the share constraint by construction · **SEVERITY: HIGH**

`rules_v2.matchup_multipliers` multiplies each player's target share **independently, with no
renormalization anywhere**:

| condition | effect |
|---|---|
| `zone ≥ 0.60` | `te_target_mult ×1.15`, `rb_rec_target_mult ×1.20` |
| `man ≥ 0.50` | **`wr1_target_mult ×0.95`**, `wr1_5 ×1.15`, `slot ×1.05` |
| `middle ≥ 0.50` | `slot ×1.10`, `te ×1.10` |
| `pressure > 0.075` | `rb_rec ×1.10` |

**Maximum achievable multiplier by role:**

| WR1 | WR1_5 | SLOT | TE | RB |
|---|---|---|---|---|
| **1.000** (floor 0.95) | 1.150 | 1.155 | **1.265** | **1.320** |

**WR1 is the only role whose multiplier cannot exceed 1.0.** Target share is zero-sum within a
team, so multiplying members independently and upward adds mass that exists nowhere. `zone ≥ 0.60`
is the common NFL case, so TE ×1.15 and RB ×1.20 fire on most teams.

**This predicts the measured pattern exactly** (`RULES_VS_BAYES_COVERAGE_AUDIT_V1` §1-2):

| position | % overridden | mean shift |
|---|---|---|
| WR | **0.0%** | — |
| QB | **0.0%** | — |
| TE | **100%** | **+0.0294** |
| RB | 94.4% | +0.0148 |

and the team sums: `bayes` **1.3756** → `rules` **1.5074**, against a true ceiling of 1.0.

**So this is a third independent compression source**, alongside the Bayesian group-mean
shrinkage (`ROLE_SHRINKAGE_FIX_RESULTS_V1`) and the allocator's downward-only normalizer
(`MC_OPPORTUNITY_LEAK_V1`). **All three push the same way: down on the alpha, up on the backups.**
Note the contrast: `_sharpen_wr_target_shares` and `_sharpen_rb_rush_shares` **do** renormalize —
the rules layer is the one place that does not.

## D3 — Zero-opportunity players still produce yards · **SEVERITY: MEDIUM-HIGH**

```python
rec_sd  = np.maximum(6.0, np.sqrt(np.maximum(targets,1))*ypt*0.55)*vol_mult
rec_yards = np.clip(rng.normal(rec_mu, rec_sd), 0.0, None)
```

The sd floor applies **regardless of opportunity**, and the clip at 0 is asymmetric, so:

| opportunity | mean | sd | **E[yards]** | bias |
|---|---|---|---|---|
| 0 targets | 0.0 | 6.0 | **2.40** | **+2.40** |
| 1 target | 7.5 | 6.0 | 7.81 | +0.31 |
| 2 targets | 15.0 | 6.0 | 15.01 | +0.01 |
| 0 carries | 0.0 | 3.0 | **1.19** | **+1.19** |

**A player with zero targets is projected 2.40 receiving yards; zero carries gives 1.19 rushing
yards.** Pure artifact of the floor plus the clip, and it lands **entirely on bench players** —
the population already over-projected by +8% to +30% (`ALPHA_COMPRESSION_BOARD_CHECK_V1` §2).

## D4 — Redistribution silently destroys mass · **SEVERITY: MEDIUM**

```python
"""...while conserving redistributed share."""
for weight, recipients in buckets:     # 0.60 WR1_5 / 0.30 SLOT+TE / 0.10 RB+FB
    if not recipients:
        continue                        # <-- that bucket's share simply vanishes
```

A team with no `WR1_5` loses **60% of the redistributed share** outright. **The docstring's stated
contract is violated**, and the loss is position-dependent, so it biases differently by team.

## D5 — Injury redistribution only fires for a perimeter WR1 · **SEVERITY: MEDIUM**

`_injury_target_overrides` searches only for `labels[...] == "WR1"`, and `_wr_role_labels`
**removes slot receivers from the perimeter ranking before assigning WR1.** So:

- an injured **slot** alpha triggers **no redistribution**;
- an injured **TE** alpha (a Kelce-shaped player) triggers **no redistribution**;
- an injured **RB** triggers none.

The vacancy rule only understands one shape of alpha.

## D6 — An injured redistribution recipient escapes his own haircut · **SEVERITY: LOW**

The `×0.50` haircut is gated on `(team,pkey) not in injury_overrides`. A player who is himself
injury-limited **and** receives redistributed share is in `injury_overrides`, so he is **boosted
and never haircut.** Narrow, but wrong in both directions at once.

## D7 — The circular WR ranking · **SEVERITY: MEDIUM (structural), LOW (current effect)**

Documented in `RULES_VS_BAYES_COVERAGE_AUDIT_V1` §3; preregistered in
`PRISM_WR_RANKING_SWAP_PREREGISTRATION_V1`. Agreement with the external depth lineage is 56/56,
so the present cost is small; the mechanism is unsound and the depth data sits unused.

---

## Things I checked that are NOT defects

Recording these so they are not re-audited:

- **`pass_eff_shock` is not double-counting.** It is `clip(normal(1.0, 0.09), 0.65, 1.35)` —
  **random game-level variation with mean 1.0**, drawn per game and shared across players, which
  is correct correlation modelling. The matchup term (`pass_eff_mult`) is applied separately in
  the rules layer. I suspected double application and was wrong.
- **`plays` and `pass_rate` clipping** (`[50,80]`, `[0.35,0.75]`) are reasonable football bounds.
- **The sharpening transforms renormalize**, as their docstrings claim.
- **`bayesian_v2` is leakage-safe** on the paths read: it consumes a pre-cutoff consensus and
  never reads results or lines.

## Severity-ordered remediation, and what must be measured first

| # | defect | why it ranks here |
|---|---|---|
| **1** | **D2** multipliers break the share constraint | largest measured mass effect (+0.13/team-week, 26% of the leak); structural; explains a measured pattern exactly |
| **2** | **D1** OUT keeps 50% | football-wrong, violates AGENTS.md's state gate, feeds the alpha haircut |
| **3** | **D3** phantom yards at zero opportunity | quantified, systematic, hits the already-over-projected tier |
| **4** | **D4/D5/D6** redistribution defects | real but narrower; D4 and D5 should be fixed together |
| **5** | **D7** circular ranking | already preregistered; do-no-harm only |

**Each needs its own preregistration.** D1 and D2 both *reduce* allocated mass, and the
`ROLE_SHRINKAGE` arms also reduce it — **so they are not independent, and fixing them together
without measuring the interaction risks over-correcting into under-allocation.** The measured
excess is finite (+0.5074 per team) and these four mechanisms are competing to remove the same
mass.

**One caveat that limits every number above:** the production-universe figures come from **799
rows over two weeks**. The positional pattern in D2 (0% vs 100%) is too stark to be noise, but
magnitudes need a wider trace. D3's figures are analytic and do not depend on sample.

No sportsbook variable was read. No production code was modified.
