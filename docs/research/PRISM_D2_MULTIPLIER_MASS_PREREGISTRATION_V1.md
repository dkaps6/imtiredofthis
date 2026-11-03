# PRISM — D2: MATCHUP-MULTIPLIER MASS VIOLATION, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Source:** `PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1.md` D2.

---

## 0. The defect, and the live multipliers recovered from production

`simulation_rules.apply_rules_to_metrics` multiplies each player's target share independently:

```python
if   role == "WR1":   tgt_mult *= mods.wr1_target_mult
elif role == "WR1_5": tgt_mult *= mods.wr1_5_target_mult
elif role == "SLOT":  tgt_mult *= mods.slot_target_mult
elif pos  == "TE":    tgt_mult *= mods.te_target_mult
elif pos  in {"RB","FB"}: tgt_mult *= mods.rb_rec_target_mult
base_tgt *= tgt_mult          # <-- no renormalisation, anywhere
```

Target share is **zero-sum within a team**. Multiplying members independently and upward creates
mass that exists nowhere. Recovering `rules_tgt_share / bayes_tgt_share` from the committed W7+W8
universes gives the **multipliers actually used in production** — exactly the `rules_v2` table
values, so this is not an inference:

| position | n | mean multiplier | distinct values observed |
|---|---|---|---|
| **WR** | 301 | **1.0000** | {1.0} |
| QB | 148 | 1.0000 | {1.0} |
| **TE** | 165 | **1.2560** | {1.1, 1.265} |
| **RB** | 162 | **1.1956** | {1.0, 1.2, 1.32} |
| FB | 23 | 1.1913 | {1.0, 1.2} |

**No WR received any matchup adjustment in either week.** Team `tgt_share` sums:
**bayes 1.3756 → rules 1.5074**, against a true ceiling of 1.0 and a realized team total of
≈0.94. **Both layers are infeasible in 100% of team-weeks; the multipliers make it worse.**

Also recorded: no recovered multiplier is near 0.50, so **D1's injury haircut did not fire on any
row in W7 or W8.** That bounds D1's *measured* footprint here without bearing on its severity
when it does fire.

## 1. Research question

**Does making the matchup multipliers redistributive rather than inflationary improve
target-share projections — and do the multipliers carry any information at all once mass is
controlled?**

## 2. Arms — a 2×2 factorial, every arm exactly recoverable

| arm | multipliers | normalisation |
|---|---|---|
| **C0** production | ON | none |
| **M1 — primary** | ON | back to the **pre-multiplier team sum** (mass-preserving) |
| **M2** | OFF (= `bayes`) | none |
| **M3** | ON | team sum → **1.0** |
| **M4** | OFF | team sum → **1.0** |

**M1 is the primary arm** and the conservative fix: it removes only the inflation the multipliers
added, leaving the overall level untouched. It is exactly the contract
`_sharpen_wr_target_shares` already states (`sharpened *= total/sharpened.sum()`).

**M1 vs M2 is the information test.** If they tie, the multipliers carry **no** signal once mass
is controlled and the simplest correct action is to **delete them**. If M1 wins, they carry real
matchup information and only needed normalising. **M3 vs M4 repeats that test under feasibility.**

## 3. Gates

**Mass is mechanical here, not statistical** — each arm's team sum is determined by construction,
so there is no threshold for me to mis-set this time (the V2 §5 failure mode is structurally
excluded):

- **G1 (mass, mechanical)** M1's team sum must equal the pre-multiplier (`bayes`) sum to **1e-12**
  in every team-week; M3 and M4 must equal **1.0** to 1e-12. **Failure voids the run.**
- **G2 (identity, mechanical)** on rows where the recovered multiplier is exactly 1.0 for the
  whole team, M1 must reproduce C0 to 1e-12. **Failure voids the run.**
- **G3 (projection — primary)** M1 beats C0 on per-player bias against realized target share,
  95% CI excluding 0, bootstrapped **over players**.
- **G4 (no tier harmed)** no pregame tier degrades by more than 0.002 — **and the top tier is
  reported first**, because that is where the alpha haircut lives.
- **G5 (side, DESCRIPTIVE ONLY)** side accuracy on W7+W8 board rows, reported **with its minimum
  detectable effect beside it. This gate cannot be passed or failed** and **no promotion may rest
  on it** — the sample is ~800 rows and cannot resolve a 1pp effect.

## 4. Kill conditions

**K1** G1 or G2 breached → **run void** · **K2** G3 fails → the inflation is not hurting
projections, which would be surprising and must be published · **K3** G4 breached — name the
tier · **K4** M1 ≈ M2 within CI → **the multipliers carry no information; recommend deleting
them** rather than normalising them · **K5** M3/M4 beat M1 but harm the top tier → the
interaction warned about in the audit is real and **no arm may be promoted without a joint
study.**

## 5. The interaction risk — the reason M1 is primary

`PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1` §remediation warns that **D1, D2 and the `ROLE_SHRINKAGE`
arms all remove the same finite mass** (+0.5074 per team). M3 and M4 scale every share down by
≈27%, **including the alpha's — who is already under-projected by −0.0392**
(`IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §5).

**So M3/M4 may fix the aggregate while making the one unsolved tier worse.** That is why the
primary arm is M1, which does not touch the level at all, and why **K5 blocks promotion of any
feasibility-enforcing arm on this study alone.**

## 6. Prediction

**M1 beats C0** (removing +0.13/team-week of phantom mass concentrated on TE/RB should help those
positions and, via renormalisation, lift WRs slightly). **M1 ≈ M2**, i.e. the multipliers carry
little information — they are threshold-triggered step functions on team-level coverage rates,
applied identically to every TE on a team regardless of that player, which is the
position-over-player pattern this whole program keeps finding. **M3/M4 improve aggregate bias and
harm the top tier (K5).**

**Named risk:** M1 raises WR shares via renormalisation, and WRs are the one group with **no**
multiplier — so if WR shares are already too high in this universe, M1 could worsen them while
fixing TE/RB. The net could be flat for the wrong reasons, and the per-position breakdown must
therefore be reported, not just the pooled figure.

My record: **one hit, five misses, with the named risk landing every time.** This prediction
carries no evidentiary weight.

## 7. Scope

`tgt_share` only — `rush_share` receives no matchup multiplier (`rules_rush_share` was a 0.0%
override in the coverage audit). No sportsbook variable enters any arm; market lines grade only.
No production code is modified by this study. D1 and D3 are **explicitly out of scope** and must
not be bundled in.
