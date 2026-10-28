# PRISM — RB carry-concentration sharpening: RESULTS (V1)

Preregistration: `PRISM_RB_CARRY_CONCENTRATION_PREREGISTRATION_V1.md` (frozen `430be98e`).
Parent diagnosis: `PRISM_RB_RUSHING_LEVEL_DIAGNOSIS_V1.md`.

## Headline

**K1 fired on the design as frozen, and the design is dead as written.** But the work it
forced produced the real, correctly-specified finding: **production's RB rush-share
vector is flat by 11 percentage points against a strictly-prior football target**, and
the direction of the error is *over-shrinkage*, not under-shrinkage.

## 1. K1: γ on raw trailing shares is 0.76, not > 1

Fitted on nflverse 2013–2025, 17,520 rows / 6,296 team-weeks, ranked by a pregame
trailing-share proxy:

| quantity | value |
|---|---|
| γ pooled | **0.76** |
| SSE improvement vs γ=1 | 5.27% |
| walk-forward γ by origin | 0.83, 0.80, 0.79, 0.76, 0.77, 0.76, 0.77, 0.76, 0.76, 0.76, 0.76, 0.76 |
| stability | min 0.76, max 0.83, spread 0.07, **sd 0.022** |
| top-1-by-proxy share | proxy 0.6973 → actual **0.6478** |

γ = 0.76 < 1, so **K1 triggered**: raw trailing shares are *more* concentrated than
reality, not flatter. Per the frozen rules, that design ships nothing.

The fit is unusually stable (sd 0.022 across twelve walk-forward origins), so this is a
well-powered result, not a noisy one. What it measures, though, is **mean reversion** —
the top back's trailing share overstates his next-game share by ~5pp. Production already
applies shrinkage of exactly this kind through its promoted empirical-Bayes baselines, so
γ=0.76 describes the raw proxy, **not** the residual flatness of production's shares.
The preregistration disclosed this as the design's main risk; it is what happened.

## 2. A retraction, made before it reached a conclusion

An intermediate measurement put production's top-back within-RB share at **0.2226** —
near-uniform — and attributed it to a spectacular bug: `metrics_ready.csv` carries one
row per *(player, market)*, so Kenneth Walker III appears 10 times, and
`_top_n_shares(raw, 5)` would keep five duplicate rows of the same back while
`_allocate_counts`'s 0.95 cap split his share across his own duplicates (0.95/5 = 0.19 ≈
0.2061 measured).

**That is not the production path and the number is withdrawn.** Production calls
`_full_roster_simulate` → `_build_full_universe`, which builds the universe from
`player_form_consensus.csv` — not from `metrics_ready` — and explicitly refuses
duplicates:

```python
if form.duplicated(["team", "player_clean_key"]).any():
    raise RuntimeError(f"football simulation universe duplicate player/team identities: {sample}")
```

The preserved universe source has 396 rows, 28 teams, **0 duplicates**, 93 unique RBs.
The duplicate-row bug does not exist in production. This is the second off-production-path
claim caught in this lane (the first was the rate-scale guard), and the lesson is the same
one: **follow the import graph to what production actually calls before measuring.**

## 3. The real finding: production over-shrinks by 11pp

Measured on the deduped W7 universe using `rules_rush_share` — the column `simulate()`
resolves first — against the strictly-prior football target, controlling for the number
of backs:

| n_backs | prod n | production top-1 | football n | football top-1 | gap |
|---|---|---|---|---|---|
| 2 | 5 | 0.6064 | 2,391 | 0.7058 | −0.0994 |
| 3 | 14 | 0.5334 | 2,975 | 0.6277 | −0.0943 |
| 4 | 6 | 0.4056 | 838 | 0.5625 | −0.1569 |

**Size-standardized: production 0.5173 vs football 0.6277, gap −0.1103, 95% CI
[−0.1453, −0.0741]** (paired-by-cell bootstrap, 4,000 draws).

The gap excludes zero, is present in **every** back-count cell, and survives controlling
for subset size (production averages 3.04 modeled backs, football 2.78). Production is
flat by ~11pp regardless of how many backs a team carries.

**Both results are consistent and together they localize the defect precisely:** raw
trailing share is over-concentrated (0.6973, needs γ=0.76 to reach 0.6478); production's
post-Bayes/post-rules share is under-concentrated (0.5173 against 0.6277). **The
shrinkage layers overshoot the football target.** That is a sharper statement than
"the distribution is too flat," and it names where to look.

## 4. Prize sizing — and why a mass-preserving fix is not sufficient

γ = **1.75** applied to production's share vector reproduces the prior-season target
(0.6283 vs 0.6277, residual 0.0006). Its effect on the top back, at 22.59 team RB carries
and 4.093 YPC:

| | share | carries | yards |
|---|---|---|---|
| today | 0.5173 | 11.7 | 47.8 |
| at γ=1.75 | 0.6283 | 14.2 | 58.1 |

**+10.3 yards to the top back, against an observed rank-1 gap of −22.34** — roughly 46%
of it. Adding the two other measured defects (YPC 4.093 vs 4.252 ≈ +2.3 yd; team volume
−0.52 carries ≈ +2.1 yd) reaches ≈ +14.7 of 22.3, leaving ≈ **−7.6 yd unexplained**.

And the transform is mass-preserving by construction, so **backs 2+ lose exactly the
10.3 yards the top back gains.** Rank-2 bias is already −5.21 [−10.62, +0.21]; spreading
a −10.3 loss over ~2 non-top backs drives it to roughly −9, near the edge of its
preregistered interval.

**That is K4 as frozen** — "improves rank 1 only by degrading rank 2 beyond its CI". The
transform is correctly *targeted* (it matches a strictly-prior football fact) but it is
not, alone, a fix for this market, because **every rank is biased low** and
mass-preservation cannot raise a level.

## 5. What this means for the two candidate fixes

The diagnosis framed A and B as alternatives. They are not — they address different
halves of the defect:

- **A (concentration)** fixes *allocation*: who gets the carries. Real, significant,
  prior-season-anchored, worth ~+10 yd on featured backs. Cannot change the level.
- **B (ensemble weight toward ML)** fixes *level*: `ml_proj` bias is **+0.79** against
  `mc_proj`'s −16.15, so it already has the level right for every rank at once.

**A without B makes second backs worse. B without A leaves the allocation wrong.** The
honest recommendation is both, in that order of confidence, each separately preregistered
and gated — not A alone.

## 6. Disposition

- the γ=0.76 proxy fit is **banked as a null** for the frozen design (K1);
- the 0.2226 production figure is **retracted** (off the production path);
- the 11pp over-shrinkage gap is **banked as a finding**, with its CI;
- the transform is implemented **INERT** behind dual promotion authority, with γ held at
  the neutral value. It changes no live projection. Promotion requires the prospective
  test and a resolution of the K4 interaction, neither of which exists yet.

No sportsbook variable entered the transform, γ, the target, or the calibration. Lines
were the comparison benchmark only. Betting remains **CLOSED**. No odds fetch performed.
