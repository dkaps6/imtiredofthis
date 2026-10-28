# PRISM — RB carry-concentration sharpening: PREREGISTRATION (V1)

**Frozen before γ was computed and before any outcome was scored.**

Follow-on to `PRISM_RB_RUSHING_LEVEL_DIAGNOSIS_V1.md`, which localized the RB
`rush_yards` defect to the Monte Carlo **level** (`mc_proj` bias −16.15, `ml_proj`
+0.79) and nominated two fixes. This preregisters **fix A** — the root-cause fix.

## 0. What changed in the mechanism before freezing

The diagnosis proposed sharpening "the carry distribution," citing lead share
model 0.558 vs actual 0.606. Two checks run before this freeze **reshaped** that
design and are recorded here because they constrain it:

**(a) The diagnosis's own magnitude did not add up.** Under its stated mechanism the
lead back's yards are model `22.59 × 0.558 × 4.093 = 51.6` vs actual
`23.11 × 0.606 × 4.252 = 59.6` — a **−8.0** gap, barely half the −16.15 observed. A
mass-preserving fix is bounded by the mass it can move, so the stated mechanism was
too small to be the whole explanation.

**(b) There is zero headroom inside the priced rows.** Bias by within-team model rank,
n=182 non-QB `rush_yards` rows with actuals, weeks 1/2/6/7:

| model rank | n | `mc_proj` | actual | bias | 95% CI |
|---|---|---|---|---|---|
| 1 | 107 | 39.31 | 61.65 | **−22.34** | [−29.04, −15.64] |
| 2 | 68 | 22.53 | 27.74 | −5.21 | [−10.62, +0.21] |
| 3+ | 7 | 11.16 | 16.29 | −5.13 | [−18.03, +7.78] |

**Every rank is biased low.** Taking mass from rank 2 to feed rank 1 would make rank 2
worse and cannot raise the level, because the transform conserves mass by construction.

**Where the mass actually is.** Per team-week, over priced RB rows only:
model **54.4** vs actual **80.3**, gap **−26.0** [−34.2, −17.8], n=107 team-weeks. If
the fixture's team RB totals transfer (92.5 model, 98.3 actual), the implied unpriced
remainder is model 38.1 vs actual 17.9 — the model hands **≈20 yd/team to backs reality
did not feed** — and the decomposition closes: −26.0 + 20.2 = −5.8 = the team-total gap.

So the operative mechanism is **leak into the deep-back tail (ranks 2–5)**, not
misallocation between rank 1 and rank 2. The headroom is ≈20 yd/team and it lives in
players who mostly do not carry a prop.

**This closure is partly inferential and is not claimed as established.** The 92.5/98.3
team totals come from the W7 fixture (26–28 teams); the −26.0 priced gap comes from the
107 board team-weeks. Those are **different populations**, and because the unpriced
remainder is derived *by subtraction* from the fixture totals, the consistency check is
partly circular. The independently measured, selection-immune quantity is the −26.0
priced-row gap. Mixing these two populations is the exact error that produced three
retracted claims in the diagnosis, and it is flagged rather than relied upon.

## 1. Hypothesis

**H1.** The simulation's RB rush-share vector is too **flat**. A one-parameter monotone
concentration transform, mass-preserving over the team's RB subset, moves opportunity
from the deep-back tail to featured backs and reduces the signed low bias of
`mc_proj` on RB `rush_yards` **without** changing team rush volume.

**H0.** The share vector's concentration already matches football (γ ≈ 1), or correcting
it does not reduce the bias.

## 2. Functional form — frozen

```
sharpened_i ∝ share_i ** γ      for i in the team's RB subset
sharpened  *= total / sharpened.sum()    # restore the original RB share mass exactly
```

- **One free parameter, γ.** γ = 1 is the identity; γ > 1 concentrates.
- **Mass-preserving by construction.** The RB subset's share total is restored exactly,
  so simulated team RB carry volume — already correct at bias −0.52 — is unchanged.
  This is a hard requirement, not a tuning choice.
- **Monotone and identity-preserving.** Ordering is preserved and relative magnitude
  differences are retained, so a back with genuinely higher pregame share stays higher.
  This is why a power transform was chosen over a **rank lookup table**: a lookup would
  overwrite every team's distribution with one positional curve, which is precisely the
  "same rule for every player" failure mode this project rejects. γ corrects the
  *concentration level* while leaving player identity intact.
- **RB subset only.** `{"RB","HB","TB","FB"}` plus any label starting with `RB`, matching
  `run_pricing_with_rb_conservation_v1.py:34-45`. QB/WR/TE rush shares are untouched.
- **Insertion point:** immediately after `_top_n_shares(raw_rush_shares, 5)` and before
  `_allocate_counts`, mirroring where `_sharpen_wr_target_shares` sits on the target path.
- **Guards:** no-op when the RB subset has ≤1 member, when its share total is ≤0, or when
  γ is outside a hard-coded sane band. γ is clamped; the clamp is a safety rail, not a fit.

## 3. Calibration — frozen

γ is fitted on **nflverse weekly player stats, 2013–2025 only**. 2026 supplies no
calibration input of any kind.

Because the model's historical share vectors do not exist on disk (they were never
archived — the AGENTS.md §5a failure), γ is fitted against a **trailing-share proxy**:
each back's share of his team's RB carries over prior weeks, which is what the
production shares are built from. Procedure per (season, week, team):

1. compute each RB's pregame trailing share of team RB carries from **prior weeks only**;
2. take the actual share of team RB carries in the scored week;
3. fit the single γ minimizing squared error between the renormalized `proxy**γ` vector
   and the actual share vector, pooled across all team-weeks.

**Ranking is by the pregame proxy, never by the outcome.** Ranking on realized carries
would manufacture concentration through max-selection — the artifact that produced three
retracted claims in the diagnosis.

**Walk-forward stability is reported, not optional:** γ refitted for each season S on
`[2013, S−1]` only. A γ that swings materially across origins is evidence of instability
and triggers kill condition K3.

**Disclosed limitation:** the proxy is not the model's true share vector. γ may therefore
be mis-scaled relative to production's actual flatness. This is the main reason the
change ships **inert** and requires prospective confirmation.

## 4. Metrics — frozen

**Primary:** signed bias of `mc_proj` on RB `rush_yards`, reported **by within-team model
rank** (1, 2, 3+) as well as pooled, plus excess absolute error versus the market line on
identical rows.

**Averaged MAE is explicitly NOT the primary metric** and no claim will rest on it, per
the standing instruction that a position-average metric cannot settle an
individual-player distribution question. Pooled MAE is reported as a guardrail only —
to catch a fix that improves bias while degrading accuracy.

**Mass conservation is a pass/fail assertion, not a metric:** simulated team RB carry
total must be unchanged to numerical tolerance.

## 5. Decision gate and promotion — frozen

The change ships **INERT** behind dual promotion authority (an in-code flag and an
artifact state that must agree), exactly as `mc_dispersion_calibration_v1` does. γ is
wired and computed but **does not alter a live projection** until promoted.

**Weeks 1–7 of 2026 are not a clean confirmatory holdout** for a concentration claim: the
gap was observed on those rows, so scoring there is descriptive, not confirmatory. It is
reported as a consistency check and labelled as such.

**The confirmatory test is prospective — week 8 onward**, scored after the fact against
the frozen γ. Promotion requires:

- **G1** rank-1 signed bias moves toward zero, and rank-2 bias does not deteriorate past
  its preregistered CI;
- **G2** team RB carry mass conserved to tolerance (hard assertion);
- **G3** pooled MAE not worse;
- **G4** walk-forward γ stable across origins;
- **G5** no sportsbook variable enters the transform or γ (AGENTS.md §3).

## 6. Kill conditions — frozen

- **K1** γ fits ≤ 1.0 → the shares are not flat; H1 is false; ship nothing.
- **K2** mass conservation fails → implementation defect; do not promote.
- **K3** walk-forward γ unstable across origins → underpowered; bank as a null.
- **K4** the transform improves rank 1 only by degrading rank 2 beyond its CI → it is
  moving mass rather than correcting concentration; reject.
- **K5** required inputs unavailable → fail closed and say so; do not substitute a
  hand-chosen γ.

## 7. Scope

No sportsbook variable enters the transform, γ, or the calibration. Lines are the
comparison benchmark only. Betting remains **CLOSED**. No paid odds fetch is authorized
or performed by this work.
