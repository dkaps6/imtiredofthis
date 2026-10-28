# PRISM — Stranded Opponent Signal Dispersion: Pre-Registration V1

**Status:** design frozen before measurement. **No fitting, no scoring, no
promotion, no threshold.** Read-only, outcome-blind for the target week.
**User-authorized:** 2026-10-27, continuing toward better projections.

## Correction that motivates this study

An earlier statement of mine — that "2026 history is gone" — was **too broad and
wrong**. Verified directly:

- **2026 nflverse PBP exists**: 18,738 plays, weeks 1–7.
- **2026 participation does not**: `load_participation(seasons=[2026])` raises
  `Season must be between 2016 and 2025`.
- The box/coverage fields are **absent from PBP entirely**
  (`defenders_in_box`, `defense_man_zone_type`, `defense_personnel`).

So the correct boundary is: **2026 game data is available; the 2026
participation feed is not.**

> ⚠️ **CORRECTION (2026-10-28):** the original wording here said "the 2026
> personnel/**NGS** feed is not" available. That **conflated two different nflverse
> endpoints and is wrong about NGS.** `load_participation(seasons=[2026])` does fail
> (*"Season must be between 2016 and 2025"*), which is why `defenders_in_box`,
> `defense_man_zone_type` and `defense_personnel` are unavailable — that part stands.
> But **`load_nextgen_stats` works for 2026 weeks 1–7**: passing 253 rows, rushing 253,
> receiving 617. The rushing feed carries `expected_rush_yards`,
> `rush_yards_over_expected`, `efficiency` and **`percent_attempts_gte_eight_defenders`**
> (a per-player stacked-box measure), and the receiving feed carries `avg_separation`,
> `avg_cushion` and `avg_yac_above_expectation`. NGS is used **nowhere** in this
> repository outside Migration 75, which is explicitly marked
> `production_actionable = False`. Any future use must be strictly-prior-weeks-only:
> target-week NGS rows are **outcomes**. See issue #535 comment 6479259425.
> Gate prevalence for box/coverage genuinely cannot
be measured on 2026. But every *stranded* opponent signal the cleared
transmission trace identified **is** computable for 2026 weeks 1–7:
`def_rush_epa`, `def_pass_epa`, `explosive_play_rate_allowed`,
`success_rate_def`, `rush_stuff_rate`, and yards-before-contact components.

## The question

The cleared traces established that these signals are built and never consumed.
They did not establish **how much matchup information that discards.** If an
opponent's rush defense barely varies across the teams a back faces, stranding
it costs little. If it varies widely, the RB rushing path is discarding real,
measurable, week-specific football information.

This measures **dispersion only**. It cannot and does not show that consuming
any signal would improve a projection.

## Chronology discipline — fail closed

Every opponent aggregate for target week N is built from **weeks 1..N-1 only**,
play-weighted. Week N's own plays are never used to describe week N's matchup.
Week 1 has no prior weeks and is therefore **excluded**, not backfilled. The
script asserts this and raises if any target week's aggregate draws on a play at
week ≥ N.

No player outcome, result, box score, line, odds or price is read. The study
reads only defensive team aggregates from prior weeks and the schedule.

## Frozen design

**Population.** 2026, target weeks **2–7** (week 1 excluded for lack of prior
evidence), all 32 defenses, from `load_pbp(seasons=[2026])`.

**Signals**, each computed per defense as an as-of-pregame aggregate over
weeks 1..N-1:

1. `def_rush_epa` — mean `epa` on `play_type == "run"`
2. `def_pass_epa` — mean `epa` on `play_type == "pass"`
3. `success_rate_def` — mean `success`
4. `explosive_rush_rate_allowed` — share of runs with `yards_gained >= 10`
5. `explosive_pass_rate_allowed` — share of passes with `yards_gained >= 20`
6. `rush_stuff_rate` — share of runs with `yards_gained <= 0`

Thresholds 10 / 20 / ≤0 are the conventional nflverse definitions and are fixed
here, not tuned.

**Reported statistics.** For each signal:

- **cross-sectional dispersion**: within each target week, the SD, IQR and
  min–max range across the 32 defenses — how different the available matchups
  are from one another.
- **faced dispersion**: for each offense, the SD and range of the opponent
  values it actually faced across weeks 2–7 — how much the matchup an offense
  confronts changes week to week.
- **rank stability**: Spearman correlation of defensive rank between consecutive
  target weeks — whether the signal is a stable property or noise.

**Pre-committed interpretation rule.** A signal is reported as carrying
`MATERIAL_DISPERSION` only if **both** its cross-sectional IQR is non-trivial
**and** its consecutive-week rank correlation is positive. A signal with wide
spread but no rank stability is reported `UNSTABLE` — wide but not a property of
the defense. Everything else is `NARROW`. These labels describe **dispersion and
stability only**; they are not evidence of predictive value, and no threshold,
coefficient or sign is proposed.

## Full disclosure — what I already saw

Source qualification required building one signal to confirm the join works. I
saw: 214 team-weeks across 32 teams and weeks 1–7 for `def_rush_epa`, and four
ARI rows (weeks 1–4, values −0.325440, −0.178375, 0.048237, −0.311366). Nothing
else — no dispersion statistic, no cross-team comparison, no other signal.

## What this cannot establish

- That consuming any stranded signal would improve accuracy. **Untested.** That
  needs a scoring window, which is closed.
- Anything about the box/coverage gates on 2026 — participation-locked.
- Production behaviour. These are PBP-derived values; whether production's
  `def_rush_epa` matches them in construction is a separate provenance question.
- Any claim about a named player's projection.

---

# ADDENDUM V2 — disjoint-sample persistence test

**Frozen before running the test below.** Required by GPT audit `6449349551`
blocker 1. **No fitting, no scoring, no threshold selection.**

## What is retracted and why

V1's "consecutive-week rank correlation" is **not** evidence of persistence and
the claims built on it are withdrawn. `build_as_of` constructs target week N
from weeks 1..N-1, so the week-6 vs week-7 comparison places weeks 1–5 against
weeks 1–6 — **five of six weeks shared**. A cumulative aggregate correlated with
itself-plus-one-week is near-tautological; high coefficients are forced by
sample reuse, not by defensive skill. The reported 0.764–0.883 figures are
**overlapping-history autocorrelations** and nothing more.

Withdrawn as established: "persistent properties of a defense", "not noise",
and the `MATERIAL_DISPERSION` label insofar as it depended on rank stability.

## Frozen design — disjoint split-half

**Samples.** Two fully disjoint, non-overlapping week ranges from 2026:
**EARLY = weeks 1–3**, **LATE = weeks 4–7**. They share no play. Week 1 is
included here because this test is not an as-of pregame construction and has no
chronology requirement between the halves — it asks only whether a defensive
characteristic measured in one window predicts its rank in a separate window.

**Statistic.** For each signal, Spearman rank correlation of defensive values
between EARLY and LATE, across eligible defenses. This is a genuine
out-of-sample persistence estimate because no play contributes to both sides.

**Eligibility, fixed now.** A defense enters a signal's correlation only if it
has **≥ 40 qualifying plays in each half** — run plays for the rush-family
signals (`def_rush_epa`, `explosive_rush_rate_allowed`, `rush_stuff_rate`),
pass plays for the pass-family (`def_pass_epa`, `explosive_pass_rate_allowed`),
and all plays for `success_rate_def`. Eligible-defense counts and any exclusions
are reported, not hidden.

**Pre-committed interpretation, with numeric cutoffs stated here rather than
left to the script.** A signal is reported:

- `PERSISTENT` if the disjoint Spearman rho **≥ 0.40**;
- `WEAKLY_PERSISTENT` if **0.20 ≤ rho < 0.40**;
- `NOT_PERSISTENT` if **rho < 0.20**.

These cutoffs are conventional and are fixed before the test is run. They
describe rank persistence between two windows **only**. They are not evidence of
predictive value for any player projection, and no threshold, coefficient or
sign is proposed from them.

**Honest limitation, stated in advance.** With 32 defenses and a 3-week / 4-week
split, a single rho carries wide uncertainty and the halves differ in length and
in opponent schedule. This test can distinguish "clearly rank-persistent" from
"clearly not" — it cannot finely rank the six signals against one another, and
it will not be used to.

## Also corrected in V2

- **The `TRIVIAL_IQR` cutoffs (0.05 EPA, 0.02 rates) were NOT in the frozen V1
  design**, which said only "non-trivial". They are **post-preregistration
  operationalization** and are relabelled as such. The numeric dispersion
  results stand; the `MATERIAL_DISPERSION` verdict is not a predeclared one.
- **"Outcome-blind" is restated precisely.** `load_pbp` reads realized `epa`,
  `success` and `yards_gained` from games already played — these are prior-play
  outcomes by construction. The correct statement is: **no target-week
  outcomes**; prior-game play results are used only to construct as-of defensive
  statistics; **no model scoring**. The absolute phrasing "outcomes inspected:
  none" is withdrawn.
- **The six signals are not six stranded production fields.** A per-signal
  provenance classification is required, because `success_rate_def` **is** read
  by production via `rules_v2.success_diff()` inside `project_game_script()`.
