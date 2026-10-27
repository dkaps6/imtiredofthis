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
personnel/NGS feed is not.** Gate prevalence for box/coverage genuinely cannot
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
