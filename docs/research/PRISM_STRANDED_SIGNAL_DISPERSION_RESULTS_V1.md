# PRISM — Stranded Opponent Signal Dispersion: Results V1

**Design frozen in:** `PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md`
(commit `fb803940`, before this measurement)
**Reproduce:** `python scripts/research/measure_stranded_signal_dispersion_v1.py`
**Artifacts:** `data/research/stranded_dispersion/stranded_signal_dispersion_v1_{as_of.csv,status.json}`

2026, target weeks **2–7**, 17,713 plays. Week 1 excluded — no prior evidence.
Outcomes inspected: **none**. Player-level columns read: **none**. Sportsbook
inputs: **0**. Nothing fitted, scored or promoted. No threshold proposed.

## Correction this study rests on

My earlier claim that "2026 history is gone" was **too broad and wrong**.
Verified directly:

- **2026 nflverse PBP exists** — 18,738 plays, weeks 1–7.
- **2026 participation does not** — `load_participation(seasons=[2026])` raises
  `Season must be between 2016 and 2025`.
- The box/coverage fields are **absent from PBP entirely**.

The correct boundary: **2026 game data is available; the 2026 personnel/NGS feed
is not.** Box and coverage gate prevalence genuinely cannot be measured on 2026.
Every *stranded* signal can.

## Result

All six stranded signals are both **widely dispersed across defenses** and
**stable properties of those defenses**, on the pre-committed criteria:

| signal | median IQR | consecutive-week rank r | median faced range | label |
|---|---|---|---|---|
| `def_rush_epa` | 0.1369 | **0.810** | 0.3073 | MATERIAL_DISPERSION |
| `def_pass_epa` | 0.1887 | **0.764** | 0.4588 | MATERIAL_DISPERSION |
| `success_rate_def` | 0.0439 | **0.781** | 0.0929 | MATERIAL_DISPERSION |
| `explosive_rush_rate_allowed` | 0.0346 | **0.849** | 0.0813 | MATERIAL_DISPERSION |
| `explosive_pass_rate_allowed` | 0.0413 | **0.849** | 0.0755 | MATERIAL_DISPERSION |
| `rush_stuff_rate` | 0.0595 | **0.883** | 0.1247 | MATERIAL_DISPERSION |

"Faced range" is, for each offense, the spread of opponent values it actually
confronted across weeks 2–7 — so a back's opponents varied by **0.3073** of rush
EPA and **0.1247** of stuff rate, median across offenses.

Rank correlations of 0.76–0.88 between consecutive target weeks mean these are
**persistent characteristics of a defense**, not week-to-week noise. That
matters: the pre-registration committed in advance to reporting a wide-but-
unstable signal as `UNSTABLE` rather than material. None of the six is unstable.

**The most rank-stable signal of the six — `rush_stuff_rate` at 0.883 — is
referenced by no production module at all.** Not carried, not merged, not read.

## What this does and does not mean

**Does:** the cleared transmission trace established that these signals are
built and discarded. This establishes that what is discarded is **not noise and
not uniform** — it is substantial, persistent, week-specific variation in the
football obstacle a player faces. Combined with the cleared finding that the one
live opponent→RB-rushing channel is a box gate whose two branches are a
near-constant and a corpse, the RB rushing path is discarding measurable
matchup information and retaining almost none.

**Does not:** show that consuming any of it would improve a projection.
Dispersion and persistence are **necessary but not sufficient**. A signal can
vary widely, persist, and still add nothing once the player's own baseline and
the finite allocation are accounted for. Establishing value requires a scoring
window, which is closed, and would need its own scoped proposal.

## Chronology

Every opponent aggregate for target week N is built from weeks 1..N-1 only,
play-weighted, and `_assert_strictly_prior()` raises if any aggregate draws on a
play at week ≥ N. Week 1 is **excluded, not backfilled**. The study reads
defensive team aggregates from prior weeks plus the schedule; it asserts that no
player-level column is present.

## Not established

- **Predictive value of any signal.** Untested; scoring window closed.
- **Box and coverage gate prevalence on 2026** — participation-locked.
- **That production's `def_rush_epa` is constructed like this one.** These are
  PBP-derived; production's provenance is a separate question.
- Anything about a named player's projection.

## Disclosed

Source qualification required building one signal to confirm the join. Seen
before the design was frozen: 214 team-weeks across 32 teams for `def_rush_epa`,
and four ARI rows (weeks 1–4: −0.325440, −0.178375, 0.048237, −0.311366).
Nothing else — no dispersion statistic and no other signal. Declared in the
pre-registration.
