# PRISM — Opponent Gate Prevalence: Pre-Registration V1

**Status:** design frozen before the per-team/per-week measurement. **No fitting,
no scoring, no promotion, no threshold proposal.**
**User-authorized:** 2026-10-27, goal explicitly stated as better projections.
**Frontier named by:** GPT audit `6446595260` — source-qualified preservation of
opponent field values, before any claim about actual individualized opponent
effects.

## Why this exists

Three consecutive audits marked gate firing frequency **NOT ESTABLISHED**. Every
defect found in the cleared transmission and role-routing traces is therefore
**unsized**: we know `def_rush_epa` is stranded and that no WR role reads zone,
but not whether any of it moves a meaningful number of players. Fixing things in
that state is guessing.

## Source qualification — what is and is not possible

**The 2026 by-week retroactive measurement is SOURCE BLOCKED.** Established
before designing anything:

- `coverage_man_rate`, `coverage_zone_rate`, `middle_open_rate`,
  `light_box_rate`, `heavy_box_rate` are **Sharp Football** fields.
  `sharpfootball_pull.py` fetches `?season=<season>` and writes
  `sharp_*_<season>.csv` — **season-to-date, with no by-week snapshot.** Fetching
  today returns current values, so using them to judge a Week-3 gate state would
  be look-ahead contamination.
- My own Week-7 preservation filter kept opponent column **names** and dropped
  their **values**: the preserved `team_form.csv` retained 4 of 108 columns and
  `sharp_team_form.csv` 3 of 47.
- `data/market_track_record/explanatory/` is **empty** — the forward archive
  contract has not produced yet.
- Nothing else in the repository commits these values.

**What is available:** nflverse `load_participation` for **2025** (prior season,
legitimate per AGENTS.md §1 for historical/distributional work) carries 45,184
plays with `defenders_in_box` **and** `defense_man_zone_type`, joins to PBP on
`(nflverse_game_id, play_id)` with **45,184 / 45,184 defteam resolved**, and
covers weeks 1–20. 2026 participation does not exist, which is the already
documented `PARTIAL_GAME` source block.

So the measurement is a **2025 distributional prior on gate reachability**, not a
2026 claim. That is sufficient for sizing, and it is the only honest version.

## FULL DISCLOSURE — what I already looked at

Source qualification required inspecting league-level aggregates, so the
following were seen **before** this design was frozen, and are declared rather
than presented later as findings of a blind study:

- `defenders_in_box == 0` occurs on 9,093 rows and is a **non-scrimmage
  sentinel**: kickoff 2,926, punt 2,042, extra_point 1,330, field_goal 1,140,
  no_play 1,071, qb_kneel 452, qb_spike 80, pass 37. Only **52** box==0 rows are
  scrimmage plays.
- `compute_personnel_rates()` applies **no play-type filter**, so production's
  participation path counts every kickoff and punt as a light box.
  League `light_box_rate`: **0.7416** unfiltered vs **0.6757** scrimmage-only.
- League zone rate depends entirely on the denominator: **0.3324** over all rows,
  **0.6810** over coverage-classified rows, **0.6893** over pass plays.

What I have **not** seen, and what this design therefore still governs: any
per-team, per-week, or per-orientation distribution; any team's individual gate
state; any count of affected players.

## Frozen design

**Population.** 2025, weeks 1–18 (regular season), all 32 teams. Scrimmage plays
`play_type ∈ {pass, run}` for the football-correct arm; unfiltered for the
production-faithful arm.

**Quantities, computed with production's own definitions** from
`make_team_form.compute_personnel_rates` — `light = box <= 6`,
`heavy = box >= 8`, mean over plays — and **not** re-derived:

1. `light_box_rate`, `heavy_box_rate`
2. `coverage_zone_rate`, `coverage_man_rate` from `defense_man_zone_type`

**Two orientations, both reported.** Participation has **no defensive team
column**; production groups by `possession_team`, so its value is *the box this
offense faced*, while `rules_v2.matchup_multipliers` reads it as
`defense.light_box_rate`, a *defensive tendency*. Both are computed:

- `OFFENSE_FACED` — grouped by `possession_team`, reproducing production
- `DEFENSE_DEPLOYED` — grouped by `defteam` via the PBP join

**Two denominators for coverage, both reported:** `ALL_ROWS` and
`COVERAGE_CLASSIFIED`.

**Two play-type arms, both reported:** `PRODUCTION_FAITHFUL_UNFILTERED` and
`SCRIMMAGE_ONLY`.

**Reported statistics**, per arm × orientation × denominator, per team-week and
pooled: the rate itself, and the fraction of team-weeks for which each
production gate condition holds — `light_box_rate >= 0.60`,
`heavy_box_rate >= 0.60`, `coverage_zone_rate >= 0.60`,
`coverage_man_rate >= 0.50`.

**Pre-committed interpretation rules**, so the conclusion cannot be chosen after
the fact:

- a gate whose condition holds in **> 90%** of team-weeks is reported
  `NEAR_CONSTANT` — it carries little matchup information regardless of its
  multiplier;
- **< 10%** is reported `NEAR_DEAD`;
- otherwise `DISCRIMINATING`.
- These labels describe **reachability only**. They are **not** a verdict on
  correctness, not a proposed threshold, and not evidence that any alternative
  would project better.

## Explicit prohibitions

No outcome, score, result, line, odds or price is read. No coefficient, sign or
threshold is proposed or fitted. No subgroup is selected after seeing results.
No promotion, and no implied permission from this measurement to change
production. 2026 gate prevalence is **not** claimed. Sharp is **not** re-fetched,
so no current-season value contaminates a past-week judgement.

## What this cannot establish

Production normally **prefers Sharp** over the participation path for box rates,
so these figures describe what the **participation-derived** fields would give,
not necessarily the values production used in any specific week. Whether the
`OFFENSE_FACED` orientation is a defect or an intended proxy is a **semantic
question for audit**, not something this measurement decides.
