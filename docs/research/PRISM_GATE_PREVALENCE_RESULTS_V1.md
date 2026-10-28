# PRISM — Opponent Gate Prevalence: Results V1 (revised)

**Design frozen in:** `PRISM_GATE_PREVALENCE_PREREGISTRATION_V1.md` (commit `be4b0838`, before the result commit)
**Revised per:** GPT audit `6448026674`
(`GATE_PREVALENCE_2025_PROXY_SUPPORTED__PRODUCTION_AND_CROSS_BASIS_GENERALIZATION_REVISION_REQUIRED`)
**Reproduce:** `python scripts/research/measure_opponent_gate_prevalence_v1.py`
**Artifacts:** `data/research/gate_prevalence/opponent_gate_prevalence_v1_{team_week.csv,status.json}`

2025 weeks 1–18, 43,049 joined plays. Outcomes inspected: **none**. Sportsbook
inputs: **0**. Nothing fitted, scored or promoted. No threshold proposed.

## Read this before any number below

Every figure here is a **2025 nflverse-participation-derived proxy**. It is
**not** a measurement of production gate firing, because production **prefers
Sharp Football** for these fields and no past-week Sharp values, denominators or
snapshots exist. Treat all of it as **reachability estimates**, not validated
live behaviour, and not evidence that any gate carries predictive information.

These are **four studied gates, not all the opponent gates** in
`rules_v2.matchup_multipliers`. **`middle_open_rate >= 0.50`** (feeding
`te_target_mult` and `slot_target_mult`) and the **pressure mismatch gate**
(feeding `rb_rec_target_mult`, `pass_eff_mult`, `sack_mult`, `int_mult`,
`volatility_mult`) are **unmeasured here**.

## Headline, correctly scoped

The pre-registration left the **time basis** open, and the script reports both.
That basis is a **ninth dimension**, so a claim of robustness must span it. V1 of
this document asserted robustness across eight arms while silently holding the
basis fixed. Corrected, across all **16** arm × basis combinations:

| gate | labels across 16 combinations | unconditional? |
|---|---|---|
| `heavy_box_rate >= 0.60` | **NEAR_DEAD 16 / 16** | **yes** |
| `light_box_rate >= 0.60` | NEAR_CONSTANT 8, DISCRIMINATING 8 | no — **basis-dependent** |
| `coverage_zone_rate >= 0.60` | NEAR_DEAD 8, DISCRIMINATING 8 | no — **denominator-dependent** |
| `coverage_man_rate >= 0.50` | NEAR_DEAD 14, DISCRIMINATING 2 | no |

**Exactly one gate is unconditionally non-informative in this proxy:
`heavy_box_rate >= 0.60`, NEAR_DEAD in every combination, 0 of 512 team-weeks
firing on the as-of basis.** Median heavy-box rate is 0.035–0.046.

The other three are conditional, and the conditions are:

- **`light_box_rate`** — NEAR_CONSTANT (90.0–99.2%) on `AS_OF_PREGAME`;
  **DISCRIMINATING** (69.9% scrimmage, 86.0% unfiltered) on `SINGLE_WEEK`.
- **`coverage_zone_rate`** — NEAR_DEAD (0.0–0.4%) on `ALL_ROWS`;
  DISCRIMINATING (71.1–84.2%) on `COVERAGE_CLASSIFIED`.
- **`coverage_man_rate`** — NEAR_DEAD (0.0–5.9%) in 14 combinations;
  DISCRIMINATING (12.1%) in the two unfiltered / coverage-classified /
  single-week combinations.

### Why `AS_OF_PREGAME` is preferred — and why that is a choice, not a result

A pregame value for week N is built from weeks 1..N-1, so the season-to-date
basis is what a gate is actually evaluated against; and a single week's rate
rests on far fewer plays, so sampling noise alone pushes rates across a fixed
threshold. Both are reasons to prefer it as the planning basis. Neither makes
its result basis-independent. **Selecting the basis is part of the analysis.**

## What this supports, with qualifiers attached

On the **2025 participation-derived, as-of-pregame, coverage-classified proxy**:

- `rb_rush_eff_mult`'s two gates are a near-constant and a corpse, so on this
  proxy the multiplier would sit at ≈ ×1.07 for nearly every team-week. Taken
  with the cleared findings that `ml_v2` and `state_v2` accept no opponent input
  and that `def_rush_epa` is transported onto `TeamContext` and read by no rule,
  this is **consistent with** an RB rushing path that the opponent cannot
  meaningfully move. It does **not** measure that in production.
- The WR label apparatus (`wr1`, `wr1_5`, `slot`) reads **only** the man gate,
  which on this proxy fires in ≤5.9% of as-of team-weeks (12.1% in two
  single-week combinations). So on this proxy the WR1 / WR1_5 / unlabeled
  distinction would be inert in the large majority of weeks, which **re-sizes
  the Cooper Kupp exclusion downward** from how the role-routing trace presented
  it.
- No WR role reads `coverage_zone_rate`, while `te_target_mult` and
  `rb_rec_target_mult` do; on the classified denominator that gate fires in
  71.1–84.2% of as-of team-weeks. **This is the asymmetry worth grounding
  next** — the highest-volume receiving position has no route to the gate that
  looks most reachable on this proxy.

None of the above establishes predictive value, and none of it licenses a
change.

## Secondary findings

**Production counts kickoffs and punts as light boxes.**
`compute_personnel_rates()` applies no play-type filter, and
`defenders_in_box == 0` is a non-scrimmage sentinel — kickoff 2,926, punt 2,042,
extra point 1,330, field goal 1,140; only **52** zero-box rows are scrimmage
plays. Unfiltered light-box firing is 99.2% against 90.0% scrimmage-only on the
as-of basis. The `heavy_box_rate` label is unaffected.

**Null box rows are now handled literally.** `compute_personnel_rates()`
evaluates `(to_numeric(box) <= 6).astype(float)` over every participation row,
and `NaN <= 6` is `False`, so a missing box counts in the denominator as
"not light". V1 of this script dropped missing-box rows from numerator **and**
denominator, so the arm was not literally source-faithful. The literal treatment
is now implemented (and `box_null_rows` is reported per team-week) rather than
argued immaterial.

**Denominator is decisive; orientation is second-order.** On the scrimmage arm:
denominator moves zone median 0.4054 → 0.6759 and firing 0.4% → 73.4%;
orientation moves light median 0.7042 → 0.7021 and zone firing 84.2% → 73.4%.

## Withdrawn

**V1 concluded that production's Sharp-fed coverage fields behave like the
`COVERAGE_CLASSIFIED` arm**, inferring it from the single source-conditioned
2026 Week-7 Kansas City zone decode. **Withdrawn.** The `ALL_ROWS` arm fires
rarely but nonzero (2 of 512 on scrimmage / defense-deployed), Sharp is a
different provider whose denominator is unknown, and one inferred gate cannot
identify a provider convention. **Sharp's denominator and provenance are
UNESTABLISHED.**

**V1's headline** — "three of the four opponent gates are non-informative, robust
across every methodological choice, all eight arms agree" — is withdrawn and
replaced by the scoped table above.

## Not established

- Not a 2026 claim; 2026 participation does not exist and Sharp has no by-week
  snapshot.
- Production gate firing, in any week.
- Sharp's denominator convention and provenance.
- The `middle_open_rate` and pressure gates (**unmeasured**).
- Whether `OFFENSE_FACED` is a defect or an intended proxy — participation has
  no defensive team column, so production groups by `possession_team` while
  `rules_v2` reads the field as a defensive tendency. A semantic question for
  audit.
- Any predictive or correctness verdict. `NEAR_DEAD` / `NEAR_CONSTANT` /
  `DISCRIMINATING` describe reachability under stated conditions only.

## Verification scope

`tests/test_measure_opponent_gate_prevalence_v1.py` pins the production
thresholds against `make_team_form.py`, the literal null treatment, the
orientation grouping actually differing, the pre-registered cutoffs, both time
bases being reported for every arm, the unconditional-gate set being exactly the
cross-basis-unanimous set, the withdrawal of the Sharp inference, and that this
document cannot reassert V1's withdrawn headline.

Local suite and CI scope are reported on the submission comment for the exact
SHA rather than claimed here.
