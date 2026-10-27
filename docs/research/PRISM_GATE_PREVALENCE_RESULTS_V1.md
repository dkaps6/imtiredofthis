# PRISM — Opponent Gate Prevalence: Results V1

**Design frozen in:** `PRISM_GATE_PREVALENCE_PREREGISTRATION_V1.md` (commit `be4b0838`)
**Reproduce:** `python scripts/research/measure_opponent_gate_prevalence_v1.py`
**Artifacts:** `data/research/gate_prevalence/opponent_gate_prevalence_v1_{team_week.csv,status.json}`

2025 weeks 1–18, 43,049 joined plays, 512 team-weeks per arm.
Outcomes inspected: **none**. Sportsbook inputs: **0**. Nothing fitted, scored
or promoted. No threshold proposed. **This is a 2025 distributional prior, not a
2026 claim.**

## Headline

**Three of the four opponent gates in `rules_v2.matchup_multipliers` are
structurally non-informative, and that result is robust to every methodological
choice tested — all eight arms agree.**

Fraction of 512 team-weeks where each gate condition holds, `AS_OF_PREGAME`:

| arm | light ≥.60 | heavy ≥.60 | zone ≥.60 | man ≥.50 |
|---|---|---|---|---|
| PROD_UNFILTERED · DEF_DEPLOYED · ALL_ROWS | 99.2% | 0.0% | 0.0% | 0.0% |
| PROD_UNFILTERED · DEF_DEPLOYED · COV_CLASSIFIED | 99.2% | 0.0% | 71.1% | 5.9% |
| PROD_UNFILTERED · OFF_FACED · ALL_ROWS | 99.2% | 0.0% | 0.0% | 0.0% |
| PROD_UNFILTERED · OFF_FACED · COV_CLASSIFIED | 99.2% | 0.0% | 80.5% | 3.1% |
| SCRIMMAGE · DEF_DEPLOYED · ALL_ROWS | 90.0% | 0.0% | 0.4% | 0.0% |
| SCRIMMAGE · DEF_DEPLOYED · COV_CLASSIFIED | 90.0% | 0.0% | 73.4% | 5.9% |
| SCRIMMAGE · OFF_FACED · ALL_ROWS | 91.2% | 0.0% | 0.2% | 0.0% |
| SCRIMMAGE · OFF_FACED · COV_CLASSIFIED | 91.2% | 0.0% | 84.2% | 2.3% |

Against the pre-committed cutoffs (`NEAR_CONSTANT` > 90%, `NEAR_DEAD` < 10%):

- **`heavy_box_rate >= 0.60` — NEAR_DEAD in 8 of 8 arms, 0.0% everywhere.**
  Median heavy-box rate is 0.035–0.046; a defense putting 8+ in the box on 60%
  of plays does not exist. The `rb_rush_eff_mult *= 0.94` branch is dead.
- **`light_box_rate >= 0.60` — NEAR_CONSTANT in 8 of 8 arms, 90.0–99.2%.**
  Production defines light as `box <= 6`, and a 6-man box is the NFL default
  front, so this is not a tactical state. It is a constant ×1.07.
- **`coverage_man_rate >= 0.50` — NEAR_DEAD in 8 of 8 arms, 0.0–5.9%.**
  Median man rate is 0.166–0.324.
- **`coverage_zone_rate >= 0.60` — the only informative gate**, and only under
  the coverage-classified denominator: **DISCRIMINATING at 71.1–84.2%**, versus
  NEAR_DEAD at 0.0–0.4% over all rows.

## Two consequences, now sized

### 1. RB rushing has effectively no opponent signal — with a number behind it

`rb_rush_eff_mult` is the **only** opponent→RB-rushing-efficiency channel in
production (cleared transmission trace). Its two gates are a constant and a
corpse: light fires 90–99% of team-weeks, heavy never. So the multiplier is
≈ ×1.07 for nearly every team in nearly every week — which carries the same
information as no opponent input at all.

Combined with the already-cleared findings that `ml_v2` and `state_v2` take no
opponent input, and that `def_rush_epa` is transported onto `TeamContext` and
read by no rule, the picture is consistent: **the opponent cannot meaningfully
move an RB's rushing projection**, and the one live channel is a near-constant.

### 2. My own Kupp finding is SMALL. The quiet finding is the big one.

This measurement shrinks a result I reported prominently and promotes one I
reported in passing.

**The WR role-labeling apparatus reads only the man gate.** `wr1_target_mult`,
`wr1_5_target_mult` and `slot_target_mult` key on `coverage_man_rate >= 0.50`,
which fires in **≤ 5.9%** of team-weeks. So whether a receiver is labeled WR1,
WR1_5 or nothing at all is **moot in roughly 94%+ of weeks** — labeled and
unlabeled WRs alike receive ×1.000. Cooper Kupp's structural exclusion is real,
but it is inert the overwhelming majority of the time. The rank-gating defect is
**low-frequency**.

**But no WR role reads zone, and the zone gate fires 71–84% of team-weeks.**
That is the material asymmetry: in roughly three weeks out of four, every tight
end gets ×1.15 (and ×1.265 with middle-open) and every back and fullback gets
×1.20 on their receiving targets, while **no wide receiver can receive any zone
adjustment at all**, because no WR branch reads `coverage_zone_rate`.

So the consequential defect is not which WR gets a label. It is that the
position with the most receiving volume in football is the only one with no
route to the single opponent gate that actually carries information.

## Secondary findings

**Production counts kickoffs and punts as light boxes.**
`compute_personnel_rates()` applies no play-type filter, and
`defenders_in_box == 0` is a non-scrimmage sentinel (kickoff 2,926, punt 2,042,
extra point 1,330, field goal 1,140; only 52 zero-box rows are scrimmage plays).
This inflates `light_box_rate` from 90.0% to 99.2% firing. It does not change
any label here — the gate is NEAR_CONSTANT either way — but it is a real
data-quality defect in the participation path.

**The denominator convention is decisive; the orientation is second-order.**
Isolating each axis on the scrimmage arm:

- denominator: zone median 0.4054 (`ALL_ROWS`) vs 0.6759 (`COVERAGE_CLASSIFIED`)
  → 0.4% vs 73.4% firing. This single convention determines whether the zone
  gate exists at all.
- orientation: light median 0.7042 (`OFFENSE_FACED`) vs 0.7021
  (`DEFENSE_DEPLOYED`); zone 84.2% vs 73.4% firing. Real, but far smaller.

Production's live values behave like the **coverage-classified** arm: the
cleared Walker/SEA decode showed the zone gate firing for Kansas City in 2026
Week 7, which is impossible under an all-rows denominator and expected under a
classified one.

## What this does not establish

- **Not a 2026 claim.** 2026 participation does not exist; Sharp serves
  season-to-date with no by-week snapshot. This is a 2025 prior.
- **Production prefers Sharp** over the participation path for box rates, so
  these are the participation-derived values, not necessarily what production
  used in a given week.
- **No correctness verdict.** `NEAR_DEAD` and `NEAR_CONSTANT` describe
  reachability. They do **not** establish that any gate is wrong, that any
  alternative threshold is better, or that changing anything would improve a
  projection. No threshold, coefficient or sign is proposed.
- **Whether `OFFENSE_FACED` is a defect or an intended proxy** remains a
  semantic question for audit.

## Disclosed

The pre-registration said "per team-week" without fixing whether that means a
single week or the season-to-date value a pregame run would hold. Both are
reported rather than one chosen afterwards; `AS_OF_PREGAME` is the
production-faithful basis and is what the tables above use. League-level
aggregates seen during source qualification were declared in the
pre-registration before the design was frozen.
