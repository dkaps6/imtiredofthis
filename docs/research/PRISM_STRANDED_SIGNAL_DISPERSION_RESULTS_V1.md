# PRISM — Stranded Opponent Signal Dispersion: Results V2

**Design:** `PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md` + **Addendum
V2** (frozen at `5618937d`, **before** the disjoint test was run)
**Revised per:** GPT audit `6449349551`
(`STRANDED_DISPERSION_2026_DESCRIPTIVE_SPREAD_SUPPORTED__PERSISTENCE_AND_SOURCE_LINEAGE_CLAIMS_REVISION_REQUIRED`)
**Reproduce:** `python scripts/research/measure_stranded_signal_dispersion_v1.py`
**Artifacts:** `data/research/stranded_dispersion/`

2026, 17,713 plays. **No target-week outcomes**; prior-game play results used
only to build as-of defensive statistics; **no model scoring**. Sportsbook
inputs: 0. Player-level columns: none. Nothing fitted or promoted.

## The headline reverses V1

**V1 claimed these signals were "persistent properties of a defense, not noise."
That is WITHDRAWN. Measured properly, it is false.**

V1's consecutive-week rank correlations compared cumulative windows that shared
all but one week — target week 6 is built from weeks 1–5 and week 7 from weeks
1–6. A cumulative aggregate correlated with itself-plus-one-week is
near-tautological, so the 0.764–0.883 figures were forced by sample reuse. I
built a statistic that could only come out high and read it as a finding.

The replacement, frozen before it ran, uses **two disjoint windows sharing no
play**: EARLY weeks 1–3 against LATE weeks 4–7.

| signal | V1 overlapping | **disjoint rho** | eligible | verdict |
|---|---|---|---|---|
| `def_rush_epa` | 0.810 | **−0.350** | 32 | NOT_PERSISTENT |
| `def_pass_epa` | 0.764 | **0.113** | 32 | NOT_PERSISTENT |
| `success_rate_def` | 0.781 | **0.080** | 32 | NOT_PERSISTENT |
| `explosive_rush_rate_allowed` | 0.849 | **0.256** | 32 | WEAKLY_PERSISTENT |
| `explosive_pass_rate_allowed` | 0.849 | **0.103** | 32 | NOT_PERSISTENT |
| `rush_stuff_rate` | 0.883 | **0.017** | 32 | NOT_PERSISTENT |

Cutoffs (`PERSISTENT` ≥ 0.40, `WEAKLY_PERSISTENT` ≥ 0.20) were fixed in the
addendum before the test ran. Eligibility required ≥ 40 qualifying plays per
half; all 32 defenses qualified for every signal, so nothing was excluded.

**The signal V1 singled out as most stable — `rush_stuff_rate` at 0.883 —
measures 0.017 on disjoint samples.** Five of six show no persistence; one is
weak.

## What now stands, and what it means

**Stands:** the six signals have real **cross-sectional dispersion**. Within a
given week, defenses differ substantially — median IQR 0.0346–0.1887, and the
median range an offense faced across weeks 2–7 was 0.3073 of rush EPA. Defenses
are not interchangeable in any single snapshot.

**Does not stand:** that this dispersion is a stable, exploitable property.
**Dispersion without rank persistence is closer to noise than to signal.** The
V1 conclusion — that production discards "substantial, persistent, week-specific
variation" — is withdrawn. What production discards is substantial and
week-specific; on this evidence it is **not** demonstrably persistent.

**Symmetric caution.** This does not establish that the signals *are* noise
either. With 32 defenses and a 3-week / 4-week split, one rho per signal carries
wide uncertainty, the halves differ in length and in opponent schedule, and the
pre-registration said so in advance. The test separates "clearly persistent"
from "clearly not"; it cannot finely rank the six, and it is not used to.

So the correct position is: **persistence is NOT ESTABLISHED**, and the case for
consuming these signals is correspondingly weaker than V1 presented it.

## The six are not six stranded production fields

V1's framing was wrong. Per-signal provenance:

| signal | class |
|---|---|
| `def_rush_epa` | **TRANSPORTED_BUT_STRANDED** — on `TeamContext`, read by no rule |
| `def_pass_epa` | **CONSUMED** on the promoted QB route (`qb_pass_synthesis_v1`) |
| `success_rate_def` | **READ BY PRODUCTION** — `rules_v2.success_diff()` reads `defense.success_rate_def`, called in `project_game_script()` |
| `explosive_rush_rate_allowed` | **NEW PBP-DERIVED CANDIDATE** — lineage to production's `explosive_play_rate_allowed` not established |
| `explosive_pass_rate_allowed` | **NEW PBP-DERIVED CANDIDATE** — lineage not established |
| `rush_stuff_rate` | **NEW PBP-DERIVED CANDIDATE** — production has a column of this name referenced by no module, but lineage to this construct is not established |

Only **`def_rush_epa`** is transported-but-stranded on the cleared evidence.
Whether `success_rate_def` changes a named player's final projection is a
separate routing question, since `pass_share` is hardcoded at 0.57 downstream.

## `MATERIAL_DISPERSION` was not a predeclared verdict

The `TRIVIAL_IQR` cutoffs (0.05 EPA, 0.02 rates) were **never in the frozen V1
design**, which said only "non-trivial". They are **post-preregistration
operationalization**. The numeric dispersion results stand; the label is not a
validated predeclared verdict and is not treated as one.

## Outcome language, corrected

`load_pbp` reads realized `epa`, `success` and `yards_gained` from games already
played — prior-play outcomes by construction. V1's absolute "outcomes inspected:
none" is **withdrawn**. The precise claim: **no target-week outcomes**;
prior-game play results used only to construct as-of defensive statistics; **no
model scoring**.

## Chronology

The as-of construction for target week N uses weeks 1..N-1 only, and
`_assert_strictly_prior()` raises on a leak — a test injects one to confirm. The
disjoint persistence test is not an as-of construction and has no chronology
requirement between halves; it asks only whether a characteristic measured in
one window predicts its rank in a separate one.

## Not established

- **Persistence of any of the six.** Five NOT_PERSISTENT, one WEAKLY.
- **That the signals are noise** — the test is underpowered to assert that.
- **Predictive value for any player projection.** Untested; scoring closed.
- **Lineage** from three of the constructs to any production column.
- Box/coverage gate behaviour on 2026 — participation-locked.
