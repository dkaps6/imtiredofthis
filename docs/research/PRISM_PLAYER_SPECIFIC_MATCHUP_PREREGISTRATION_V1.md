# PRISM — Player-specific matchup response: PREREGISTRATION (V1)

**Frozen before any scoring.** Written and committed prior to fitting or evaluating
anything. If a result below contradicts the hypothesis, that is the finding.

## 0. Why this study exists

Production answers "what obstacle does this player face this week" with **generic
positional rules**, and most of them are inert:

- `te_target_mult *= 1.10` — applied to every TE on the slate, identically;
- `wr1_target_mult` = **1.0 on all 396 W7 rows** — no WR gets any matchup adjustment;
- the SLOT branch is unreachable dead code;
- `rules_pass_rate` = **0.57 on all 1,412 rows** — a literal constant;
- the `middle_open` gate fired for **all 32 teams** — a constant, not a discriminator.

(Established in `PRISM_REAL_GATE_STATE_WEEK7_V1`, `PRISM_SLOT_ROLE_UNREACHABLE_V1`,
`PRISM_GAME_SCRIPT_TRANSMISSION_TRACE_V1`, `PRISM_RATE_SCALE_PATH_ATTRIBUTION_V1`.)

The goal is a named-player answer: *this player, against this defense, given the
profile we built for him.* A position-keyed multiplier cannot express that, because it
is identical for every player sharing the label. This study tests whether a
player-specific answer is **earnable from data** before any such mechanism is built.

## 1. Hypothesis

**H1.** A named receiver's week-*W* outcome, measured against **his own** strictly-prior
baseline, depends on the interaction between *his own prior profile* and the
*opponent's prior defensive profile*, beyond what a position-level defensive
adjustment explains.

**H0.** Any apparent player-specific matchup response is explained by (a) the player's
own baseline, (b) a position-level defensive main effect, and (c) noise. If H0 holds,
generic positional rules are defensible and no player×defense mechanism is warranted.

The primary player axis is **prior aDOT** (air yards per target). Mechanism: a
4-yard-aDOT slot receiver and a 14-yard-aDOT boundary receiver face materially
different consequences from the same zone-heavy defense. That is a player property,
not a position property, and it is already present in the identity card
(`mean_air_yards`).

## 2. Data, frozen

**Season:** 2025 only (full season). 2026 is the live/production season and is NOT
scored here. 2025 is the last season inside the `load_participation` window
(2016–2025), which is the only authoritative play-level coverage source available.

**Outcomes** (`nflreadpy.load_player_stats`, 2025): `receiving_yards` (primary),
`targets` and realized `ypt` (secondary, reported but not used to select the winner).

**Player axis** — strictly from weeks `1..W-1`, never week `W`:
- `prior_adot` = Σ`receiving_air_yards` / Σ`targets`
- `prior_ypt` = Σ`receiving_yards` / Σ`targets`
- `prior_tgt_per_game`, `prior_catch_rate`

**Defense axis** — strictly from the opponent's defensive plays in weeks `1..W-1`
(`nflreadpy.load_participation` joined to PBP for team identity):
- `def_zone_rate` = ZONE / (ZONE + MAN) over coverage-labeled plays
- `def_man_rate` = 1 − `def_zone_rate`
- `def_light_box_rate` (`defenders_in_box` ≤ 6), `def_heavy_box_rate` (≥ 8)
- shell rates: COVER_1, COVER_2, COVER_3, COVER_4, COVER_6, 2_MAN (reported; only
  man/zone and box enter the frozen candidate)

**No sportsbook variable enters this study at any stage.** No market column is read.

## 3. Eligibility, frozen thresholds

- player: ≥ **20** prior targets and ≥ **3** prior games with a target;
- defense: ≥ **100** prior coverage-labeled plays;
- week `W` ≥ **4** (so a prior window exists);
- player must have ≥ **1** target in week `W` (the outcome must exist).

## 4. Anomaly gate (AGENTS.md, non-negotiable)

Every player-week is state-labeled **before** entering evidence. A player-week with
zero targets *and* zero offensive participation is `DNP_INACTIVE` and is excluded
from BOTH the prior window and the outcome. A week whose participation is present but
far below the player's own norm is `PARTIAL_GAME` and is excluded from the **outcome**
set while remaining visible in diagnostics. Labels are reported as counts; nothing is
silently dropped. Where participation cannot be resolved the row is
`UNRESOLVED_CONTEXT` and fails closed (excluded from outcomes, reported).

## 5. Models, frozen

Let `ŷ_base` be the player's own prior expectation (`prior_ypt × prior_tgt_per_game`).

- **M0 — player baseline only:** `y ~ ŷ_base`
- **M1 — positional matchup (what production does):**
  `y ~ ŷ_base + position × def_zone_rate + position × def_heavy_box_rate`
- **M2 — player-specific matchup (candidate):** M1 **+** `prior_adot × def_zone_rate`
  **+** `prior_adot × def_man_rate`

Linear, ordinary least squares, no regularization tuning, no feature search. The
feature list above is the entire candidate space. **No feature may be added after
scoring begins.**

## 6. Evaluation, frozen

**Temporal holdout, never fit on what is scored.** Fit on weeks 4–11, score weeks
12–18. One split, fixed now.

**Disjoint-window persistence.** The `prior_adot × def_zone_rate` coefficient is
estimated independently on two **non-overlapping** week ranges (4–10 and 11–18) and
compared for sign agreement and magnitude. This exists because overlapping as-of
windows previously manufactured persistence in
`PRISM_STRANDED_SIGNAL_DISPERSION_RESULTS_V1` (`rush_stuff_rate` 0.883 → 0.017 once
windows were made disjoint). Sign disagreement across disjoint windows **falsifies H1**
regardless of holdout improvement.

**Player-level primary metric.** Per AGENTS.md, position-aggregate MAE is a guardrail
only. The primary read is per-player: among players whose `prior_adot` is far from
their position's mean (frozen: top and bottom **quartile** of within-position aDOT),
does M2 beat M1 on that player's own held-out games? A win driven only by
position-average error reduction does **not** support H1.

## 7. Decision rule, frozen before scoring

**H1 is supported only if all three hold:**
1. M2 beats M1 on held-out weeks 12–18 on player-level error;
2. the `prior_adot × def_zone_rate` coefficient has the **same sign** in both disjoint
   windows;
3. the improvement is concentrated in the off-mean-aDOT quartiles, not spread evenly
   (which would indicate a position-level effect mislabeled as player-specific).

**If any fails, H1 is NOT supported** and that is reported as the result. No
subgroup will be searched for a rescue, no top-N selection applied, no named player
carved out post hoc, and the study will not be re-specified to find a winner.

## 8. What this study does NOT do

- does not change any production code or promoted model;
- does not touch 2026 projections, pricing, or betting (betting remains CLOSED);
- does not propose a mechanism — a mechanism is only justified if H1 survives;
- does not resolve SLOT unreachability or the `PERCENTY_COLS` scale bug, which remain
  separate open items;
- does not use alignment (LWR/RWR/SWR), which `run_player_form_v2.py` collapses to
  position family. aDOT is used as the available player-specific proxy. If H1 survives,
  restoring alignment becomes the natural follow-on.
