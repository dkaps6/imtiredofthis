# PRISM — Opponent / Environment Transmission Trace V1

**Status:** read-only mechanism trace. **No fitting, no scoring, no promotion.**
**Authorized by:** GPT audit `6431984732` (read-only mechanism-first).
**Revised per:** GPT re-audits `6432370310` and `6432388032`
(`WALKER_EXACT_CASE_CLEARED__SOURCE_GENERALIZATION_DIRECTIONALLY_VALID_BUT_REPORTING_SEMANTICS_NEED_CORRECTION`;
`WALKER_W7_TRANSMISSION_CLEARED__ROLE_GENERALIZATION_PARTIAL_CLEAR_REQUIRES_SCOPE_AND_DATAFLOW_CORRECTION`).
**Consumes (does not redo):** `PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`
@ `cca416bd2cdad8f142a5a063923149344c7e5c5d`
**Reproduce:** `python scripts/research/trace_opponent_environment_transmission_v1.py`
**Artifacts:** `data/research/opponent_transmission/opponent_environment_transmission_v1{.csv,_status.json}`

Sportsbook inputs used: **0**. Outcomes inspected: **none**. Week 7 used to choose
a coefficient, threshold, sign or winning candidate: **no**.

---

## What is CLEARED, and what this document does not claim

**Cleared by both re-audits — the Walker Week-7 exact case.** For the frozen W7
generic rushing path: Seattle's rush-defense context existed; `def_rush_epa` did
not reach Walker's rushing mean; explosive suppression was carried but not
consumed; RB-specific YPT/YBC/stuff are not wired into the generic player
projection; neither 0.60 box gate fired; so opponent-caused opportunity,
efficiency, mean and sigma movement were exactly **0**; receiving did move
through the zone-coverage target-share rule.

**Also cleared** — the generic RB rushing path has a very thin opponent
interface, and major available run-defense families are stranded from it.

**Six claims in the first version of this document were too broad and have been
narrowed.** They are listed in *Corrections* at the end, because a reader who
saw V1 needs to know precisely what changed and why.

---

## Four stages, not two

A column referenced by the module that *copies* it is not a column a projection
*uses*. V1 of the trace script conflated these and reported `def_rush_epa` and
`explosive_play_rate_allowed` as "consumed" on the strength of a
`context_bridge_v3` reference. That was a reporting bug, caught by audit; the
script's own `teamcontext_transmission()` had it right all along.

The trace now reports four stages:

1. **present on the player row** — the column reaches `metrics_ready.csv`;
2. **bridged by transport** — `context_bridge_v3` copies it onto `TeamContext`;
3. **referenced by a consumer** — a literal reference in a rule, learned
   component, simulator or promoted adapter;
4. **read by a rule** — for `TeamContext` fields, whether any rule reads the
   attribute.

The sharpest category is **transported-but-stranded**: carried onto the very
object `matchup_multipliers()` receives, then read by no rule. Exactly two
football signals are in it:

| signal | populated onto `TeamContext` | read by any rule | live consumer elsewhere |
|---|---|---|---|
| `def_rush_epa` | yes | **no** | only `pricing_v2` (orphaned, below) |
| `explosive_play_rate_allowed` | yes | **no** | none |
| `def_pass_epa` | yes | no | `qb_pass_synthesis_v1` (promoted) |
| `proe` | yes | no | `simulation_v2` (volume) |

`context_bridge_v3` populates 17 `TeamContext` fields; rules read 13.

## Counts are inventory, not certified dataflow

108 opponent columns reach the player row; 88 carry football signal; **63 are
not literally referenced by any consumer module.**

**These are literal-reference inventory counts and carry no causal claim.** A
base-name hit does not prove the opponent-suffixed value is the one used — a
same-named offensive or player-side field produces an identical hit. `proe` is
the clearest case: a literal `proe` reference in `simulation_v2` is not evidence
that `proe_opp` moved a player projection. The certified dataflow findings are
the `TeamContext` populated-vs-read comparison above, which is exact.

## What the source does and does not establish about RB rushing

**It does establish a live channel.** `rules_v2.matchup_multipliers →
rb_rush_eff_mult`, applied at `simulation_rules.py:231` as
`rules_ypc = base_ypc * rb_rush_eff_mult`:

```python
if light >= 0.60: values["rb_rush_eff_mult"] *= 1.07
if heavy >= 0.60: values["rb_rush_eff_mult"] *= 0.94
```

This **can** move the MC rushing mean, and therefore the generic ensemble,
whenever MC carries weight. It is a real mechanism, not a dead one.

**What is established is narrower:**

> Outside the light/heavy-box gate, the generic RB rushing mean has no opponent
> run-defense transmission. `def_rush_epa`, explosive suppression, YBC/stuff and
> RB-specific allowance fields do not reach it through their natural channels.

For Walker in Week 7 the gate did not fire, hence exact zero movement. **How
often it fires league-wide is NOT ESTABLISHED** (see *Not established*), so this
document does not call the gate dead, does not describe it as near-binary in any
population-severity sense, and makes **no every-RB/every-week claim**.

## Two of three generic ensemble components are opponent-blind

`scripts/modeling/ensemble_v2.py` declares `COMPONENTS = ("mc_proj", "ml_proj",
"state_proj")` and states that Bayesian v2 is intentionally **not** a fourth
independent projection because it is already inside the MC baseline. (V1 of this
document said "2 of 4"; the correct figure is **2 of 3**.)

- `ml_v2.FEATURE_COLUMNS` = `hist_games` + `prev_/mean3_/mean5_` of the player's
  own 13 box stats + 5 position dummies. It reads only `player`/`team`/`position`
  from context, and `team` is an identity key for dedup.
- `state_v2` is the same: required `{player, team, position}`, features built
  from the player's own prior game logs.

Neither selects columns dynamically, so this is architectural absence rather
than weak signal. **The MC component can express opponent effects** where a
canonical rule exists and fires.

## Why receiving moved and rushing did not, for Walker

Same player, same opponent, same week. Rushing depended on a box gate that did
not fire; receiving depended on `coverage_zone_rate >= 0.60`, which did. This is
a statement about **which route was wired and which condition held in that
game** — not about how commonly either condition holds.

## Role generalization, corrected

`rules_v2.matchup_multipliers` declares exactly ten multipliers:

| multiplier | role | channel | opponent input | threshold |
|---|---|---|---|---|
| `wr1_target_mult` | WR1 | opportunity | `coverage_man_rate` | `>= 0.50` |
| `wr1_5_target_mult` | WR1.5 | opportunity | `coverage_man_rate` | `>= 0.50` |
| `slot_target_mult` | slot | opportunity | `coverage_man_rate`, `middle_open_rate` | `>= 0.50` |
| `te_target_mult` | TE | opportunity | `coverage_zone_rate`, `middle_open_rate` | `>= 0.60`, `>= 0.50` |
| `rb_rec_target_mult` | RB rec | opportunity | `coverage_zone_rate`, pressure | `>= 0.60`, `> 0.075` |
| `rb_rush_eff_mult` | RB rush | efficiency | `light/heavy_box_rate` | `>= 0.60` |
| `pass_eff_mult` | **shared** pass | efficiency | pressure mismatch | `> 0.075` |
| `sack_mult`, `int_mult` | QB | event rates | pressure mismatch | `> 0.075` |
| `volatility_mult` | sigma | distribution | pressure mismatch | `> 0.075` |

**1. Receiving opportunity has role-specific opponent gates — wiring cleared.**
Every receiving role *represented by the canonical matchup multipliers* has at
least one coverage-structure gate. That is not every receiving **player**:
`_wr_role_labels()` labels slot WRs plus only the top two perimeter WRs by target
share, so **WR3 and below receive no WR target multiplier at all**. Whether these
gates fire often, or add predictive value, is not established here.

**2. Receiving efficiency is not opponent-blind; it lacks a ROLE-SPECIFIC
channel.** Pressure mismatch sets `pass_eff_mult` to 0.98 or 1.01, and
`simulation_rules.py:230` applies `rules_ypt = base_ypt * pass_eff_mult`. So
opponent context **does** change receiving YPT through a shared channel. What
production lacks is any role-specific conversion channel from the available
`ypt_allowed_wr` / `_slot` / `_te` / `_rb` / `_outside` signals — which are
referenced by no production module at all. The correct statement:

> No receiving role has a role-specific opponent efficiency channel; receiving
> YPT gets only the shared pressure-based `pass_eff_mult`.

**3. Sigma: the only DIRECT EXPLICIT volatility multiplier in the generic skill
path is pressure-driven, and `explosive_play_rate_allowed` is not used as a
direct tail input.** Opponent context can still change final `model_sd`
indirectly — `simulation_v2.py:150` computes
`rec_sd = max(6.0, sqrt(max(targets,1)) * ypt * 0.55) * vol_mult`, where `ypt` is
`rules_ypt` and `targets` are multinomial draws, so a change to target shares or
to `pass_eff_mult` moves realized variance without touching `volatility_mult`.
Separately, the **promoted QB C2 path is its own opponent-aware distribution
route**: `run_qb_distribution_state_context.py` builds `pass_efficiency_spot` and
`rush_efficiency_spot` from opponent `def_prior_*` allowance fields, and
`model/qb_distribution_state_selector_v1.json` consumes them in its six-feature
contract (`pass_opportunity_spot`, `pass_efficiency_spot`,
`rush_opportunity_spot`, `rush_efficiency_spot`, `pred_qb_attempts`, `week`).
This generic-skill trace does not overwrite that promoted architecture.

### The shape of the gap

Stated only as strongly as the source supports:

> Production provides role-specific opponent channels for receiving
> **opportunity**, a single shared pressure channel for passing **efficiency**,
> and one coarse box-threshold channel for RB rushing **efficiency**. It
> provides **no role-specific opponent conversion-efficiency channel** from the
> `ypt_allowed_*` family, and no direct tail/sigma input from explosive
> suppression.

## An orphaned mechanism, not an absent one

`scripts/pricing_v2.py:125-131` already contains a rush-defense adjustment:

```python
rush_epa = _num(row, "def_rush_epa_opp", "opp_def_rush_epa", "def_rush_epa")
if market in {"rush_yards", "rush_att", "rush_rec_yards"} and np.isfinite(rush_epa):
    mult *= 1.0 + float(np.clip(rush_epa, -0.30, 0.30)) * 0.25
```

Applied by `pricing_v2._projection`. `run_pricing_v2.py` imports **only**
`_fair_market_prob` and `_fair_odds`, and `full-slate.yml` never references
`pricing_v2`. So it is **not canonical** and is **not endorsed**: the coefficient
was never validated on the promoted stack and must not be scored on Week 7.
Flagged because it changes what a fix would involve.

## Not established

- **Gate firing frequency**, for any gate. Production persists the multiplier
  (`artifact_contracts.py:38` `rush_eff_mult`; `simulation_rules.py:236`
  `rules_rush_eff_mult`), but the Week-7 preservation filter kept only the
  column *inventory* and dropped every opponent value and the multiplier.
  Answerable from a legal `team_form` rebuild or a future archived slate; not
  from the durable archive. A §5a gap in my own preservation script, not in
  production.
- **Whether any threshold is "reachable" or "ordinary"** in the league
  distribution. Source code cannot support a population-frequency statement.
- **Predictive value** of consuming any stranded family. Untested, and must not
  be scored on Week 7.

## Method validity

Consumption is established by literal reference, which is valid only while the
learned components declare features statically. The script re-checks and **fails
closed** if `ml_v2` stops declaring `FEATURE_COLUMNS` as an explicit tuple, if
dynamic selection appears, or if any module constructs an `_opp` name at runtime.

## Corrections to V1 of this document

All six were raised by re-audit and all six were verified against source before
acceptance.

| # | V1 claim | corrected to |
|---|---|---|
| 1 | `def_rush_epa` "consumed" in the player path | **transported-but-stranded**; transport and consumption are now separate stages |
| 2 | "2 of 4 ensemble components" | **2 of 3**; `COMPONENTS` is `(mc, ml, state)` and Bayesian v2 is inside MC |
| 3 | "opponent run defense cannot reach an RB rushing mean, every RB every week"; "dead" gate | the box gate **is** a live mean channel; the claim is narrowed to *outside* the gate, and no frequency/severity wording remains |
| 4 | "opponent reaches per-role efficiency for essentially no one" | receiving YPT **does** move via shared `pass_eff_mult`; what is absent is a **role-specific** channel |
| 5 | "opponent reaches sigma only through pressure" | only the **direct explicit** multiplier is pressure-driven; sigma also moves indirectly, and promoted QB C2 is a separate opponent-aware route |
| 6 | "0.50–0.60 thresholds sit inside ordinary NFL profiles" | removed; frequency is not measured, so no population claim is made |

Two defects I found in my own first pass, before audit: opponent columns also
arrive as `_opp_x`/`_opp_y` merge-collision variants, so matching only `_opp`
dropped `coverage_zone_rate` and misreported receiving coverage as fully
stranded; and omitting the promoted QB modules made family G read as stranded.

## Explicitly separate

The zero-history / position-prior entitlement defect is a **distinct cause** with
its own trace and disposition. Inside-5 TD opportunity quality remains
design-only and unscored; its negative evidence stands (correlation with
red-zone 0.890; the clip is not the main compression problem).
