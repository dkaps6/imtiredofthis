# PRISM — Opponent / Environment Transmission Trace V1

**Status:** read-only mechanism trace. **No fitting, no scoring, no promotion.**
**Authorized by:** GPT audit comment `6431984732`,
`WALKER_ENVIRONMENT_TRANSMISSION_RESEARCH_UNBLOCKED_IN_PARALLEL__READ_ONLY_MECHANISM_FIRST`
**Consumes (does not redo):** `PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`
@ `cca416bd2cdad8f142a5a063923149344c7e5c5d`
**Reproduce:** `python scripts/research/trace_opponent_environment_transmission_v1.py`
**Artifacts:** `data/research/opponent_transmission/opponent_environment_transmission_v1{.csv,_status.json}`

Sportsbook inputs used: **0**. Outcomes inspected: **none**. Week 7 used to choose
a coefficient, threshold, sign or winning candidate: **no**.

---

## The question

The accepted decomposition established *that* Seattle's defense produced zero
movement in Kenneth Walker III's generic rushing projection while moving his
receiving projection. It did not establish *why*, nor whether the cause is
specific to Walker.

This trace answers that from production source code, which is durable and
needs no Week-7 data. The finding is therefore **not a Walker finding**. It
holds for every RB in every week.

## Headline

**Opponent run defense cannot reach an RB's rushing mean in the canonical
pipeline.** Not weakly — structurally. Four independent reasons, each verified
in source:

1. **Two of the four ensemble components take no opponent input at all.**
   `ml_v2.FEATURE_COLUMNS` is `hist_games` + `prev_/mean3_/mean5_` of the
   player's own 13 box stats + 5 position dummies. It reads only
   `player`/`team`/`position` from context, and `team` is an identity key for
   dedup. `state_v2` is the same: required set `{player, team, position}`,
   features from the player's own prior game logs. Neither selects columns
   dynamically, so this is architectural absence, not weak signal.

2. **`def_rush_epa` is carried onto the rule object and then ignored.**
   `context_bridge_v3` populates 17 `TeamContext` fields; rules read 13.
   The four carried-and-ignored are `def_rush_epa`, `def_pass_epa`,
   `explosive_play_rate_allowed`, `proe`. The data sits on the very object
   `matchup_multipliers()` receives and no rule reads it.

3. **The most RB-specific opponent signals in the repository are referenced by
   no production module whatsoever** — not carried, not merged, not read:
   `rush_stuff_rate` (runs killed at the line),
   `yards_before_contact_per_rb_rush` (the clearest run-fit measure available),
   `ypt_allowed_rb` / `_slot` / `_te` / `_wr` / `_outside`. Built by the Sharp
   Football provider into `team_form.csv` and `sharp_team_form.csv`, stored,
   and never consumed.

4. **The one live channel is a near-binary gate.** The only
   opponent→RB-rushing-efficiency path in production is
   `rules_v2.matchup_multipliers → rb_rush_eff_mult`, applied at
   `simulation_rules.py:231` as `rules_ypc = base_ypc * rb_rush_eff_mult`:

   ```python
   if light >= 0.60: values["rb_rush_eff_mult"] *= 1.07
   if heavy >= 0.60: values["rb_rush_eff_mult"] *= 0.94
   ```

   Neither branch fires → `1.0` → the opponent has exactly no effect.

## Why receiving moved and rushing did not

Same player, same opponent, same week. The asymmetry is **gate reachability**,
not football.

| | rushing | receiving |
|---|---|---|
| rule input | `light_box_rate`, `heavy_box_rate` | `coverage_zone_rate`, pressure mismatch |
| threshold | `>= 0.60` | `>= 0.60` zone, `> 0.075` pressure |
| fired in Week 7 | no → `1.0` | **yes** → `rb_rec_target_mult *= 1.20` |

A zone rate at or above 60% is an ordinary NFL defensive profile. A *box* rate
at or above 60% is not. The receiving gate sits inside the league distribution;
the rushing gates sit at its edge.

## The QB/RB asymmetry

The promoted QB route consumes its opponent signal. The RB route does not.

| signal | on `TeamContext` | read by a rule | live consumer | net |
|---|---|---|---|---|
| `def_pass_epa` | yes | no | `qb_pass_synthesis_v1` (promoted M89/M90) | **LIVE for QB** |
| `def_rush_epa` | yes | no | only `pricing_v2` (orphaned, below) | **DEAD for RB** |
| `explosive_play_rate_allowed` | yes | no | builder only | **DEAD** |
| `proe` | yes | no | `simulation_v2` (volume) | **LIVE** |

This is the cleanest statement of the gap: the QB has an opponent-aware mean;
the RB's opponent-aware mean does not exist.

## An orphaned mechanism, not an absent one

`scripts/pricing_v2.py:125-131` already contains the missing adjustment:

```python
rush_epa = _num(row, "def_rush_epa_opp", "opp_def_rush_epa", "def_rush_epa")
if market in {"rush_yards", "rush_att", "rush_rec_yards"} and np.isfinite(rush_epa):
    mult *= 1.0 + float(np.clip(rush_epa, -0.30, 0.30)) * 0.25
```

It is applied by `pricing_v2._projection` (line 180). `run_pricing_v2.py`
imports **only** `_fair_market_prob` and `_fair_odds` from that module, and
`full-slate.yml` never references `pricing_v2`. So the mechanism is written and
signed but unwired.

This changes what the fix is: an **integration** question, not a modeling one.
It is **not** endorsed here. That coefficient was never validated on the
promoted stack, and per the authorizing audit it must not be scored on Week 7.

## Candidate families, separated by football meaning

The authorizing audit requires these be kept distinct: "rush EPA, stuff/YBC,
explosive suppression, box structure, expected game state, receiving coverage
are not interchangeable." They are not, and the reason is that **they do not
belong in the same channel.**

| family | football question | channel | avail | consumed | stranded |
|---|---|---|---|---|---|
| A run fit / box structure | Is there room at the first level? | opportunity + efficiency | 12 | 3 | 9 |
| B run stop at the line | Does this defense kill runs before they start? | efficiency / **mean** | 6 | 1\* | 5 |
| C explosive suppression | Does it surrender the long one? | **sigma / right tail** | 2 | 1\* | 1 |
| D RB receiving coverage | Does it concede the back out of the backfield? | receiving opportunity + efficiency | 8 | 2 | 6 |
| E pass rush / blitz | Does pressure force check-downs? | receiving opportunity + pass efficiency | 10 | 3 | 7 |
| F expected game state | How many snaps, how pass-leaning? | opportunity only | 39 | 13 | 26 |
| G QB-facing pass defense | How well does it defend the pass? | QB mean (promoted route) | 11 | 4 | 7 |

\* carried onto `TeamContext` but read by no rule; see the table above.

108 opponent columns reach the player row; 88 carry football signal; **61 are
stranded.**

**A single "defense scalar" would be wrong on football grounds, not merely
methodological ones.** C belongs in sigma — a defense that surrenders few
explosive runs compresses the right tail; routing that into the mean would
mis-model the distribution and is plausibly connected to the documented P3
high-end ceiling-compression limitation. F belongs in opportunity. B belongs in
the mean. One scalar cannot be simultaneously correct for three channels.

## Not established

- **Gate firing frequency.** How often `light/heavy_box_rate >= 0.60` holds is
  *not* measured here. Production does persist the multiplier
  (`artifact_contracts.py:38` `rush_eff_mult`; `simulation_rules.py:236`
  `rules_rush_eff_mult`), but the Week-7 preservation filter kept only the
  column *inventory* and dropped every opponent value and the multiplier
  itself. Answerable from the next live run or a `team_form` rebuild; **not**
  from the durable archive. This is a §5a gap in my own preservation script,
  not in production.
- **Accuracy effect.** Whether consuming any stranded family would improve a
  projection is untested and must not be scored on Week 7.

## Method validity

Consumption is established by literal column-name reference. That is sound
only while the learned components declare features statically, so the script
re-checks the invariant on every run and **fails closed** if `ml_v2` stops
declaring `FEATURE_COLUMNS` as an explicit tuple, if dynamic selection appears,
or if any module constructs an `_opp` name at runtime.

Two defects in the first pass of this trace were found and fixed:
opponent columns also arrive as `_opp_x`/`_opp_y` merge-collision variants, so
matching only `_opp` dropped `coverage_zone_rate` and misreported receiving
coverage as fully stranded; and omitting the promoted QB modules made family G
read as stranded. Both are corrected in the committed script.

## Explicitly separate

The zero-history / position-prior entitlement defect is a **distinct cause**
with its own trace and disposition, per the authorizing audit. It is not folded
into this trace. Inside-5 TD opportunity quality likewise remains design-only
and unscored; its negative evidence stands (correlation with red-zone 0.890,
the clip is not the main compression problem).

---

## Does the pattern generalize by role? Yes — in a specific shape

Same read-only method, same source. `rules_v2.matchup_multipliers` declares
exactly ten multipliers. Classified by what they move:

| multiplier | role | channel | opponent input | threshold |
|---|---|---|---|---|
| `wr1_target_mult` | WR1 | opportunity | `coverage_man_rate` | `>= 0.50` |
| `wr1_5_target_mult` | WR1.5 | opportunity | `coverage_man_rate` | `>= 0.50` |
| `slot_target_mult` | slot | opportunity | `coverage_man_rate`, `middle_open_rate` | `>= 0.50` |
| `te_target_mult` | TE | opportunity | `coverage_zone_rate`, `middle_open_rate` | `>= 0.60`, `>= 0.50` |
| `rb_rec_target_mult` | RB rec | opportunity | `coverage_zone_rate`, pressure | `>= 0.60`, `> 0.075` |
| `rb_rush_eff_mult` | RB rush | **efficiency** | `light/heavy_box_rate` | `>= 0.60` |
| `pass_eff_mult` | team pass | efficiency | pressure mismatch | `> 0.075` |
| `sack_mult`, `int_mult` | QB | event rates | pressure mismatch | `> 0.075` |
| `volatility_mult` | **sigma** | distribution | pressure mismatch | `> 0.075` |

Three things follow, and they are the generalization:

**1. Opponent reaches receiving OPPORTUNITY for every receiving role.** Each of
WR1, WR1.5, slot, TE and RB-receiving has at least one coverage-structure gate,
at thresholds of 0.50–0.60 that sit inside ordinary NFL defensive profiles. This
channel works.

**2. Opponent reaches per-role EFFICIENCY for essentially no one.** There is no
`wr_eff_mult`, no `te_eff_mult`, no yards-per-target multiplier for any
receiver. The only role-specific efficiency multiplier in the file is
`rb_rush_eff_mult` — the dead box gate. And the signals that would drive the
missing ones, `ypt_allowed_wr` / `_slot` / `_te` / `_outside`, are referenced by
**no production module at all**. The efficiency inputs exist for all four
receiving roles and not one is consumed.

**3. Opponent reaches SIGMA only through pressure.** `volatility_mult` is the
single distribution-shape channel and it is gated solely on pressure mismatch.
`explosive_play_rate_allowed` — the natural tail input — is carried onto
`TeamContext` and read by no rule.

So the shape of the gap is consistent across positions:

> **Production models how much opportunity the opponent concedes. It almost
> never models how well that opportunity converts, and it shapes the tail from
> pressure alone.**

Walker is not a special case. He is the clearest case, because rushing yards is
the one market whose *opportunity* channel is finite-allocated upstream and
whose *efficiency* channel is the one dead gate — so for him both halves fail at
once.

This remains read-only. No candidate here is scored, ranked or promoted, and
nothing in this section proposes a threshold or coefficient.
