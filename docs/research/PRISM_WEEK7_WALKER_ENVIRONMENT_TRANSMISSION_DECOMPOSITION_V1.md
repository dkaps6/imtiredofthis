# PRISM Week-7 Kenneth Walker III Environment-Transmission Decomposition V1

**Date:** 2026-10-26  
**Lane:** read-only architecture / forensic decomposition  
**Scientific status:** **OPEN — NOT A PREDICTIVE LOCK**  
**Outcome use:** **0**  
**Sportsbook reacquisition:** **0**  
**New OddsAPI calls:** **0**  
**Production changes:** **0**

## 1. Scope and authority

This document is the first requested generic-skill transmission decomposition under the permanent PRISM player-first rule:

> THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.

> ANOMALIES ARE QUESTIONS, NOT TRENDS.

It uses the exact already-paid Week-7 production artifact only. It does **not** use Week-7 outcomes to fit, select, rescue, or promote anything.

Canonical paid-board lineage:

- workflow run: `39849371821`
- artifact: `12456298985`
- artifact digest: `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- artifact head: `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- target: 2026 Week 7
- player: Kenneth Walker III
- team/opponent in the canonical 2026 slate: KC vs SEA

The purpose is architecture tracing only:

1. current-player / neutral-opponent baseline;
2. legal pregame opponent inputs physically available;
3. fields actually consumed by production;
4. fields collected but dropped or stranded;
5. opponent-caused opportunity movement;
6. opponent-caused efficiency movement;
7. opponent-caused mean movement;
8. opponent-caused sigma movement;
9. rushing-vs-receiving defensive asymmetry;
10. player-first opportunity-state defects revealed by the same decomposition.

## 2. Exact as-run Walker state

From `data/football_simulation_universe.csv` and `data/model_rule_simulation_inputs.csv`:

| State | Value |
|---|---:|
| player | Kenneth Walker III |
| position / depth role | RB / RB1 |
| prior games | 17 |
| current games | 5 |
| blended rush share | 0.599862 |
| Bayesian rush share | 0.524947 |
| rules rush share | 0.524947 |
| blended YPC | 5.550684 |
| Bayesian YPC | 4.959611 |
| rules YPC | 4.959611 |
| rules plays estimate | 56.932121 |
| rules pass rate | 0.570000 |
| rules rush-efficiency multiplier | 1.000000 |
| rules volatility multiplier | 1.000000 |

From `outputs/props_priced_clean.csv`:

| Component | Walker rush yards |
|---|---:|
| MC projection | 40.568542 |
| ML projection | 98.073554 |
| State projection | 58.685803 |
| final ensemble projection | 66.045893 |
| final model SD | 28.097688 |

Frozen `rush_yards` ensemble weights in the same artifact:

- MC = `0.5569542426070742`
- ML = `0.4430457573929258`
- State = `0.0`

Therefore Walker's final Week-7 rush-yards mean is exactly the calibrated MC/ML blend. The state-model number is diagnostic only for this market because its weight is zero.

## 3. Pregame Seattle defensive information that was available

The paid artifact had substantially more Seattle information than production used for Walker rushing.

### 3.1 Reached TeamContext

From `data/team_context_v3.csv`:

| SEA field | Value | Provenance state |
|---|---:|---|
| def_rush_epa | -0.202932 | `team_form_legacy_merged`; provider audit still needed |
| explosive_play_rate_allowed | 0.073034 | guarded current-2026 PBP |
| light_box_rate | 0.497 | `team_form_legacy_merged`; provider audit still needed |
| heavy_box_rate | 0.164 | `team_form_legacy_merged`; provider audit still needed |
| success_rate_def | 0.367978 | guarded current-2026 PBP |
| coverage_zone_rate | 0.892 | current-slate provider semantics pending audit |
| coverage_man_rate | 0.056 | current-slate provider semantics pending audit |
| pressure_rate_generated | 0.209069 | M89/M90 promoted rolling-8 |
| def_pass_epa_allowed | -0.205481 | M89/M90 promoted rolling-8 |
| def_pass_success_allowed | 0.375842 | M89/M90 promoted rolling-8 |
| def_ypa_allowed | 4.753773 | M89/M90 promoted rolling-8 |

These being present does **not** mean every source is scientifically qualified. This table is an availability / code-path statement only.

### 3.2 Present in TeamForm / metrics but dropped before generic TeamContext/rules consumption

Walker's exact pricing row also carried:

| SEA opponent field | Value |
|---|---:|
| ypt_allowed_rb | 5.1 |
| ypt_allowed_wr | 6.2 |
| ypt_allowed_te | 4.4 |
| ypt_allowed_outside | 7.7 |
| ypt_allowed_slot | 4.2 |
| yards_before_contact_per_rb_rush (two collected variants) | 0.7 / 0.7 |
| rush_stuff_rate (two collected variants) | 24.5% / 19.1% |

These fields were available upstream in the paid artifact but are not consumed by the generic Walker rushing projection.

## 4. What production actually consumed for Walker rushing

The current generic rule path is:

`context_bridge -> TeamContext -> project_game_script / matchup_multipliers -> simulation -> calibrated ensemble`

### 4.1 Game script

`project_game_script()` consumes:

- KC offensive success rate;
- SEA defensive success rate;
- pass-rush/protection pressure mismatch;
- KC offensive pace / plays inputs.

For KC vs SEA:

- KC success_rate_off = `0.493421`
- SEA success_rate_def = `0.367978`
- success differential = `+0.125444`

That produces diagnostic script probabilities of approximately:

- lead = `0.919135`
- neutral = `0.074781`
- trail = `0.006084`

But the generic function then hardcodes:

- `pass_share = 0.57`

and the joint simulator consumes `rules_pass_rate = 0.57`.

**Critical architecture fact:** the very strong modeled lead state does not change the generic pass/rush split. The opponent success-rate information reaches script diagnostics but is stranded before Walker's rush opportunity.

### 4.2 Rushing matchup multiplier

Generic RB rushing efficiency reacts only to two threshold rules:

- if opponent light-box rate >= 0.60: multiply YPC by 1.07;
- if opponent heavy-box rate >= 0.60: multiply YPC by 0.94.

Seattle:

- light box = `0.497`
- heavy box = `0.164`

Neither gate fires.

Therefore:

- `rules_rush_eff_mult = 1.000000`
- Walker Bayesian YPC `4.959611` remains rules YPC `4.959611`.

The following available Seattle signals do **not** move generic Walker rushing:

- `def_rush_epa = -0.202932`;
- `explosive_play_rate_allowed = 0.073034`;
- RB YPT allowed = `5.1`;
- YBC allowed = `0.7`;
- stuff rate = `24.5% / 19.1%`.

An old helper in `scripts/pricing_v2.py` does contain a direct `def_rush_epa` matchup multiplier. That helper is **not the live projection authority**. Current production imports only `_fair_market_prob` and `_fair_odds` from that module; the live football mean is built through `run_pricing_v2.py`, canonical rules, joint simulation, and the full-roster production wrappers.

## 5. Exact Walker rushing opponent-movement result

The other rush-yards ensemble arms are opponent-blind:

- ML V2 uses lagged player history + position indicators only;
- State V2 uses player outcome-state transitions only;
- Bayesian V2 uses position population prior + prior/current player evidence only.

For the MC arm, Seattle changes neither Walker rushing opportunity nor Walker rushing efficiency under the currently active rules:

- opponent-caused pass/rush split movement = **0**
- opponent-caused rush-share movement = **0**
- opponent-caused YPC multiplier movement = **0**
- opponent-caused volatility multiplier movement = **0**

So, under the current generic architecture, replacing Seattle with a neutral opponent that activates none of the box thresholds leaves the Walker rush-yards path unchanged.

### Read-only neutral-opponent counterfactual

| Quantity | As-run SEA | Neutral generic opponent | Opponent movement |
|---|---:|---:|---:|
| rules pass rate | 0.570000 | 0.570000 | 0 |
| rules rush share | 0.524947 | 0.524947 | 0 |
| rules YPC | 4.959611 | 4.959611 | 0 |
| rules rush-eff multiplier | 1.000000 | 1.000000 | 0 |
| rules volatility multiplier | 1.000000 | 1.000000 | 0 |
| MC rushing mean authority | unchanged | unchanged | 0 |
| ML rush projection | 98.073554 | 98.073554 | 0 |
| final rush-yards mean | 66.045893 | 66.045893 | **0** |
| final model SD | 28.097688 | 28.097688 | **0** |

This is an **architecture diagnosis**, not evidence that Seattle's defensive inputs should receive any particular fitted coefficient.

## 6. Rushing-versus-receiving defensive asymmetry

The same Seattle defense **did** alter Walker's receiving opportunity.

Walker's receiving state:

- Bayesian target share = `0.104325`
- SEA zone rate = `0.892`
- generic rule: if zone >= 0.60, RB receiving target multiplier = `1.20`
- rules target share = `0.125190`

So the current generic architecture applies a +20% rule-layer target-share change to Walker receiving while applying **zero** matchup change to his rushing share or rushing efficiency.

Pressure does not add another receiving adjustment here:

- SEA pressure generated = `0.209069`
- KC pressure allowed = `0.183234`
- mismatch = about `+0.025835`
- rule gate requires `> +0.075`

Therefore that gate does not fire.

The paid final outputs were:

- rec_yards MC = `15.646946`
- rec_yards ML = `18.581081`
- rec_yards State = `16.413248`
- rec_yards final = `16.541507`
- rush_rec_yards final = `82.587400`
  - rush component = `66.045893`
  - receiving component = `16.541507`

This asymmetry is real in production: Seattle coverage changes Walker receiving opportunity; Seattle run-defense quality does not change Walker rushing in this case.

## 7. Separate player-first opportunity problem exposed by Walker

The environment audit also exposed a more serious player-identity / role-state problem that is **not** caused by Seattle.

The full football universe has KC raw Bayesian rushing-share mass:

- all-player raw rush-share sum = `1.800438`

Canonical rushing simulation keeps only the top five rushing shares, whose Week-7 KC sum was:

- top-five raw sum = `1.496822`

The simulator then scales those five shares to a maximum modeled mass of `0.95`.

Top five:

| Player | Role | Current games | Prior games | Bayesian rush share | Effective allocation probability | Analytic expected carries* |
|---|---|---:|---:|---:|---:|---:|
| Kenneth Walker III | RB1 | 5 | 17 | 0.524947 | 0.333173 | 8.156 |
| Ben Vansumeren | FB1 | 0 | 0 | 0.359568 | 0.228210 | 5.587 |
| Justin Fields | QB2 | 0 | 9 | 0.220009 | 0.139635 | 3.418 |
| Emmett Johnson | RB2 | 5 | 0 | 0.216735 | 0.137557 | 3.368 |
| Brashard Smith | RB3 | 1 | 15 | 0.175563 | 0.111426 | 2.728 |

\*Using the rules mean of `56.932121` plays and hardcoded `0.57` pass rate; these are analytic multinomial expectations, not a replay of the sampled 25k draws.

The most important row is Ben Vansumeren:

- current games = `0`;
- prior games = `0`;
- raw player rush share = missing;
- Bayesian rush share = `0.359568`.

Why: Bayesian V2 maps FB into the RB position group. With no player evidence, the position-group prior can become the player's rush-share estimate. That estimate then enters the finite rushing-room competition as if it were player-specific entitlement.

This is a direct player-first architecture concern:

> a position prior is currently capable of assigning a zero-history player a large opportunity share, and that share can materially suppress a named RB1.

Justin Fields creates a second role-state concern:

- role = QB2;
- current games = 0;
- prior games = 9;
- Bayesian rush share = `0.220009`;
- that historical share enters current KC rushing-room competition despite no current-game evidence and without a target-week role/participation allocation layer.

The result for Walker is:

1. blended player-form rush share: `0.599862`;
2. Bayesian shrinkage: `0.524947`;
3. current generic room normalization: effective allocation probability `0.333173`.

At the rules mean team volume this is only about `8.16` analytic expected carries before sampled efficiency noise.

This finding is outcome-blind and was visible from the pregame artifact alone.

## 8. Why this matters for Stage-2

Qualified conditional allocation research exists but is not wired to production.

This decomposition does **not** assume Stage-2 automatically fixes Walker. It establishes why the already-assigned frozen Stage-2 readiness re-probe is materially relevant:

- the current generic rushing room is not sufficiently player-/role-conditional;
- zero-history position priors can enter as player entitlement;
- inactive/backup/current-role state can fail to constrain historical opportunity;
- the fallback top-five normalization is doing large corrective work after individually estimated shares already exceed a coherent room total.

Any future production bridge must still pass the already-frozen conditional allocation gates and must remain football-only / strictly pregame.

## 9. Disposition

### Opponent/environment transmission

**OPEN — architecture gap confirmed.**

For Kenneth Walker III Week 7:

- Seattle run-defense information existed;
- much of it was dropped or stranded;
- the generic rushing opponent adjustment was exactly a no-op;
- opponent-caused rush opportunity movement = 0;
- opponent-caused rush efficiency movement = 0;
- opponent-caused final mean movement = 0;
- opponent-caused final sigma movement = 0;
- receiving did receive an opponent coverage adjustment.

This does **not** promote a new defensive coefficient and does **not** fit Week 7.

### Player identity / role / allocation

**OPEN — concrete pregame architecture defect exposed.**

The zero-history-FB and zero-current-game-QB2 competition is exactly the type of player-first state error the Player Identity Card + qualified allocation work is intended to make visible before projection.

### Betting construction

Separate lane. Nothing in this document changes the accepted finding that the Week-7 parlay result was primarily a construction / threshold / exposure failure.

## 10. Exact next audit actions

1. Claude reruns the already-frozen Week-7 Stage-2 **source-readiness** contract; no refit, no outcome read, no odds call.
2. Correct Player Identity Card V1 around participation denominators, target-week team/roster authority, sparse/no-history players, and stable identity.
3. Add a read-only rushing-room trace that exposes, per player:
   - pre-Bayes opportunity state;
   - Bayesian opportunity state;
   - target-week role / participation state;
   - top-N inclusion;
   - room-normalization factor;
   - final allocation probability;
   - residual bucket;
   - expected team opportunity.
4. Repeat the generic transmission decomposition on representative WR/TE cases (JSN vs Kupp, then Barner) without using Week-7 outcomes for candidate selection or fitting.
5. Keep betting-construction V2 downstream and separate.
