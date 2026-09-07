# CROSS-POSITION PHASE D — GAME-STATE CONSERVATION + PLAYER ENVIRONMENT SENSITIVITY V1

## Purpose
Continue the frozen catastrophic-player forensic chain without broad feature fishing. Phase D asks two predeclared questions using only evidence already produced in Phases A-C:

1. **Same-game conservation:** do large QB/WR/TE/RB errors co-move in a way consistent with one wrong latent pass-vs-run opportunity state?
2. **Player environment sensitivity:** which established, high-opportunity players repeatedly amplify favorable environments, resist adverse environments, or behave largely independent of the generic matchup environment?

This is diagnostic research only. No production coefficient, mean projection, player hierarchy, threshold, sportsbook input, or model architecture is changed by this run.

## Frozen inputs
- Phase C `cross_position_all_rows_game_spot.csv` and `cross_position_catastrophic_game_spot.csv` from the same workflow lineage.
- Phase B `cross_position_pbp_casebook.csv` only for descriptive labels after the Phase-D signatures are computed; no postgame PBP field may be used to construct a pregame environment or player sensitivity predictor.
- Existing frozen model rows: M89/M90 QB, M38/C2 WR baseline rows, TE-R5, RB-P3.

## Integrity rules
- Strict-prior game-spot fields must come unchanged from Phase C.
- No sportsbook features.
- No same-game or future-game information used to construct pregame spot variables.
- Position thresholds remain exactly QB 100, WR 50, TE 40, RB 40 yards.
- Primary QB in a team-game is selected by **largest predicted QB opportunity**, never by actual outcome.
- Player sensitivity eligibility is determined only from sample size and predeclared opportunity status, never from the observed direction of the desired effect.

---

# D1 — Same-game pass/run conservation

## Common cohort
Use team-games where the frozen Phase-C rows contain:
- one primary QB row,
- at least one WR row,
- at least one TE row,
- at least one RB row.

Primary scoreboard: 2024-2025 because the authoritative QB cohort is 2024-2025. Secondary receiver/RB-only diagnostics may use broader overlap but may not replace the primary scoreboard.

## Team-game residuals
All residuals are `actual - predicted`.

- `qb_attempt_resid`: primary QB actual opportunity - predicted opportunity.
- `qb_yard_resid`: primary QB actual passing yards - predicted passing yards.
- `wr_target_resid`: sum of WR actual opportunity - sum of WR predicted opportunity.
- `wr_yard_resid`: sum of WR actual yards - sum of WR predicted yards.
- `te_target_resid`: sum of TE actual opportunity - sum of TE predicted opportunity.
- `te_yard_resid`: sum of TE actual yards - sum of TE predicted yards.
- `rb_carry_resid`: sum of RB actual rush opportunity - sum of RB predicted rush opportunity.
- `rb_rush_yard_resid`: sum of RB actual rush yards - sum of RB predicted rush yards.
- `receiver_target_resid = wr_target_resid + te_target_resid`.
- `receiver_yard_resid = wr_yard_resid + te_yard_resid`.

## Frozen diagnostics
Report Pearson, Spearman, sign-agreement/opposition rates, and bootstrap 95% CIs for:

1. QB attempts vs WR target residual.
2. QB attempts vs TE target residual.
3. QB attempts vs combined WR+TE target residual.
4. QB attempts vs RB carry residual — expected football sign is **negative**.
5. QB pass-yard residual vs WR yard residual.
6. QB pass-yard residual vs TE yard residual.
7. QB pass-yard residual vs combined WR+TE yard residual.
8. QB pass-yard residual vs RB rushing-yard residual — diagnostic only.
9. Combined receiver target residual vs RB carry residual — expected football sign is **negative**.

## Same-game catastrophic co-occurrence
For each team-game, report whether catastrophic misses occurred in the same direction or opposite direction across positions. Primary combinations:
- QB underprojection + WR/TE underprojection.
- QB overprojection + WR/TE overprojection.
- QB underprojection + RB overprojection.
- QB overprojection + RB underprojection.

Also report the inverse combinations so the evidence cannot be cherry-picked.

## Latent game-state signatures
Classify team-games mechanically:
- `PASS_STATE_HIGH`: qb_attempt_resid > 0 AND receiver_target_resid > 0 AND rb_carry_resid < 0.
- `PASS_STATE_LOW`: qb_attempt_resid < 0 AND receiver_target_resid < 0 AND rb_carry_resid > 0.
- `PASS_RECEIVER_SHARED_ONLY`: QB and receiver opportunity residuals share sign but RB does not oppose.
- `RUSH_ONLY`: RB carry residual is material while QB/receiver residuals do not share the opposite sign.
- `MIXED` otherwise.

No threshold optimization is allowed. Materiality summaries may additionally show absolute residual quartiles, but the base signature uses sign only.

---

# D2 — Player environment sensitivity / resistance

## Eligibility
Per position, a player is eligible only if all are true:
- at least **16** evaluated games,
- at least **4 FAVORABLE** games,
- at least **4 ADVERSE** games,
- at least **40%** of evaluated games are opportunity-quartile Q4 or Q3,
- at least two seasons represented where available.

Also publish a stricter `STAR_Q4` table requiring at least **50% Q4** games.

## Frozen player metrics
For each eligible player compute:
- mean signed residual overall, FAVORABLE, NEUTRAL, ADVERSE.
- mean absolute error overall and by spot.
- catastrophic underprojection rate by spot.
- catastrophic overprojection rate by spot.
- `fav_minus_adverse_residual`.
- OLS slope of signed residual on continuous Phase-C game-spot score.
- Spearman residual-vs-spot correlation.
- adverse-spot mean residual.
- adverse-spot catastrophic-underprojection rate.
- favorable-spot catastrophic-underprojection rate.
- first-half-of-career-window slope and second-half-of-career-window slope, split chronologically inside the evaluated sample.
- sign stability of the two slopes.

## Frozen descriptive candidate labels
These labels are **diagnostic**, not promotion rules.

- `ENVIRONMENT_SENSITIVE_CANDIDATE`: favorable-minus-adverse residual >= 25% of that position's catastrophic threshold AND both chronological-half slopes are positive.
- `ENVIRONMENT_RESISTANT_CANDIDATE`: abs(favorable-minus-adverse residual) <= 10% of threshold AND abs(full-sample slope) is in the bottom half of eligible players at that position.
- `ADVERSE_SPOT_CEILING_CANDIDATE`: adverse mean residual >= 0 AND adverse catastrophic-underprojection rate >= the position median among eligible players.
- `FAVORABLE_SPOT_AMPLIFIER_CANDIDATE`: favorable catastrophic-underprojection rate exceeds adverse rate by >= 10 percentage points AND favorable-minus-adverse residual >= 15% of threshold.
- `UNSTABLE_OR_UNRESOLVED` otherwise.

No player can be called "matchup proof" by this run. The strongest allowed language is `environment resistant candidate` or `adverse-spot ceiling candidate` pending prospective validation.

---

# D3 — Decision outputs

Produce:
- `phase_d_same_game_team_casebook.csv`
- `phase_d_same_game_correlations.csv`
- `phase_d_catastrophic_cooccurrence.csv`
- `phase_d_game_state_signatures.csv`
- `phase_d_player_environment_sensitivity.csv`
- `phase_d_star_q4_environment_sensitivity.csv`
- `phase_d_candidate_labels.csv`
- `phase_d_result.json`

## Interpretation rules
- A strong shared pass-state pattern supports improving the common pass/run opportunity state, not independently adding yards to QB/WR/TE.
- A repeated negative QB/receiver-vs-RB opportunity relationship supports explicit finite pass/run opportunity competition in simulation.
- A stable player environment-sensitivity pattern can justify a future **targeted prospective test** of player-specific environment response.
- A player-specific pattern that is unstable across chronological halves is not actionable.
- No Phase-D result is allowed to directly change production projections. Any proposed football feature must map to an observed forensic mechanism and then pass a new frozen prospective test.
