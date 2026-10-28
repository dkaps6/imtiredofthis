# PRISM — Player-specific matchup response: RESULTS (V1)

**Preregistration:** `PRISM_PLAYER_SPECIFIC_MATCHUP_PREREGISTRATION_V1.md`, frozen at
`da7e7c1c` **before** any scoring.
**Script:** `scripts/research/measure_player_specific_matchup_response_v1.py`
**Record:** `data/research/player_matchup/player_specific_matchup_response_v1.json`
**Per-player rows:** `data/research/player_matchup/player_specific_matchup_response_v1_rows.csv`

## Verdict: H1 NOT SUPPORTED

Prior aDOT interacted with the opponent's prior man/zone profile does **not** improve
named-player receiving-yards prediction out of sample. Reported as the result; no
subgroup was searched, no model re-specified, no player carved out.

### Evidence

Panel: **1,765 player-games, 231 players**, 2025, built from 22,055 coverage-labeled
plays. Anomaly gate active: 4,389 `HEALTHY_FULL`, 369 `PARTIAL_GAME` (excluded from
outcomes), 322 `HEALTHY_FULL_SNAPS_UNKNOWN`, 270 `DNP_INACTIVE` (excluded from both
prior and outcome), 23 `UNRESOLVED_CONTEXT` (failed closed).

**Receiving yards — holdout weeks 12–18, fit on 4–11:**

| model | what it adds | holdout MAE |
|---|---|---|
| M0 | the player's own prior baseline only | 22.560 |
| M1 | + **positional** matchup (what production does) | **22.389** |
| M2 | + **player-specific** aDOT × coverage | 22.398 |

**Targets (preregistered as secondary, reported, not decisive):** M0 **1.963**,
M1 1.977, M2 1.975 — the positional matchup model is *worse than the player's own
baseline*.

**Decision rule, as frozen:**

1. M2 beats M1 on player-level error — **PASS**, but by `+0.002` yards MAE with
   105 of 221 players improved (47.5%). That is a coin flip, not an effect.
2. `prior_adot × def_zone_rate` sign persists across disjoint windows — **PASS** on
   sign only. Magnitude collapses roughly sixfold (`−0.880` on weeks 4–10 →
   `−0.144` on 11–18) and the companion `prior_adot × def_man_rate` term **flips
   sign** (`+1.504` → `−0.430`). Unstable.
3. Improvement concentrated in off-mean-aDOT quartiles — **FAIL**. Off-mean players
   `−0.007`, middle quartiles `+0.012`. The sign is backwards: what little movement
   exists sits with the *typical*-aDOT players, which is the signature of a
   position-level effect, not a player-specific one.

Rule 3 fails, so **H1 is not supported**, and rules 1 and 2 pass only in a degenerate
sense that reinforces the null.

## The more important finding: the matchup layer is a weak lever at all

This was not the question asked, but it falls straight out of the frozen comparison and
it matters more than the verdict.

**M1 beats M0 by 0.171 yards of MAE on a 22.56 baseline — 0.76%.** On targets, the
positional matchup model is net *negative*. So the opponent/environment layer is worth
well under one percent on receiving yards and nothing on targets, **even when it works
as designed**.

That reframes the earlier transmission findings. Production's matchup layer is inert
(`wr1_target_mult` = 1.0 on all 396 W7 rows, SLOT unreachable, `rules_pass_rate` the
constant 0.57, `middle_open` firing for all 32 teams). Those are real defects — and
this study says repairing them buys very little for receiving markets. The inert layer
is cheap, not catastrophic.

**Where the leverage actually is:** M0 — the player's own prior baseline alone — lands
within 0.76% of the best model tested. The player profile is doing essentially all of
the work. And M0 here is *crude*: `prior_ypt × prior_tgt_per_game`, nothing else.

The identity card already holds far richer named-player axes that M0 never touched:
`red_zone_opp`, `inside_10_opp`, `inside_5_opp`, `goal_to_go_opp`, `third_down_opp`,
`two_minute_opp`, `early_down_opp`, `team_red_zone_share_current`,
`opp_per_week_recent2` vs `opp_per_week_earlier` (trajectory),
`opp_week_to_week_sd` (reliability), `yards_per_opp_sd`, `catch_rate`.

Steps 1–3 of the operating principle — identity, current role, opportunity allocation
— are where the measurable room is. Step 4 (opponent) is small for these markets.

## What this result does NOT establish

A null at this power is "not detectable here with this axis", not "does not exist".

- **Alignment was not tested.** `run_player_form_v2.py` collapses LWR/RWR/SWR to
  position family, so slot-vs-boundary — the strongest player-specific matchup axis —
  does not survive into any artifact. aDOT was used as the available proxy, as the
  preregistration stated. Alignment remains genuinely untested, not refuted.
- **The defense axis is team-level.** Who actually covers *this* player is a different
  question; `cb_coverage_player` / `wr_cb_exposure` exist and were not used.
- **Power is poor.** 22.4 yards of MAE against week-to-week receiving variance, at
  n = 1,765, cannot resolve sub-percent effects. Both the positional and the
  player-specific effects may sit under the detection floor.
- **One season** (2025), the last inside the `load_participation` window.
- **Receiving only.** Rushing was not tested, and box-count-versus-runner-profile is
  the most mechanically plausible remaining matchup interaction
  (`defenders_in_box` is real, play-level, and already loaded here).

## Recommended next step

Stop investing in the opponent layer for receiving markets. Test instead whether the
identity card's player profile improves the **named-player baseline** beyond
`prior_ypt × prior_tgt_per_game` — same discipline, preregistered and frozen first,
player-level primary metric, disjoint-window persistence.

That is step 1–3 work, it is purely player-level, it uses evidence we already built,
and this study says it is where the room is.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read at any stage. Betting remains CLOSED. The `PERCENTY_COLS`
scale bug and SLOT unreachability remain separate open items, neither resolved here.
