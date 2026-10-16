# PLAYER ROLE MATCHUP — DEPTH INTERACTION PREDICTIVE V1 FREEZE

Frozen: 2026-10-11

Status: **FROZEN BEFORE RECEIVING-OUTCOME SCORING. TWO INDEPENDENT PLAYER-SPECIFIC INTEGRATION CANDIDATES. NO PRODUCTION CHANGE.**

Parent source qualification:
- plan: `docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_SOURCE_V1_PLAN.md`
- result: `docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_SOURCE_V1_RESULT.md`
- source run: `38189258833` SUCCESS
- source artifact: `11721146177`
- source digest: `sha256:8e1227f2b27ac1b42c1790e1666302dbfcecf59b643df8807aa56baa3a540fcc`
- source disposition: `PLAYER_DEPTH_MATCHUP_SOURCE_QUALIFIED`
- source-ready player-weeks: 7,184
  - WR 4,666
  - TE 2,518
  - 2024 3,518
  - 2025 3,666

Historical football-only baseline authority:
- right-tail run: `37493425352`
- artifact: `11427113291`
- digest: `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`
- exact historical reconstruction semantics are the same as
  `score_football_matchup_transmission_candidates_v1.py`.

## Question

Does an opponent defense affect **this individual receiver** differently based
on where that receiver is actually targeted?

The candidate is not:

> weak pass defense -> boost every WR/TE.

It is:

> weight the opponent's strictly-prior depth-band receiving efficiency by this
> player's own strictly-prior target-depth mix, compare that with a strict-prior
> league depth-band baseline for the same player mix, and adjust only the
> player's receiving efficiency.

Opportunity / entitlement is frozen.

## Evaluation universe

- seasons: 2024 and 2025
- regular-season Weeks 2-18
- exact historical pregame universes used by the football-only FMT harness
- source interaction rows come only from the exact qualified V1 source artifact

No 2026 outcome is used.

## Frozen league depth baseline

For each target season/week and each frozen depth band:

1. take one unique qualified opponent-defense state per defense;
2. use only that defense state's strictly-prior target counts and YPT allowed;
3. compute target-weighted cross-defense league YPT:

`league_ypt_band = SUM(def_targets_band * def_ypt_band) / SUM(def_targets_band)`

No target-week defensive result is included.

For an individual source-ready player:

`opp_mix_ypt = SUM(player_share_band * opponent_def_ypt_band)`

`league_mix_ypt = SUM(player_share_band * league_ypt_band)`

`raw_matchup_ratio = opp_mix_ypt / league_mix_ypt`

If any required state is invalid or the denominator is nonpositive, multiplier
is 1.0.

## Frozen safety bounds

Reuse the already-existing FMT matchup multiplier safety bounds:

- floor: 0.50
- ceiling: 1.80

`depth_matchup_mult = clip(raw_matchup_ratio, 0.50, 1.80)`

These bounds are inherited, not selected from this candidate's outcomes.

No slope, exponent, shrinkage coefficient, threshold, role carveout, or subgroup
is fitted.

## Candidate 1 — WR

ID:
`PLAYER-DEPTH-MATCHUP-YPT-WR-V1`

For a source-ready WR only:

`rules_ypt_candidate = rules_ypt_baseline * depth_matchup_mult`

Everything else remains baseline.

Unsupported WR rows remain baseline.

TE / RB / QB rules remain baseline.

## Candidate 2 — TE

ID:
`PLAYER-DEPTH-MATCHUP-YPT-TE-V1`

For a source-ready TE only:

`rules_ypt_candidate = rules_ypt_baseline * depth_matchup_mult`

Everything else remains baseline.

Unsupported TE rows remain baseline.

WR / RB / QB rules remain baseline.

## Explicit invariants

Both candidates must preserve:
- player target entitlement;
- room target allocation;
- team target/pass volume;
- pass rate;
- catch-rate rules;
- M38 WR1 anchor;
- WR-R15;
- TE-R5P;
- QB M89/M90 and QB C2;
- ML and State components;
- ensemble weights;
- simulation seeds;
- all non-`rules_ypt` model inputs.

No player is added or removed because of target-game production.

## Historical baseline reproduction

Before candidate scoring, the reconstructed football-only baseline must reproduce
the frozen right-tail baseline authority with max absolute final-mean gap
<=1e-8, exactly as in FMT V1.

Otherwise the run is mechanically invalid and no science is read.

## Primary scoring population

Each position is scored independently on **source-ready receiving-yard rows**.

For each candidate and season require:
- WR support >=500 rows / >=50 games / >=50 players;
- TE support >=200 rows / >=50 games / >=25 players.

Rows must have:
- exact candidate position;
- market = `rec_yards`;
- qualified V1 depth-matchup state;
- finite actual, baseline projection, candidate projection.

## Broad protection population

Also score all canonical historical receiving-yard rows for the candidate
position, including unsupported rows that remain baseline.

This broad population is a protection gate, not a way to dilute the primary
source-ready test.

## Frozen inference

Absolute-error improvement:

`AE_improvement = abs(baseline - actual) - abs(candidate - actual)`

For every primary season/position cell:
- game-cluster bootstrap, 5,000 reps;
- player-cluster bootstrap, 5,000 reps;
- seed: `20261012`.

A primary season cell is supported improvement only if both 95% CI lower bounds
are >0.

## Position-specific PASS gate

WR and TE are adjudicated independently.

A position receives:

`PLAYER_DEPTH_MATCHUP_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

only if ALL pass:

1. baseline authority reproduced within 1e-8;
2. primary support floor met in BOTH 2024 and 2025;
3. primary source-ready MAE improves in BOTH seasons;
4. game-cluster 95% lower bound >0 in BOTH seasons;
5. player-cluster 95% lower bound >0 in BOTH seasons;
6. primary RMSE non-worse in BOTH seasons;
7. broad all-position MAE non-worse in BOTH seasons;
8. no statistically supported broad harm in either season;
9. target/reception opportunity projections remain unchanged within numerical
   tolerance;
10. source-match identity coverage among eligible baseline rows is reported and
    no result-driven remapping is used;
11. sportsbook inputs = 0;
12. candidate coefficients fit = 0;
13. threshold searches = 0;
14. 2026 outcomes = 0;
15. production changed = false.

Otherwise that position receives:

`PLAYER_DEPTH_MATCHUP_HISTORICAL_FAIL_CLOSED`

## No rescue rule

After scoring:
- do not change depth bands;
- do not tune the 0.50/1.80 bounds;
- do not fit a shrinkage coefficient;
- do not choose only deep receivers, slot receivers, WR1s, field-stretchers, or
  any other subgroup;
- do not combine WR and TE to rescue one position;
- do not switch to completion-rate interaction after seeing YPT failure.

A failed position closes this exact V1 candidate.

## Advancement

A historical PASS authorizes only a separately frozen future-week shadow using
the exact same source state and transform.

It does not authorize immediate production promotion.
