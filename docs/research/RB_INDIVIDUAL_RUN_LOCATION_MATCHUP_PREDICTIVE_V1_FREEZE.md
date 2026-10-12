# RB PLAYER ROLE MATCHUP — INDIVIDUAL RUN-LOCATION PREDICTIVE V1 FREEZE

Frozen: 2026-10-11

Status: **FROZEN BEFORE RB OUTCOME SCORING. NO PRODUCTION CHANGE.**

Parent source:
- `docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_V1_PLAN.md`
- `docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_V1_RESULT.md`
- source run `38189983578`
- artifact `11720966759`
- digest `sha256:dec7d48d1a2c39ef255a6391b0014f75779ceb5939dfc7a90feca384074caad5`
- source disposition `RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_QUALIFIED`
- source-ready RB player-weeks: **2,654**

Historical football-only baseline authority:
- same 2024-2025 canonical historical reconstruction used by Football Matchup
  Transmission V1;
- right-tail authority run `37493425352`;
- artifact `11427113291`;
- digest `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`.

## Question

Does the opponent defense affect **this individual RB's rushing efficiency**
differently based on where that RB has actually carried the ball?

This is not:
- bad run defense -> boost RB;
- team directional mix x defense;
- M95A/M95B;
- a new carry-share rule.

RB opportunity remains frozen.

## Evaluation universe

- 2024 and 2025 regular seasons
- Weeks 2-18
- exact historical pregame player universes
- exact source-ready player/location states from Source V1
- no 2026 outcomes

## Frozen location state

Player mix:
- LEFT share
- MIDDLE share
- RIGHT share

Opponent state:
- YPC allowed LEFT
- YPC allowed MIDDLE
- YPC allowed RIGHT

For each target season/week/location build a strict-prior league YPC baseline by
target-weighting the unique qualified defense states:

`league_ypc_location = SUM(def_rushes_location * def_ypc_location) / SUM(def_rushes_location)`

For each source-ready RB:

`opp_mix_ypc = SUM(player_share_location * opponent_def_ypc_location)`

`league_mix_ypc = SUM(player_share_location * league_ypc_location)`

`raw_matchup_ratio = opp_mix_ypc / league_mix_ypc`

Invalid/missing/nonpositive denominator -> multiplier 1.0.

## Frozen safety bounds

Reuse the existing matchup-multiplier safety bounds already frozen elsewhere in
the repository:

- floor 0.50
- ceiling 1.80

`run_location_matchup_mult = clip(raw_matchup_ratio, 0.50, 1.80)`

No fitted slope, exponent, shrinkage factor, threshold, or role carveout.

## Candidate

ID:
`RB-INDIVIDUAL-RUN-LOCATION-YPC-V1`

For a source-ready RB/HB/FB only:

`rules_ypc_candidate = rules_ypc_baseline * run_location_matchup_mult`

Everything else remains baseline.

Unsupported RB rows remain baseline.

## Mandatory invariants

The candidate must preserve:
- projected RB rushing attempts / carry opportunity;
- frozen RB carry/snap allocation;
- RB receiving opportunity;
- team rush/pass volume;
- game script;
- all WR/TE/QB projections;
- catch/target rules;
- ML and State components;
- ensemble weights;
- simulation seeds.

Only the declared RB per-carry rushing-efficiency seam may change. RB `rush_rec_yards` may move only through its rushing component; its receiving component must remain exact.

## Primary scoring

Primary population:
- source-ready RB/HB/FB
- market `rush_yards`
- finite actual / baseline / candidate
- evaluated separately in 2024 and 2025

Support floor per season:
- >=500 rows
- >=100 games
- >=50 players

Absolute-error improvement:
`abs(baseline-actual) - abs(candidate-actual)`

Inference:
- 5,000 game-cluster bootstrap reps
- 5,000 player-cluster bootstrap reps
- seed `20261012`

## PASS gate

Disposition:
`RB_RUN_LOCATION_MATCHUP_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

only if ALL pass:

1. baseline authority reproduced <=1e-8;
2. primary support met in BOTH seasons;
3. primary MAE improves in BOTH seasons;
4. game-cluster 95% lower bound >0 in BOTH seasons;
5. player-cluster 95% lower bound >0 in BOTH seasons;
6. primary RMSE non-worse in BOTH seasons;
7. broad all-RB rushing MAE non-worse in BOTH seasons;
8. no statistically supported broad harm;
9. projected rushing-attempt opportunity max delta <=1e-10;
10. all non-rushing-yard untreated projections max delta <=1e-10;
11. sportsbook inputs 0;
12. OddsAPI calls 0;
13. 2026 outcomes 0;
14. fitted coefficients 0;
15. threshold searches 0;
16. production changed false.

Otherwise:
`RB_RUN_LOCATION_MATCHUP_HISTORICAL_FAIL_CLOSED`.

## No rescue rule

After scoring, do not:
- tune location bands;
- tune multiplier bounds;
- add success rate post-hoc;
- select bellcows / explosive backs / outside runners / inside runners;
- combine with M95A/M95B;
- alter carry opportunity.

A failed exact V1 candidate closes this mechanism.

A historical PASS authorizes only a separately frozen future-week shadow.
