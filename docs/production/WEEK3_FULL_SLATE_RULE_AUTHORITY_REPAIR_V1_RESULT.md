# Week 3 Full Slate Football Rule Authority Repair V1 — Result

Date: 2026-09-27

Disposition: **WEEK3_FULL_SLATE_RULE_AUTHORITY_REPAIR_PASS**

Status: **EXACT FAILED-ARTIFACT REPLAY PASS — READY FOR PRODUCTION PR**

## Frozen plan

`docs/production/WEEK3_FULL_SLATE_RULE_AUTHORITY_REPAIR_V1_PLAN.md`

Plan commit:
`a6fada0e265a205a810319a86dd02f46c511e503`

## Failure authority

Original failing live Full Slate:
- run `36276421905`
- head `753a36a8ca77feb16373eb7c2439efed05e9781f`
- preserved artifact `10917037948`
- digest `sha256:738422833937cd40ddbd7dd85ebf01741d11aec079eb681a7f8cf87818058f90`

Original failure:
`full-universe football assumption drift for rules_tgt_share: max_abs_diff=0.014064361123645175`

The availability-aware 30-team seam had already passed in that run. The prior 32-team diagnosis is superseded for this failure.

## Root cause reproduced exactly

LAR WR Puka Nacua:
- exists in PlayerForm / ModelContext / Bayesian football authority;
- has zero sportsbook-shaped `metrics_ready` rows in the failed prepared slate;
- raw PlayerContext target share: **0.3090024330900243**
- Bayesian target share: **0.2647748823867376**
- sportsbook-shaped metrics rows: **0**

The injury redistribution rule built its Bayesian share map from only the metrics rows passed to it. Therefore the pricing-shaped rule pass omitted Puka's posterior and fell back to raw context share, while the full-football pass used the Bayesian posterior. That changed the vacancy amount redistributed to LAR successors.

## Frozen repair

Production pricing now:
1. loads the complete leakage-safe PlayerForm Bayesian baseline once;
2. applies that baseline to pricing rows;
3. passes that same complete football authority into the rule layer for injury redistribution.

The rule layer retains its previous row-local behavior for callers that do not provide a complete baseline.

No coefficient, injury status, 50% alpha vacancy fraction, 60/30/10 recipient weighting, matchup multiplier, M38/TE-R5P/WR-R15 setting, ensemble weight, ML/State model, QB synthesis, RB synthesis, availability rule, or sportsbook input changed.

## Authoritative replay

Corrected exact-artifact replay:
- run `36281702615`
- head `87c6ba73c67000de4d2953047e65c89e96527a8a`
- artifact `10919081863`
- digest `sha256:e3a9829738041cb61bf77d6de3a6eb180d0d4ac3bee91baf4caff2ef3a69039d`

No new OddsAPI acquisition.
No Week-3 outcomes used.

PASS:
- focused injury-redistribution regression tests;
- exact failed Week-3 prepared artifact restore;
- frozen Puka root-cause precondition;
- current availability/team seams;
- complete production pricing stack;
- football-rule parity;
- RB Rush+Receiving Conservation V2 production certification.

## Exact replay output

- football teams: **30**
- canonical games: **15**
- football players: **421**
- priced unique players: **362**
- football players without priced offers: **59**
- priced unique player-market keys checked: **833**
- priced distribution misses: **0**
- final priced rows: **3,304**
- provider event aliases: **15**
- sportsbook rows used to define football universe: **0**
- sportsbook inputs used to generate football distributions: **false**

Every protected football-assumption difference is exactly **0.0**, including:
- rules_tgt_share
- rules_rush_share
- rules_ypt
- rules_ypc
- rules_ypa
- rules_catch_rate
- rules_plays_est
- rules_pass_rate
- rule efficiency/volatility multipliers
- red-zone shares
- offensive TD rate.

RB Rush+Receiving V2:
- applied player-games: **37**
- applied rows: **142**
- max pathwise identity gap: **0.0**
- max final-vs-target gap: `1.4210854715202004e-14`
- sportsbook inputs to V2 football: **0**
- Week-1 rows changed: **0**

## Production implication

The LAR target-share blocker is closed on the exact data that failed live.

This branch is based on PR #644's roster-boundary fix, so a production PR from this branch contains both currently known Week-3 blockers:
1. quarantine unrostered non-core sportsbook rows instead of killing a paid slate;
2. make injury redistribution consume complete football Bayesian authority rather than sportsbook-shaped row presence.

Next gate: Repo CI / review, then merge and run a clean no-live-odds Full Slate from main before any new paid odds acquisition.
