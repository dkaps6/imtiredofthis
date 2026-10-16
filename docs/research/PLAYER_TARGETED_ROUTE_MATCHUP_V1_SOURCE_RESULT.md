# Player Targeted-Route × Opponent Route-Family Matchup V1 — Source Result

Date: 2026-10-14

Status: **COMPLETE / PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY / RESEARCH ONLY**

Frozen contract:
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `159b8213a734e3f4ee16fc86fbb51950099f3323`
- workflow: `Research Player Targeted Route Matchup Source V1`
- run: `38637901016` — SUCCESS
- job: `116105672352`
- artifact: `11928258474`
- artifact digest: `sha256:b957ff0b37f0b2d8f45232bcd44657928cecc75f5f5e6d1c292ee8e35729228c`

## Frozen disposition

`PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY`

Every frozen source/support gate passed.

## Semantic contract that passed

The source is eligible only for **targeted-route mix**:

- nflverse participation `route` is treated as the route label for the primary/targeted receiver on the play;
- it is **not** treated as total routes run;
- it cannot create route participation, TPRR, YPRR, or non-target route state;
- no WR-CB assignment is inferred.

This semantic restriction remains binding in Stage B.

## Source support

Historical source seasons:
- 2022
- 2023
- 2024
- 2025

Scored seasons:
- 2023
- 2024
- 2025

Raw route-labeled participation rows:
- 75,773

Joined route/PBP rows:
- 72,206

Game/play join rate:
- 100%

Valid WR/TE targeted-route rows under the frozen taxonomy:
- 46,666

Distinct valid WR/TE players:
- 554

Identity coverage on taxonomy-eligible route rows:
- 93.1827%

## Season support

2023:
- valid targeted-route rows: 14,028
- distinct WR/TE: 327
- players with enough history to reach >=4 strictly-prior eligible games: 257
- defenses with >=100 prior route observations by Week 8: 32

2024:
- valid targeted-route rows: 13,923
- distinct WR/TE: 343
- players with enough history to reach >=4 strictly-prior eligible games: 246
- defenses with >=100 prior route observations by Week 8: 32

2025:
- valid targeted-route rows: 13,567
- distinct WR/TE: 347
- players with enough history to reach >=4 strictly-prior eligible games: 253
- defenses with >=100 prior route observations by Week 8: 32

## Frozen route-family support

Ten route families exceed 1,000 pooled valid rows:

- HITCH/CURL: 7,609
- QUICK OUT: 6,986
- GO: 5,607
- IN/DIG: 4,543
- SCREEN: 4,228
- SLANT: 4,205
- DEEP OUT: 3,538
- POST: 3,214
- SHALLOW CROSS/DRAG: 3,114
- CORNER: 2,611

Lower-support families retained under the frozen source taxonomy but must obey the support rules:
- SWING: 521
- WHEEL: 410
- TEXAS/ANGLE: 80

Do not regroup route families after outcome exposure to rescue a result.

## Integrity

- sportsbook inputs used: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- production changed: false
- targeted-route semantics preserved: true
- fake all-route participation constructed: false

## Scientific consequence

Stage A is complete and **Stage B is authorized** under the already-frozen contract.

The next question is:

> For this named WR/TE, does his own strictly-prior targeted-route mix, crossed with this opponent's strictly-prior route-family susceptibility, contain stable pregame information about his next-game receiving efficiency/output beyond the neutral route-family expectation?

Stage B must preserve the frozen design:

Player history:
- same named player only;
- trailing 8 eligible player-games maximum;
- minimum 4 strictly-prior player-games;
- targeted-route mix only.

Opponent history:
- target opponent only;
- trailing 8 defensive games maximum;
- strictly prior only;
- minimum support as frozen.

Primary derived state:
`matchup_delta_m = sum_r(player_route_mix_r * opponent_prior_allowed_m_r) - sum_r(player_route_mix_r * league_prior_allowed_m_r)`

Frozen primary outcomes, reported WR and TE separately:
- next-game receiving yards per target;
- next-game catch rate;
- next-game receiving yards.

Target count remains diagnostic only because the source has no all-route denominator.

Validation:
- 2023 development;
- 2024 independent confirmation;
- 2025 independent confirmation;
- player-cluster and game-cluster confidence required;
- player-level reporting required;
- position-level aggregate alone can never qualify the hypothesis.

A Stage-B PASS authorizes only a separately frozen full-stack integration candidate. It does not itself change M38/WR-R15, TE-R5P, production, sportsbook logic, or betting selection.

## Anti-rescue

If Stage B fails:
- do not change history windows;
- do not regroup route families;
- do not select favorable players/defenses/routes;
- do not infer total routes;
- do not infer WR-CB assignments;
- do not add sportsbook inputs;
- do not relabel the same signal as route participation.
