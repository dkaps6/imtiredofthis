# PRISM 2026 Week-5 Strict-Prior Named-Player Replay V1 — Result

Date: 2026-10-17

Status: **FORENSIC STRICT-PRIOR CURRENT-SEASON QA COMPLETE / NO PROMOTION**

Model:
**PRISM — Player Role & Individual Situation Model**

## Why this replay exists

The governing model principle is player-first:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

The user correctly challenged the project not to wait for another shadow week when
2026 already contains usable completed evidence.

This replay therefore uses the completed first four weeks strictly as prior
player history and Week 5 strictly as the evaluation week.

It does **not** rewrite the frozen Week-7-to-10 WR dual prospective confirmation.
It is an additional current-season forensic QA lane.

## Exact source authority

Strict-prior Weeks 1-4 player-output decomposition:
- run `37777319154` — SUCCESS;
- exact scientific head `cb1e511adcb58e40fc41d0949306083a304e83dd`;
- artifact `11550446909`;
- digest `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`.

Week-5 preserved full-model postmortem:
- canonical pregame Full Slate run `37860592615`;
- canonical pregame artifact `11585973062`;
- postmortem run `38395384228` — SUCCESS;
- postmortem artifact `11812200405`;
- digest `sha256:5888cab39506ae0bf70034dfb60f297defd28dd3219ea8118870738706f55ac3`.

Replay implementation:
`scripts/research/audit_prism_week5_strict_prior_player_replay_v1.py`.

## Frozen retrospective construction

For every Week-5 player-market row:

1. same named player only;
2. same market only;
3. history must contain **all four** Weeks 1, 2, 3 and 4;
4. only grading-valid / decomposition-eligible prior rows are accepted;
5. prior player difficulty is mean absolute projection error across Weeks 1-4;
6. prior difficulty is percentile-ranked **within the same market**;
7. HIGH_TRUST = <=25th percentile prior difficulty;
8. LOW_TRUST = >=75th percentile;
9. Week-5 error is scored only after the prior state is constructed;
10. no sportsbook line or price is used as a predictor.

Markets with exact Weeks 1-4 decomposition parity:
- pass yards;
- rush yards;
- receiving yards;
- receptions.

Rush+receiving is excluded because the Weeks 1-4 decomposition artifact does not
contain that market.

## Support

- eligible Week-5 player-market rows: **318**;
- unique named players: **166**;
- distinct games: **15**;
- HIGH_TRUST rows: **78**;
- LOW_TRUST rows: **82**;
- row digest:
  `sha256:454d1f15fa404324254f31674a61f65be2a3e60b193d21665e00030f3fc50127`.

## Primary current-season result

Pooled after normalizing both prior difficulty and Week-5 error within market:

- Spearman association: **+0.30995**;
- HIGH_TRUST mean Week-5 error percentile: **0.42151**;
- LOW_TRUST mean Week-5 error percentile: **0.65250**.

So the named player's own same-market difficulty over Weeks 1-4 carried forward
into Week 5 in aggregate.

That is player-level current-season evidence, not position-average evidence.

## Market-specific result

| Market | Rows | Prior difficulty vs W5 error rho | HIGH_TRUST W5 MAE | LOW_TRUST W5 MAE |
|---|---:|---:|---:|---:|
| pass_yards | 20 | -0.0120 | 77.93 yd | 74.14 yd |
| rush_yards | 45 | +0.2343 | 29.02 yd | 38.81 yd |
| rec_yards | 127 | +0.3411 | 14.63 yd | 34.48 yd |
| receptions | 126 | +0.3556 | 1.12 | 2.59 |

### Interpretation

Receiving markets show a large and coherent current-season player-difficulty
separation.

Rushing yards show a weaker but favorable separation.

QB pass yards show **no** Week-5 persistence under this simple same-player
same-market construction. This is directionally consistent with the protected
QB reliability state and is another reason not to create one universal
"player volatility" rule.

## Player-first lesson

The result supports the PRISM architecture but does not complete it.

A named player can be persistently difficult to project while the *reason* for
that difficulty changes week to week.

That means PRISM must not reduce player identity to a permanent correction or a
single volatility scalar.

Reliability belongs **after**:
- current role/state;
- teammate competition;
- opportunity allocation;
- opportunity quality;
- opponent/environment;
- efficiency.

The next replay layer must therefore inspect the named player's actual football
situation, not merely his prior error.

## Scientific boundary

This replay was designed after Week-5 outcomes existed.

Therefore it is:
- valid strict-prior forensic QA;
- useful current-season evidence;
- useful for prioritizing the player-first replay;
- **not** a preregistered Week-5 prospective confirmation;
- **not** authority to change a projection mean;
- **not** authority to change a distribution;
- **not** authority to choose bets;
- **not** authority to change production.

No model was fit.
No threshold was optimized.
No sportsbook information was used upstream.
No OddsAPI call was made.
Production changed: false.

## Exact next replay action

Continue immediately into a named-player Week-4/Week-5 situation casebook.

For each major miss/hit, reconstruct only information that existed pregame and
ask, in order:

1. Who is this player historically?
2. What current role/state was active?
3. What teammate competition or availability changed?
4. Was opportunity quantity allocated correctly?
5. Was the opportunity subtype/quality understood correctly?
6. What opponent/environment property mattered to **this player's path**?
7. Was individual efficiency the miss?
8. How reliable was the resulting player-market projection?
9. Did PRISM map the football thesis to the correct market?

Aggregate MAE remains downstream QA only.
