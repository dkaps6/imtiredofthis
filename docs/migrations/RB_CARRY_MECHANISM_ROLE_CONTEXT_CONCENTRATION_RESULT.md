# RB Carry-Mechanism Role/Context Concentration Result

## Canonical run

- Run: `34066662563`
- Job: `101576504546`
- Tested SHA: `2e3f15b1077b7a5d58a23500290fa73488f6f82c`
- Artifact: `rb-carry-mechanism-role-context-concentration`
- Artifact ID: `9999153461`
- Artifact SHA256: `7adff227b8e71b36b9ae7c5453266adc63a31c1700ec1b2686e1537a8f1f298e`
- Conclusion: `success`

## Frozen disposition

`NO_STRONG_ROLE_CONTEXT_CONCENTRATION_FOR_CARRY_ERRORS`

## Carry-dominant cohort

- 35 players
- 473 player-games

Depth-vs-projected-carry-order mismatch:

- carry-component absolute ratio: `1.162880252851498`
- carry MAE ratio: `1.0988973760544807`
- rushing-yard MAE ratio: `1.0646147290537211`

These were directionally higher but missed the frozen `1.20`, `1.15`, and broader concentration gates.

Across the five frozen states:

- only `1/5` reached carry-component-absolute ratio >= `1.15`
- `0/5` reached rushing-yard-MAE ratio >= `1.10`

The other simple transition states (limited prior history, no prior same-team game, rookie, injury-created context) did not show broad carry-error concentration in the carry-dominant cohort.

## Secondary YPC-dominant observations

The YPC-dominant cohort contained 25 players / 337 player-games. Some sparse transition states produced very large carry-component ratios, notably limited prior history and no prior same-team game, but these did not translate into comparably large rushing-yard MAE and were explicitly secondary/descriptive. They cannot rescue the failed primary gate.

## Scientific interpretation

The earlier concern that current depth-chart / role transition state might be the main missing explanation for RB carry error is **not supported as a broad standalone mechanism**. Depth mismatch has some signal, but not enough to justify a global hard-coded correction or carry-share override.

This strengthens the case for continuing RB research by mechanism:

- carry-dominant players need richer player/team/game opportunity states than simple depth/rookie/injury flags;
- YPC-dominant players require a separate efficiency/matchup lane using the existing blocking, defensive-rush, trench, player-efficiency and game-context research lineage rather than trying to fix them with role logic.

No frozen thresholds are changed and no failed state is promoted.

- Sportsbook inputs used: `false`
- Model fitting used: `false`
- Production changed: `false`