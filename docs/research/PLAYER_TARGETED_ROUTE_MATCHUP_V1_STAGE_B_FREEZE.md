# Player Targeted-Route × Opponent Route-Family Matchup V1 — Stage-B Statistical Freeze

Date: 2026-10-14

Status: **FROZEN AFTER SOURCE-READY / BEFORE ANY STAGE-B OUTCOME SCORE**

Parent:
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
- Stage-A disposition: `PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY`
- source run: `38629731504`

This amendment fixes the exact Stage-B scoring mechanics before any matchup result is computed.

## 1. Eligible target player-game

WR and TE are evaluated separately.

A target player-game is eligible only when:
- regular season;
- target player position is WR or TE from exact weekly roster identity;
- target game has >=1 official PBP target;
- player has >=4 strictly-prior eligible player-games;
- player route mix uses at most the trailing 8 strictly-prior games;
- target opponent has >=100 valid targeted-route observations across at most its trailing 8 strictly-prior defensive games;
- no row from the target week is allowed into any history/reference quantity.

Cross-season prior games are allowed. Chronology uses `season * 100 + week`; source starts in 2022.

## 2. Route-family matchup construction

For each target player-game:

### Player mix
From the player's trailing <=8 prior games, count valid targeted-route labels and normalize to weights `w_r`.

### Opponent route performance
For each route family `r`, from the opponent's trailing <=8 prior defensive games:
- allowed catch rate;
- allowed receiving yards per target.

A defense-route cell must have >=10 target observations. If it has fewer than 10, that route family receives the strictly-prior league-neutral value for the same route family. This contributes zero matchup delta for that family rather than fabricating a noisy defensive estimate.

### League neutral
For each route family, use all valid league targeted-route observations strictly before the target week.

### Matchup deltas
`matchup_delta_ypt = sum_r w_r * (opponent_ypt_r - league_ypt_r)`

`matchup_delta_catch = sum_r w_r * (opponent_catch_r - league_catch_r)`

No target-game data enters these quantities.

## 3. Player baseline / actual outcomes

Official PBP target rows are used for player-game outcomes independent of whether the participation route label is present on every target.

For each player-game:
- targets = official PBP target count;
- receptions = completed passes to player;
- rec_yards = sum receiving yards from targeted pass plays;
- ypt = rec_yards / targets;
- catch_rate = receptions / targets;
- mean_air_yards = mean target air yards where available.

Player baselines use trailing <=8 strictly-prior target games:
- `prior_player_ypt`;
- `prior_player_catch`;
- `prior_player_rec_yards`;
- `prior_player_air_yards`.

Primary residual outcomes:
- `ypt_residual = actual_ypt - prior_player_ypt`;
- `catch_residual = actual_catch_rate - prior_player_catch`;
- `rec_yards_residual = actual_rec_yards - prior_player_rec_yards`.

## 4. Three predeclared position-metric families

For WR and TE separately:

1. **YPT family**
   - x = `matchup_delta_ypt`
   - y = `ypt_residual`

2. **Catch family**
   - x = `matchup_delta_catch`
   - y = `catch_residual`

3. **Receiving-yard family**
   - x = `matchup_delta_ypt`
   - y = `rec_yards_residual`

Expected direction is **positive** for all three.

No other metric can rescue V1.

## 5. Exact association statistic

Primary season statistic = Spearman correlation between x and y.

Development:
- 2023 is reported only.

Independent confirmation:
- 2024 Spearman must be >0;
- 2025 Spearman must be >0.

Pooled 2024-2025 inference uses paired cluster bootstrap of Spearman correlation.

### Player-cluster bootstrap
- cluster = stable player ID;
- sample players with replacement;
- include all scored rows for sampled players with multiplicity;
- 2,000 valid replicates;
- seed = 2026101401.

### Game-cluster bootstrap
- cluster = target game ID;
- sample games with replacement;
- include all scored rows for sampled games with multiplicity;
- 2,000 valid replicates;
- seed = 2026101402.

Hard requirement for a family:
- pooled observed Spearman >0;
- player-cluster 95% percentile CI lower bound >0;
- game-cluster 95% percentile CI lower bound >0.

## 6. Target-depth anti-proxy control

Because route family can correlate with target depth, V1 must survive a predeclared air-yard control.

Within each position on pooled 2024-2025 rows:
1. regress x on `prior_player_air_yards` with intercept using finite rows;
2. regress y on `prior_player_air_yards` with intercept;
3. compute Spearman correlation between the two residual series.

Bootstrap the depth-controlled residual Spearman with the exact same player/game cluster schemes and seeds.

Hard requirement:
- observed depth-controlled Spearman >0;
- player-cluster CI lower >0;
- game-cluster CI lower >0.

If air-yards coverage for a position-metric family is <70% of otherwise eligible pooled confirmation rows, disposition that family `DEPTH_CONTROL_SOURCE_LIMITED`; it cannot advance in V1.

## 7. Player-level reporting

For every position-metric family:
- qualifying named players;
- scored player-games by season;
- median player x and y;
- for players with >=6 scored confirmation games:
  - per-player Spearman;
  - median per-player Spearman;
  - fraction of per-player Spearman values >0.

These are required diagnostics. Aggregate position MAE is not a Stage-B criterion.

## 8. Multiple-comparison protection

There are exactly six predeclared confirmation families:
- WR YPT;
- WR catch;
- WR rec yards;
- TE YPT;
- TE catch;
- TE rec yards.

For each family, define the one-sided pooled player-cluster bootstrap p-value as:

`p = (1 + count(bootstrap_rho <= 0)) / (1 + valid_replicates)`

Apply Benjamini-Hochberg across the six p-values at q=0.10.

A family must have BH-adjusted q <=0.10 in addition to every hard sign/CI/depth-control requirement.

## 9. Frozen dispositions

Per family:
- `PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY_CONFIRMED` if every hard requirement passes;
- `NO_ACTIONABLE_PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY` otherwise;
- `DEPTH_CONTROL_SOURCE_LIMITED` if the frozen air-yard support floor fails.

Overall:
- `PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED` if at least one WR or TE family confirms;
- otherwise `NO_ACTIONABLE_PLAYER_TARGETED_ROUTE_MATCHUP`.

WR and TE are still reported independently. A WR result does not imply TE validity and vice versa.

## 10. Boundaries

- no sportsbook inputs;
- no OddsAPI calls;
- no 2026 outcomes;
- no production changes;
- no route-family regrouping after results;
- no history-window changes;
- no defensive-cell floor changes;
- no subgroup rescue;
- no use of route label as total routes run;
- no WR-CB assignment inference.

A Stage-B confirmation authorizes only a separately frozen full-stack integration candidate against the promoted named-player mean architecture.
