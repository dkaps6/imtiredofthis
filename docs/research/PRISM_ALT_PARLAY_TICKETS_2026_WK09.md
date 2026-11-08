> **SUPERSEDED 2026-11-08 by `PRISM_ALT_PRICING_VALIDATION_2026_W9.md`.** The selection
> method below could not establish a priced edge and the football model rejects 18
> of its 24 legs. Use `data/research/alt_parlay_tickets_2026_wk09_v2.csv`. Retained
> as lineage.

# 2026 Week 9 — alternate-line parlay tickets, recorded pre-kickoff

**Recorded:** 2026-11-08, before the 18:00Z kickoffs · **Rows:** `data/research/alt_parlay_tickets_2026_wk09_v1.csv`
**Status:** RECORD FOR GRADING. Not a promotion. No production behaviour changed.

Every leg is an **ALTERNATE** rung, never a main line. Provider keys are
`player_pass_yds_alternate`, `player_reception_yds_alternate`,
`player_receptions_alternate`, `player_rush_yds_alternate`,
`player_rush_reception_yds_alternate`. The `main_line` recorded beside each leg
is only the anchor the clearance table is indexed on — the denominator, not the
bet. Main-line side-picking is closed: realized P(over) is 0.476-0.508 in every
line quartile (`PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md`).

**Source boundary:** paid capture run 42487807719, archived at
`data/market_track_record/alt_props/2026_wk09_sun3.csv` (10,155 rows) and priced
to `..._sun3_edges.csv` (10,171 rows). Prices as of 2026-11-08T14:08Z.

## Selection criteria, applied uniformly

1. the clearance table beats the offered price (`ev_pop > 0`);
2. the player's own 2026 record also beats it (`ev_own > 0`) — "robust";
3. at least 6 games of own evidence, denominator from `load_snap_counts`
   (`offense_snaps > 0`) so zero-involvement games are counted;
4. one leg per game, so multiplying leg probabilities is defensible.

Pool: **172 robust legs** — receptions 72, rec_yards 61, pass_yards 23,
rush_yards 15, rush_rec_yards 1.

Probability estimate is the table's `cleared`, which is validated
out-of-sample: 13,990 cells over 2026 weeks 1/2/6/7/8, maximum band deviation
3.9pp (`alt_clearance_backtest_2026_v1.csv`). **No haircut is applied.** An
earlier 0.9578/0.5833 haircut derived from one low-volume game is retracted.

## Known caveats, stated on the record

- **`rush_yards` runs light.** Backtest: 52.2% actual vs 57.1% predicted
  (-4.9pp, n=3,440). HIGH is four-fifths that market — discount it.
- **Cross-game independence is assumed, not measured.** League-wide passing
  volume correlates by week, so joint probabilities are optimistic by an
  unmeasured amount.
- **Own-record samples are 6-8 games.** They are a filter, not an estimator.
- **Dawson Knox is BUF v MIN, Monday 20:15 ET.** SAFE and MODERATE do not
  settle until Monday night.
- Alternate ladders are OVER-only, so every ticket is structurally long.

## The tickets

### SAFE — 3 legs · +151 · win 70.5% · EV +0.77

| market | matchup | player | rung | price | needs | table | own |
|---|---|---|---|---|---|---|---|
| receptions | NYJ v KC | Braelon Allen | >=1 | -275 | 73.3% | 89.0% | 7/8 |
| receptions | BUF v MIN | Dawson Knox | >=1 | -280 | 73.7% | 89.0% | 7/7 |
| receptions | CAR v DEN | Mitchell Evans | >=1 | -280 | 73.7% | 89.0% | 6/7 |

Allen: RB, ~80% snaps, targeted in 8/8 games (1,1,3,2,4,2,7,2).
Knox: TE, ~70% snaps, a reception in all seven games, never targetless.
Evans: TE, ~50% snaps, targeted 7/7; his one blank was 1 target, 0 catches.
Band rationale: >=70% is the best-validated cell type, 3,952 cells, 80.6%
predicted vs 80.3% actual.

### MODERATE — 5 legs · +279 · win 53.7% · EV +1.03

SAFE plus:

| market | matchup | player | rung | price | needs | table | own |
|---|---|---|---|---|---|---|---|
| rec_yards | DAL v IND | Ryan Flournoy | >19.5 | -580 | 85.3% | 85.7% | 8/8 |
| receptions | HOU v LAC | Foster Moreau | >=1 | -350 | 77.8% | 89.0% | 6/7 |

Flournoy: WR, ~72% snaps, 4-9 targets every week, receiving yards
22/23/22/37/90/40/70/38 — floor of 22, so a 19.5 rung at 0.41x of his 47.5
anchor is the most consistent leg on the board. Margin over price is thin
(85.3 vs 85.7) and rests on his own record.
Moreau: TE, ~39% snaps — the lowest-usage leg in this tier; his miss was a
0-target game.

### MEDIUM — 4 legs · +1519 · win 9.7% · EV +0.58 — all yards

| market | matchup | player | rung | price | needs | table | own |
|---|---|---|---|---|---|---|---|
| rec_yards | NO v CLE | Noah Fant | >3.5 | -107 | 51.7% | 59.6% | 4/6 |
| rec_yards | CHI v TB | Roschon Johnson | >2.5 | -116 | 53.7% | 61.0% | 4/6 |
| rush_yards | KC v NYJ | Emmett Johnson | >14.5 | +114 | 46.7% | 51.3% | 4/7 |
| rush_yards | GB v NE | MarShawn Lloyd | >9.5 | +110 | 47.6% | 52.2% | 7/8 |

Lloyd is the anchor: 3-13 carries every game, rushing 37/20/11/13/31/10/20/7,
so 7/8 over 9.5. Fant: TE, ~52% snaps, but two 0-target games. Roschon Johnson:
RB at only ~17% snaps — a usage-dependent leg. Emmett Johnson: RB, ~27% snaps,
two zero-carry games.
Band rationale: 30-70% grades 41.2% and 57.9% against 42.9% and 60.2%
predicted over 8,391 cells — usable, which an earlier single-game read denied.

### HIGH — 5 legs · +7515 · win 2.6% · EV +0.96 — QB rushing

| market | matchup | player | rung | price | needs | table | own |
|---|---|---|---|---|---|---|---|
| rush_yards | SF v LV | Brock Purdy | >19.5 | +148 | 40.3% | 48.2% | 5/7 |
| rush_yards | NO v CLE | Tyler Shough | >19.5 | +154 | 39.4% | 47.0% | 3/7 |
| rec_yards | HOU v LAC | Woody Marks | >19.5 | +169 | 37.2% | 42.5% | 3/7 |
| rush_yards | KC v NYJ | Emmett Johnson | >14.5 | +114 | 46.7% | 51.3% | 4/7 |
| rush_yards | GB v NE | MarShawn Lloyd | >9.5 | +110 | 47.6% | 52.2% | 7/8 |

Purdy is the largest single-leg edge on the board (+7.9pp): 2-10 carries a
game, rushing 29/30/34/29/6/28/5, so 5/7 over 19.5 at a 1.30x rung.
Shough: same shape, more volatile (8/23/36/5/0/32/7).
Marks: RB, ~51% snaps, one negative-yardage game.
**This ticket is four-fifths `rush_yards`, the one market the backtest shows
running 4.9pp light. Discount accordingly.**

### LOTTERY — 7 legs · +50,464,972 · win ~0.005% · EV +0.69

| market | matchup | player | rung | price | needs | table | own |
|---|---|---|---|---|---|---|---|
| pass_yards | GB v NE | Jordan Love | >374.5 | +850 | 10.5% | 12.6% | 1/8 |
| pass_yards | KC v NYJ | Patrick Mahomes | >329.5 | +690 | 12.7% | 13.9% | 1/7 |
| pass_yards | DET v MIA | Jared Goff | >349.5 | +630 | 13.7% | 14.7% | 1/7 |
| pass_yards | LV v SF | Kirk Cousins | >339.5 | +720 | 12.2% | 12.9% | 1/8 |
| pass_yards | HOU v LAC | C.J. Stroud | >349.5 | +825 | 10.8% | 11.3% | 1/7 |
| rec_yards | CAR v DEN | Chuba Hubbard | >24.5 | +164 | 37.9% | 39.6% | 4/7 |
| rec_yards | DAL v IND | CeeDee Lamb | >124.5 | +360 | 21.7% | 22.6% | 2/8 |

Five QB passing ceilings, each cleared exactly once this season, all at 1.2x-1.4x
rungs: Love 387 high on 29-53 attempts; Goff 412; Mahomes 382; Cousins 365;
Stroud 353. Lamb has 153 and 189 on 21-target and 18-target weeks. Every leg's
edge over price is 1-2pp, so this is a ticket and is labelled as one. Cutting it
to 3-4 legs is the sane version.

## Grading

Grade from nflverse weekly stats once week 9 settles, against
`data/market_track_record/alt_props/2026_wk09_sun3_edges.csv`. Report per leg,
per ticket, and per probability band, and append the band result to
`alt_clearance_backtest_2026_v1.csv` so the calibration series grows by a week
rather than being re-derived from one game.
