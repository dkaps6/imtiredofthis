# Week 6 2026 — Blind Full-Slate Alt-Line Card V1

Status: **PREREGISTERED, UNGRADED**
Constructed: 2026-10-19, before any Week 6 result was consulted
Grading: not before nflverse publishes Week 6 weekly player stats

## Why this exists

The Week 6 Sunday games are already played, nflverse has not published, and no
result was looked up while this card was built. That makes it a genuine
out-of-sample test of the construction method in
`MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md` rather than a retrospective.
The card is frozen here so the grade is against a committed record instead of
chat memory.

## Source lineage

- priced board: run `38792341891`, artifact `12000113891`
- pricing provenance `PRESERVED_REPLAY`, `odds_api_refetched=false`
- sportsbook snapshot frozen `2026-10-15T19:11:06Z`
- screen: `scripts/operations/build_alt_line_candidate_pool_v1.py`
- screen run: `39294938759` (26 gate tests green)

Board 2,528 rows -> 408 kept -> 222 after book dedupe. Gate accounting
reconciles exactly: 1,274 market_not_eligible, 627 non_positive_edge, 81
insufficient_player_evidence, 79 degenerate_low_line, 48 rb_rush_under_cluster,
7 not_a_playable_role, 4 gadget_market, + 408 kept = 2,528.

## Stated limitation, declared before grading

Section 4 of the construction process requires reconciling current
availability and the current line before entry. **That layer was deliberately
not applied to the 13 Sunday games**, because sourcing Friday/Saturday news now
would risk pulling in results and would destroy the blind test. Availability
here is whatever the board resolved on 2026-10-15.

For the Monday game, where that layer *was* applied, it changed three things
materially (Evans questionable, Croskey-Merritt out, Bosa out). So this card
should be expected to underperform a real Sunday card, and the gap between the
two is the measurable contribution of the late-news layer.

## Two findings from building it

**1. Receptions carries a one-directional model skew.** After the gates, the
over/under split is balanced in pass_yards (1.08:1) and rush_yards (1.10:1),
mildly skewed in rec_yards (1.64:1), and **3.71:1 toward UNDER in receptions**.
Tightening the gates made it worse, not better. That is the signature of a
systematic bias, not 63 independent edges, and it is the same shape as the
already-documented RB rushing-under cluster. The receptions market is excluded
from this card and is proposed as a research lane.

**2. The model's edge is offense-level, not player-level.** Of the 58 usable
candidates, the top 10 by cushion come from 6 distinct offenses and the top 15
from 8. Legs per offense: TEN 8, JAX 6, DEN 6, NO 4, TB 4, ATL 4, CHI 4. Taking
the top five by edge would buy one correlated bet on three offenses while
looking like a diversified five-leg ticket. This is the same mechanism behind
the Week 3 parlay failure, and it is a direct instance of the PRISM problem: a
team-level opinion wearing player-level clothes. Every ticket below is
therefore capped at one leg per game.

## Cards

All legs are OVER / + yardage, because FanDuel posts alt ladders only for +
yardages and the method's whole purpose is buying cushion. `P` is from the
model mean and sd; see the calibration caveat below.

### SAFE — 3 legs, P 0.651

| player | role | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|
| Omarion Hampton | RB1 | 20+ rush yds | 39.8 | +19.8 | 0.882 | KC-LAC |
| Kendre Miller | RB1 | 20+ rush yds | 42.6 | +22.6 | 0.873 | NO-NYG |
| Dalton Schultz | TE1 | 20+ rec yds | 40.0 | +20.0 | 0.845 | HOU-JAX |

### MODERATE — 4 legs, P 0.389

| player | role | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|
| Cooper Kupp | WR2 | 10+ rec yds | 28.1 | +18.1 | 0.855 | DEN-SEA |
| Kyle Pitts | TE1 | 20+ rec yds | 34.6 | +14.6 | 0.798 | ATL-CHI |
| Calvin Ridley | WR1 | 10+ rec yds | 21.8 | +11.8 | 0.777 | IND-TEN |
| Bucky Irving | RB1 | 10+ rec yds | 17.5 | +7.5 | 0.734 | PIT-TB |

### HIGH — 5 legs, P 0.152

| player | role | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|
| Omarion Hampton | RB1 | 30+ rush yds | 39.8 | +9.8 | 0.721 | KC-LAC |
| Trevor Lawrence | QB1 | 225+ pass yds | 256.3 | +31.3 | 0.691 | HOU-JAX |
| Cade Otton | TE1 | 30+ rec yds | 39.4 | +9.4 | 0.691 | PIT-TB |
| Cam Ward | QB1 | 200+ pass yds | 225.2 | +25.2 | 0.671 | IND-TEN |
| Rome Odunze | WR1 | 40+ rec yds | 49.0 | +9.0 | 0.658 | ATL-CHI |

Both QB legs carry the promoted M89/M90 synthesis with
`model_proj == qb_synthesis_proj`.

### LOTTERY — 4 legs, P 0.012

| player | role | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|
| Kyle Pitts | TE1 | 40+ rec yds | 34.6 | -5.4 | 0.379 | ATL-CHI |
| Kendre Miller | RB1 | 50+ rush yds | 42.6 | -7.4 | 0.354 | NO-NYG |
| Dalton Schultz | TE1 | 50+ rec yds | 40.0 | -10.0 | 0.306 | HOU-JAX |
| Calvin Ridley | WR1 | 30+ rec yds | 21.8 | -8.2 | 0.298 | IND-TEN |

Probability shrinks through threshold distance on already-qualified players,
not by adding longshots. No player appears on more than two tickets; every
ticket is one leg per game.

## Calibration caveat

The listed `P` values come from the model's own mean and sd. In the Weeks 1-3
sample the 0.80-0.90 stated-probability band went **6-9**. Alt-line
construction deliberately targets that band, so the SAFE tier is built on the
least trustworthy part of the probability curve. The product probabilities are
printed for shape only and should not be read as calibrated ticket odds.

## Grading protocol

When Week 6 stats publish, grade each leg and classify every loss as exactly
one of:

- `MODEL_MISS` — the player was available in his stated role and the
  projection was simply wrong;
- `AVAILABILITY_MISS` — inactive, early exit, or a role change that the
  2026-10-15 board could not have known, i.e. attributable to the §4 layer
  this card declares it skipped;
- `STATE_UNRESOLVED` — cause not determinable from available sources; per the
  anomaly gate, fail closed rather than assign it to either bucket.

Report leg-level hit rate by tier, cushion-vs-realized calibration, and the
`MODEL_MISS` / `AVAILABILITY_MISS` split. The split is the point: it measures
how much of the method's performance depends on late news versus the football
model, which no single-game card can isolate.
