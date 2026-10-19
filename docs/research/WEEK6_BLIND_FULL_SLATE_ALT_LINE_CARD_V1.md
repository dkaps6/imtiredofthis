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
- screen run: `39297764305` (29 gate tests green)

Board 2,528 rows -> 396 kept -> 215 after book dedupe. Gate accounting
reconciles exactly: 1,274 market_not_eligible, 627 non_positive_edge, 81
insufficient_player_evidence, 70 degenerate_low_line, 48 rb_rush_under_cluster,
28 not_a_playable_role, 4 gadget_market, + 396 kept = 2,528.

## Correction, 2026-10-19 (pre-grading)

The first version of this card listed Calvin Ridley as TEN's WR1. He is
model_role WR4 on an 11.7% target share. The user caught it on the output; the
gate did not.

Two distinct role-semantics defects were behind it, failing in opposite
directions by position.

**Receivers, backs, tight ends.** Ourlads' WR1/WR2/WR3 name the alignment --
LWR, SWR, RWR -- and depth_index marks starter against backup within a slot.
player_form carries that label through as `depth_role` while computing the real
usage hierarchy into `model_role`. The screen gated on `depth_role`, and since
every rostered receiver has an alignment label in WR1-WR3, it admitted whole
rooms: it rejected 7 rows out of 2,528, a number implausible enough that it
should have been the tell. TEN shows the inversion both ways -- Ridley reads
depth_role=WR1 at rank WR4, while the room's actual WR1 is Carnell Tate, who
reads depth_role=WR3 because he lines up at RWR.

**Quarterbacks.** Gating on `model_role` instead then dropped Cam Ward, TEN's
starter with 22 games, because `model_role` ranks quarterbacks by descending
`ypa`. In all three rooms where the fields disagree the model_role order is
exactly the ypa order: TEN reads Trubisky QB1 on 8.94 over four games ahead of
Ward on 5.82 over 22; CHI reads Bagent QB1 on 9.45 over five games ahead of
Caleb Williams on 7.09 over 19; IND inverts Richardson and Leonard. Six of 37
quarterback rows carry the swap, and JAX, KC and DEN hide it only because their
ypa order happens to match their depth order.

So rank is now read per position: `model_role` for RB/WR/TE, `depth_role` for
QB, where a single-alignment position makes the depth label a true order. That
raised `not_a_playable_role` from 7 rejections to 28.

Three legs changed as a result:

| was | now | why |
|---|---|---|
| Calvin Ridley 10+/30+ rec (WR4, tgt 11.7%) | Wan'Dale Robinson 20+/60+ rec (WR2, tgt 24.9%) | Ridley is the fourth target and the second body at LWR |
| Kendre Miller 20+/50+ rush (RB2, rush 29.1%) | Alvin Kamara 15+/40+ rush (RB1, rush 39.0%) | same inversion in NO's backfield: Kamara is model RB1 carrying depth_role RB2 |
| Cam Ward dropped by the model_role gate | Cam Ward 200+ pass restored | he is the certified starter; the ypa rank was wrong |

This is a defect in `player_form` worth fixing upstream. The screen only stops
it reaching a ticket.

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

### SAFE — 3 legs, P 0.668

| player | rank | usage | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|---|
| Alvin Kamara | RB1 | rush 39.0% | 15+ rush yds | 31.0 | +16.0 | 0.896 | NO-NYG |
| Omarion Hampton | RB1 | rush 50.5% | 20+ rush yds | 39.8 | +19.8 | 0.882 | KC-LAC |
| Dalton Schultz | TE1 | tgt 18.6% | 20+ rec yds | 40.0 | +20.0 | 0.845 | HOU-JAX |

### MODERATE — 4 legs, P 0.450

| player | rank | usage | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|---|
| Wan'Dale Robinson | WR2 | tgt 24.9% | 20+ rec yds | 46.8 | +26.8 | 0.899 | IND-TEN |
| Cooper Kupp | WR3 | tgt 13.1% | 10+ rec yds | 28.1 | +18.1 | 0.855 | DEN-SEA |
| Kyle Pitts | TE1 | tgt 16.7% | 20+ rec yds | 34.6 | +14.6 | 0.798 | ATL-CHI |
| Bucky Irving | RB1 | rush 57.8% | 10+ rec yds | 17.5 | +7.5 | 0.734 | PIT-TB |

### HIGH — 5 legs, P 0.152

| player | rank | usage | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|---|
| Omarion Hampton | RB1 | rush 50.5% | 30+ rush yds | 39.8 | +9.8 | 0.721 | KC-LAC |
| Trevor Lawrence | QB1 | starter | 225+ pass yds | 256.3 | +31.3 | 0.691 | HOU-JAX |
| Cade Otton | TE1 | tgt 17.8% | 30+ rec yds | 39.4 | +9.4 | 0.691 | PIT-TB |
| Cam Ward | QB1 | starter | 200+ pass yds | 225.2 | +25.2 | 0.671 | IND-TEN |
| Rome Odunze | WR1 | tgt 20.5% | 40+ rec yds | 49.0 | +9.0 | 0.658 | ATL-CHI |

Both QB legs carry the promoted M89/M90 synthesis with
`model_proj == qb_synthesis_proj`.

### LOTTERY — 4 legs, P 0.007

| player | rank | leg | model | cushion | P | game |
|---|---|---|---|---|---|---|
| Kyle Pitts | TE1 | 40+ rec yds | 34.6 | -5.4 | 0.379 | ATL-CHI |
| Dalton Schultz | TE1 | 50+ rec yds | 40.0 | -10.0 | 0.306 | HOU-JAX |
| Wan'Dale Robinson | WR2 | 60+ rec yds | 46.8 | -13.2 | 0.265 | IND-TEN |
| Alvin Kamara | RB1 | 40+ rush yds | 31.0 | -9.0 | 0.239 | NO-NYG |

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
