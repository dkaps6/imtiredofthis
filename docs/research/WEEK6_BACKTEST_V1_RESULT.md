# 2026 Week 6 Backtest — Frozen Result (V1)

Status: **frozen result**. Produced 2026-10-20 from the committed board, not
from an expiring artifact.

## Provenance

| field | value |
|---|---|
| season / week | 2026 / 6 |
| board | `data/market_track_record/boards/2026_wk06.csv` |
| graded board ledger | `data/market_track_record/graded/2026_wk06_graded.csv` |
| graded card ledger | `data/market_track_record/graded/2026_wk06_cards_graded.csv` |
| preregistered cards | `data/market_track_record/cards/2026_wk06_preregistered_cards_v1.csv` |
| grading run | `39437225099` (job `118677327074`) |
| ledger commit | `6b34cb24` |
| population label | **`UNGATED_ALL_PRICED_ROWS`** |

### Population warning — do not pool

There is **no preserved pregame production decision-gate evidence for 2026
week 6**. Weeks 1-2 are graded on the production-decided population
(`PRODUCTION_DECISION_GATES_V1.json`); Week 6 keeps every priced row. The
rows are labelled `UNGATED_ALL_PRICED_ROWS` precisely so the two populations
cannot be pooled by accident. Any cross-week aggregate that mixes them is
invalid.

The gate evidence for Week 6 was never made durable and **cannot be
reconstructed after kickoff** — it records which teams were already locked
and which player-markets the pregame workbook blocked. That is a permanent
loss for this week, of the same kind as the Weeks 3-5 priced boards
(AGENTS.md §5a).

## Settlement integrity

| field | value |
|---|---|
| archived bet rows | 312 |
| selected settlement rows | 312 |
| verified actual rows | 308 |
| via stats table | 301 |
| via snap-confirmed zero | 7 |
| void (DNP) | 4 |
| still unresolved | **0** |
| unresolved identity statuses | none |

## Board result — 308 decided bets

| | bets | W-L | hit | units | ROI | mMAE | vMAE | closer |
|---|---|---|---|---|---|---|---|---|
| all markets | 308 | 168-140 | 0.545 | +11.77 | +0.038 | 16.58 | 15.92 | 0.474 |

### Finding 1 — the model was less accurate than the market and still profitable

Model MAE 16.58 against Vegas MAE 15.92. The model was closer to the realised
number on only **47.4%** of decided rows — it lost the accuracy contest — and
returned **+3.8%**.

This is the first clean single-week measurement of the proposition that side
selection and line accuracy are separable. It does not license ignoring
accuracy: three of five markets are negative, and the two profitable ones are
exactly where accuracy is competitive or better.

### Finding 2 — by market, edge and accuracy agree in one place only

| market | bets | W-L | hit | units | ROI | mMAE | vMAE | closer |
|---|---|---|---|---|---|---|---|---|
| rush_yards | 62 | 42-20 | 0.677 | +17.01 | +0.274 | 14.68 | 15.98 | **0.597** |
| receptions | 94 | 51-43 | 0.543 | +5.26 | +0.056 | 1.73 | 1.73 | 0.500 |
| rec_yards | 105 | 53-52 | 0.505 | −5.10 | −0.049 | 20.38 | 18.68 | 0.390 |
| rush_rec_yards | 25 | 12-13 | 0.480 | −2.33 | −0.093 | 32.50 | 31.18 | 0.480 |
| pass_yards | 22 | 10-12 | 0.455 | −3.07 | −0.140 | 49.11 | 45.86 | 0.409 |

`rush_yards` is the **only** market where the model beat Vegas on MAE and was
closer more often than not. It is also the largest positive-ROI market. That
co-occurrence is the one place this week where the edge has an accuracy
explanation behind it rather than a side-selection one.

### Finding 3 — RB is the entire week

| position | bets | W-L | hit | units | ROI | mMAE | vMAE | closer |
|---|---|---|---|---|---|---|---|---|
| RB | 103 | 67-36 | 0.650 | **+24.80** | +0.241 | 17.16 | 17.70 | 0.553 |
| QB | 44 | 23-21 | 0.523 | −0.64 | −0.015 | 29.33 | 27.64 | 0.432 |
| TE | 50 | 26-24 | 0.520 | −0.16 | −0.003 | 11.17 | 10.08 | 0.420 |
| WR | 111 | 52-59 | 0.468 | −12.22 | −0.110 | 13.41 | 12.26 | 0.441 |

RB returned **+24.80u**; QB, WR and TE summed to **−13.02u**. The profitable
total is one position's result net of three positions' losses. A week-level
"+3.8% ROI" headline that does not say this is misleading.

Position x market, the concentrated cells:

| cell | bets | W-L | hit | units | ROI | closer |
|---|---|---|---|---|---|---|
| RB rush_yards | 38 | 28-10 | 0.737 | +14.82 | +0.390 | 0.684 |
| RB receptions | 20 | 14-6 | 0.700 | +7.71 | +0.385 | 0.500 |
| RB rec_yards | 22 | 13-9 | 0.591 | +2.60 | +0.118 | 0.409 |
| QB rush_yards | 22 | 13-9 | 0.591 | +2.43 | +0.110 | 0.455 |
| TE rec_yards | 27 | 15-12 | 0.556 | +1.17 | +0.044 | 0.296 |
| RB rush_rec_yards | 23 | 12-11 | 0.522 | −0.33 | −0.014 | 0.522 |
| WR receptions | 51 | 26-25 | 0.510 | −1.11 | −0.022 | 0.471 |
| TE receptions | 23 | 11-12 | 0.478 | −1.33 | −0.058 | 0.565 |
| QB pass_yards | 22 | 10-12 | 0.455 | −3.07 | −0.140 | 0.409 |
| WR rec_yards | 56 | 25-31 | 0.446 | −8.88 | −0.159 | 0.429 |

This is a single week. It is consistent with the promoted RB work but
**cannot promote anything on its own**, and the RB concentration is as
plausibly a Week-6 environment fact as a model property.

### Finding 4 — the alt-line venue cannot express the model's profitable side

| side | bets | W-L | hit | units | ROI |
|---|---|---|---|---|---|
| OVER | 95 | 50-45 | 0.526 | +1.03 | +0.011 |
| UNDER | 213 | 118-95 | 0.554 | +10.75 | +0.050 |

**69% of decided volume and 91% of profit sat on UNDERs.**

The bet-slip construction rule targets FanDuel alt lines, which exist only
for **+ yardages**. The card is therefore **OVER-only by construction** and
structurally cannot carry the side where this week's money was.

This is a property of the venue, not a defect in the rule, but it has a hard
consequence: **a card hit rate is not an estimate of the model's hit rate.**
They sample different halves. Comparing them as if they measure the same
thing is an error, and this document is the first place the asymmetry is
quantified.

Side balance by market (`WIN`/`LOSS`/`VOID`):

| market | side | LOSS | VOID | WIN |
|---|---|---|---|---|
| pass_yards | OVER | 5 | 0 | 5 |
| pass_yards | UNDER | 7 | 0 | 5 |
| rec_yards | OVER | 24 | 0 | 20 |
| rec_yards | UNDER | 28 | 2 | 33 |
| receptions | OVER | 8 | 0 | 12 |
| receptions | UNDER | 35 | 2 | 39 |
| rush_rec_yards | OVER | 1 | 0 | 0 |
| rush_rec_yards | UNDER | 12 | 0 | 12 |
| rush_yards | OVER | 7 | 0 | 13 |
| rush_yards | UNDER | 13 | 0 | 29 |

### Finding 5 — pass_yards: no effect, again

| | bets | W-L | hit | units | ROI | mMAE | vMAE | closer |
|---|---|---|---|---|---|---|---|---|
| pass_yards (dedup) | 22 | 10-12 | 0.455 | −3.07 | −0.140 | 49.11 | 45.86 | 0.409 |

- mean individual breakeven: 0.5292
- observed hit rate: 0.4545, 95% CI [0.2465, 0.6626]
- independent game clusters: 13
- **cluster-aware one-tail p: 0.75503**

Week 6 independently re-confirms the Week-3 postmortem conclusion. The
QB/pass-yards narrowing recommendation raised and retracted earlier in this
line of work **stays retracted**. Do not re-open it on this week's RB result
by symmetry, and do not add a QB/pass-yards carveout.

## Preregistered card result

Both cards were preregistered in
`docs/research/WEEK6_BLIND_FULL_SLATE_ALT_LINE_CARD_V1.md` before outcomes
were read, and graded at their own alt thresholds.

| card | legs | hit | tickets won |
|---|---|---|---|
| FULL_SLATE_BLIND | 7-9 | 43.8% | 0 of 4 |
| MNF_SF_WAS | 9-2 | 81.8% | 3 of 4 |

Tier detail:

| card | tier | result | legs |
|---|---|---|---|
| FULL_SLATE_BLIND | SAFE | LOST | 2/3 |
| FULL_SLATE_BLIND | MODERATE | LOST | 3/4 |
| FULL_SLATE_BLIND | HIGH | LOST | 1/5 |
| FULL_SLATE_BLIND | LOTTERY | LOST | 1/4 |
| MNF_SF_WAS | SAFE | **WON** | 2/2 |
| MNF_SF_WAS | MODERATE | **WON** | 3/3 |
| MNF_SF_WAS | HIGH | **WON** | 3/3 |
| MNF_SF_WAS | LOTTERY | LOST | 1/3 |

### Finding 6 — the declared availability experiment returned null

Loss attribution across all 27 legs: **11 MODEL_MISS, 0 AVAILABILITY_MISS,
0 STATE_UNRESOLVED.** All 27 legs resolved `RESOLVED_GSIS`; all 27 players
participated.

The full-slate card explicitly declared that it skipped the §4 availability
pass; the MNF card applied it. **No loss on either card was an availability
loss.** The 81.8%/43.8% split therefore has nothing to do with the
availability layer. The declared experiment produced a null result.

### Finding 7 — the card split is concentration, not skill

The MNF card's 11 legs cover **3 players**: Jayden Daniels on all four tiers,
Rachaad White on all four, Brock Purdy on three. Daniels (300 pass yds) and
White (27 rush yds) account for 8 of the 9 hits.

That is a violation of the construction rule's own §8 ("no player on more
than two tickets unless there is an explicit reason"), and it is the
mechanism behind the 3-of-4 ticket result. Effective n is approximately **3
correlated bets**, two of which landed — not 11 independent legs.

The full-slate card spread 16 legs over 11 players, much nearer independence.
The same concentration rule cuts the other way there: Omarion Hampton (18
rush yds against a 39.8 projection) appeared on two tiers and cost both.

**Conclusion: 81.8% vs 43.8% is mostly the difference between a concentrated
bet that won and a diversified one that did not.** It is not evidence that
single-game analysis outperforms slate-wide analysis, and it must not be
cited as such.

### Calibration — stated probability vs realised

| stated band | n | mean stated | realised | gap |
|---|---|---|---|---|
| 0.00-0.30 | 5 | 0.207 | 0.200 | −0.7pp |
| 0.30-0.50 | 2 | 0.343 | 0.500 | +15.7pp |
| 0.50-0.70 | 7 | 0.651 | 0.571 | −8.0pp |
| 0.70-0.85 | 7 | 0.770 | 0.714 | −5.6pp |
| 0.85-1.01 | 6 | 0.878 | 0.833 | −4.5pp |

Residual overconfidence of 4-8pp in the mid and high bands, against the
**−29.2pp** gap measured over weeks 1-3. That is a real directional
improvement. **n=27 and one week cannot promote it**, and two of the five
bands have n<=5.

## Open items — not results

1. **TB team-level discontinuity.** Both `investigate_low_output` flags are
   Tampa Bay: Bucky Irving 3 rec yds (projection 17.5) and Cade Otton 4 rec
   yds (projection 39.4), both with `snap_participated=True`. Two skill
   players on the same team collapsing together is one team-level event, not
   two independent model misses. It must be explained under the anomaly gate
   before it is counted as evidence in either direction. Snap participation
   cannot see an early exit, which is why the grader flags rather than
   classifies these.

2. **Selection bias is unresolved.** The falsification check reports 28/32
   teams with any graded row, `pass_yards` covering 22 teams, and six teams
   holding a board row with no graded `pass_yards` bet: BAL, BUF, KC, SF, TB,
   WAS. Verbatim: "10 teams unaccounted for; selection bias is unresolved
   until each is explained." Every per-market rate above inherits this.

3. **Quote selection.** 147 of 312 graded rows (47.1%) were priced away from
   cross-book consensus, median absolute gap 0.50; 21 rows had the projection
   inside the book spread. 71 distinct `vegas_odds` values, so the result is
   not an artefact of uniform pricing.

4. **Upstream `player_form` role defects remain unfixed.** The two defects
   found while building this card (`depth_role` is an alignment slot, not a
   usage rank; `model_role` ranks QBs by descending ypa) are currently
   blocked only at the candidate screen. The underlying artifact is still
   wrong for any other consumer.

## What this week does and does not authorise

Authorised:

- recording the accuracy/profit divergence as a measured single-week fact;
- tracking `rush_yards` / RB as the cell where edge and accuracy co-occur;
- treating the OVER-only alt-line constraint as a known structural limit on
  what any card can express.

Not authorised:

- promoting anything on one week;
- pooling this ungated population with the gated weeks 1-2 ledger;
- re-opening QB pass-yards mean research (p=0.755 here, after 0-of-54
  surviving slices at q=0.10 in the Week-3 postmortem);
- citing the MNF card as evidence for single-game over slate-wide method;
- reading the TB rows as model misses before the anomaly gate clears them.
