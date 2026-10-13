# WEEK 1-5 CONTINUITY / WEEK 5 POSTMORTEM SYNTHESIS V1

Date: 2026-10-13

Status: **POSTMORTEM SYNTHESIS COMPLETE. NO WEEK-5 OUTCOME TUNING.**

Parents:
- `docs/handoffs/NFL_HANDOFF_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION_CURRENT.md`
- `docs/research/WEEK5_FULL_MODEL_POSTMORTEM_RESULT_V1.md`
- `docs/research/WEEK5_PLAYER_FAILURE_CASEBOOK_V1_RESULT.md`
- `docs/research/WEEK5_PROSPECTIVE_OPPORTUNITY_LOCK_POSTMORTEM_RESULT_V1.md`
- `docs/research/WEEK5_USER_FACING_TICKET_CONCENTRATION_POSTMORTEM_V1.md`

## Weeks 1-4 selected-board baseline

Closed factual record:
- W1: 204-205, -20.92u
- W2: 192-198, -22.06u
- W3: 233-208, +2.26u
- W4: 220-207, -3.38u

Cumulative W1-W4:
- 849-818
- hit rate: 50.93%
- units: -44.10u
- ROI: -2.65%
- model MAE: 17.89
- selected-line MAE: 16.86
- model closer than line: 46.07%

The W1-W4 downstream confidence problem was severe:
- 70-100% stated-probability rows: n=653
- mean stated probability: 80.52%
- realized win rate: 53.60%
- 57 eligible clustered slices
- 0 BH-FDR survivors

Raw edge was not validated confidence.

## Week 5 selected board

Week 5:
- 350 decided bets
- 199-151
- hit rate: 56.86%
- units: +26.17u
- ROI: +7.48%

By position:
- QB: 25-28, -5.73u
- RB: 75-49, +17.27u
- WR: 73-50, +14.51u
- TE: 26-24, +0.12u

By market:
- rush+rec: 20-6, +11.64u
- receptions: 66-45, +14.11u
- rec yards: 61-51, +3.05u
- pass yards: 14-12, +0.47u
- rush yards: 38-37, -3.10u

## Cumulative selected-board state through Week 5

Pure arithmetic continuity from the frozen weekly records:
- wins: **1,048**
- losses: **969**
- decided bets: **2,017**
- hit rate: **51.96%**
- units: **-17.93u**
- simple cumulative units / decided-bet denominator: approximately **-0.89%**

Week 5 materially reduced the season-to-date selected-board loss, but it did not
erase it.

## Raw projection continuity

Week 5 full football board:
- model MAE: 18.9288
- median preserved-line MAE: 17.8431
- model closer than line: 50.26%
- model direction hit rate vs median line: 55.32%

Compared with W1-W4:
- Week-5 raw model MAE was not materially better than W1-W4;
- the market remained slightly better on raw mean accuracy in Week 5;
- relative model-vs-line competitiveness improved from the W1-W4 46.07%
  closer-than-line rate to 50.26%.

Therefore:
**Week 5 profitability came primarily from downstream selection / price handling,
not from a solved football mean-projection layer.**

Do not claim the projection model is fixed because the selected Week-5 board
made money.

## Week-5 tail failure decomposition

Top 40 absolute final-model player-market misses:
- efficiency-dominant: 19
- opportunity-dominant: 14
- mixed opportunity/efficiency: 7

This is the key Week-5 root-cause result.

The model has at least two distinct unresolved player-level failure families:
1. opportunity / role allocation uncertainty;
2. player efficiency tail uncertainty.

Neither is solved by a generic defense-vs-position multiplier.

## Prospective opportunity-lock evidence

Week-5 lock #1:
- RB 50/50 carry-share + snap-share shadow: worse than recent-carry-share control;
- RB receiving shadow: baseline persistence defect prevents legal comparison;
- WR target-share trajectory: essentially flat/slightly worse;
- TE target-share trajectory: small positive first lock.

No broad opportunity mean allocator is promoted from Week 5.

The strongest independent opportunity evidence instead comes from the separate
three-season uncertainty program:
- individual opportunity-share volatility replicated in 2023, 2024, 2025;
- Beta-Binomial allocation uncertainty historical PASS:
  - RB carry share;
  - WR target share;
- RB target share failed closed;
- TE target share not authorized under that family.

## User-facing ticket continuity

A profitable selected board did not translate reliably into the user's parlays.

Week-5 examples:
- Deshaun Watson repeated across multiple tickets:
  - genuine mixed opportunity/efficiency projection failure;
  - one wrong player thesis destroyed multiple tickets;
  - defect: cross-ticket player concentration.
- BUF-LAR core alt ticket:
  - Allen 200+ hit;
  - Nacua 50+ hit;
  - Kyren Williams 50+ missed with 49;
  - near-threshold variance, not equivalent to Watson.
- higher-payout BUF-LAR expansion:
  - Coleman and Kincaid also failed;
  - modest model-to-alt distance was incorrectly treated as high survivability.

Required downstream distinction:
- selected-board value != calibrated leg survival probability;
- alt distance != confidence;
- one-player concentration != diversification.

## Current Week-6 readiness

As of the 2026-10-13 readiness probe:
- Week-5 stats complete;
- Week-5 snaps complete;
- schedule context available;
- chronology violations = 0;
- current usage coverage among played ~99.46%;
- stable player-ID coverage ~99.85%;
- latest weekly roster authority is still Week 5;
- exact Week-6 roster source unavailable;
- target-week injury rows unavailable.

Disposition:
`PLAYER_STATE_FUTURE_WEEK_NOT_READY`

The Week-6 lock must remain fail-closed until exact target-week identity is
available. Do not silently carry Week-5 roster authority forward.

## Priority roadmap after postmortem

### Priority 1 — finish opportunity uncertainty, not another mean tweak

The three-season share-volatility signal is the strongest new player-specific
evidence.

Continue:
1. joint team-conserving allocation mechanics;
2. historical joint scoring for RB carry and WR target only;
3. prospective Week-6+ shadow after valid target-week locks exist.

Do not reopen failed RB target / TE target Beta-Binomial cells.

### Priority 2 — isolate efficiency-tail uncertainty separately

Week-5 top failures were 19/40 efficiency-dominant.

A new efficiency lane must be historical and outcome-blind at feature freeze.
It must not be a Week-5 player retune.

Candidate question:
- does strictly-prior individual efficiency volatility / regime state identify
  next-game efficiency-tail risk for QB YPA, receiver YPT/catch efficiency, and
  RB YPC?

This must be separately frozen before scoring.

### Priority 3 — ticket construction becomes its own downstream risk process

Prospectively:
- cap repeated non-lock player exposure across paid tickets;
- never call an alt leg safe merely because line < model mean;
- track failure type and portfolio concentration;
- keep lottery tickets separate from core process.

## Bottom line

Week 5 provides evidence of **real model signal**, but not a finished model.

What improved:
- downstream selected-board profitability.

What remains unresolved:
- raw player mean accuracy;
- opportunity-role uncertainty;
- efficiency tails;
- calibrated leg survival probability;
- cross-ticket concentration.

The next work must attack those exact unresolved mechanisms, not restart generic
feature hunting and not tune Week-5 outcomes.
