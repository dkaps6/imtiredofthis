# WEEK 5 USER-FACING TICKET / CONCENTRATION POSTMORTEM V1

Date: 2026-10-13

Status: **RETROSPECTIVE TICKET-CONSTRUCTION AUDIT. NO MODEL RETUNING FROM USER-FACING RESULTS.**

Parents:
- `docs/research/WEEK5_FULL_MODEL_POSTMORTEM_RESULT_V1.md`
- `docs/research/WEEK5_PLAYER_FAILURE_CASEBOOK_V1_RESULT.md`
- `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`

## Why this appendix exists

Week 5 exposed a separation between:
1. underlying model / selection quality; and
2. user-facing ticket survivability.

Canonical selected-settlement Week-5 board:
- 350 decided selected bets
- 199-151
- **56.86% hit rate**
- **+26.17 units**
- **+7.48% ROI**

Yet multiple user-facing parlays lost because one or a small number of shared
legs failed.

Therefore ticket construction must be evaluated as its own downstream process.
A profitable selected board can still be turned into losing user tickets by:
- repeating the same player thesis across tickets;
- treating alt-line distance from model mean as if it were a calibrated hit
  probability;
- failing to distinguish projection error from ordinary near-line variance;
- allowing one fragile player to dominate portfolio failure probability.

## Watson concentration case

Known Week-5 user-facing constructions:

High 7:
- Michael Penix Jr 200+ pass
- Kyle Pitts 25+ receiving
- Brenton Strange 30+ receiving
- Evan Engram 15+ receiving
- Jameis Winston 175+ pass
- **Deshaun Watson 175+ pass**
- Isaiah Likely 30+ receiving

Lottery 8:
- Daniel Jones 200+ pass
- Rome Odunze 40+ receiving
- Xavier Hutchinson 15+ receiving
- Jameis Winston 200+ pass
- **Deshaun Watson 200+ pass**
- Isaiah Likely 40+ receiving
- A.J. Barner 20+ receiving
- Greg Dulcich 20+ receiving

User reported:
- Watson finished with **155 passing yards**;
- the other legs on the tickets containing Watson hit.

Canonical model / casebook:
- Watson final model: **238.23**
- actual: **155**
- predicted attempts: 24.77
- actual attempts: 33
- predicted YPA: 7.28
- actual YPA: 4.70
- label: `MIXED_OPPORTUNITY_EFFICIENCY`

Interpretation:
- this was a genuine player projection miss;
- it was NOT primarily an alt-line-selection miss;
- repeating Watson across tickets converted one wrong player thesis into
  multiple ticket losses.

Construction defect:
`CROSS_TICKET_PLAYER_CONCENTRATION`

## Monday BUF-LAR core-alt ticket

Recommended core:
- Josh Allen 200+ passing
- Puka Nacua 50+ receiving
- Kyren Williams 50+ rushing

Final:
- Allen: **210 passing** -> HIT
- Nacua: **138 receiving** -> HIT
- Williams: **49 rushing** -> MISS by **1 yard**

Ticket:
- 2 / 3 legs hit
- one-leg miss
- losing leg missed the alt threshold by one yard

Interpretation:
`ALT_LINE_NEAR_MISS__NOT_EQUIVALENT_TO_WATSON_PROJECTION_FAILURE`

This must not be diagnosed like Watson:
- Watson 155 vs 175/200 was a wrong player thesis;
- Williams 49 vs 50 was ordinary threshold variance around an intentionally
  lowered survival line.

The ticket still lost. The process therefore needs explicit expected ticket
survival / concentration controls rather than treating every model-backed alt
line as equally safe.

## Monday higher-payout suggestion

Suggested expansion:
- Allen 200+ pass
- Nacua 50+ receiving
- Williams 50+ rushing
- Keon Coleman 40+ receiving
- Dalton Kincaid 40+ receiving

Observed known finals:
- Allen 210 -> HIT
- Nacua 138 -> HIT
- Williams 49 -> MISS
- Coleman 7 receiving -> MISS
- Kincaid 7 receiving -> MISS

This higher-payout construction would not qualify as a one-leg near miss.

Key warning:
alt lines cannot be added simply because they sit below the model mean.
Coleman/Kincaid demonstrate that a modest positive model-to-alt gap does not
equal high survival probability when individual role / target opportunity is
uncertain.

## Stafford standalone warning

Recommended separately:
- Matthew Stafford UNDER 278.5 pass

Canonical model:
- ~242.3 pass yards

Actual:
- **383**

Casebook:
- Stafford is a large Week-5 opportunity-dominant miss.

This was a bad player/game thesis, not a pricing near miss.

## Required ticket-construction changes

These are PROCESS controls, not model-science retunes:

1. **Player concentration cap**
   - one non-lock player thesis may not appear across multiple paid tickets by
     default;
   - repeated exposure requires explicit portfolio-level justification.

2. **Separate projection confidence from alt-line distance**
   - model mean > alt line is insufficient;
   - no leg may be called a survival/lock leg without a validated or explicitly
     conservative uncertainty assessment.

3. **Failure-type tagging before reuse**
   - near-threshold variance;
   - projection/opportunity miss;
   - efficiency collapse/explosion;
   - stale-role / current-role uncertainty;
   - identity/injury/availability problem.

4. **Ticket-level survival accounting**
   - evaluate portfolio probability of at least one ticket surviving;
   - do not optimize only individual leg plausibility.

5. **Do not duplicate the same failure mode**
   - avoid tickets dominated by one player, one game, one position/model seam,
     or one correlated offensive thesis.

6. **Keep entertainment lottery tickets separate**
   - lottery construction may be long;
   - it must not be confused with the core process or used as evidence of model
     quality.

## What Week 5 says

The Week-5 selected board itself was profitable.

The user-facing ticket layer did not monetize that value reliably.

Therefore the immediate downstream problem is:
`POSITIVE_SELECTED_BOARD__INSUFFICIENT_PORTFOLIO_CONSTRUCTION`

This is additive to, not a substitute for, continuing player-specific model
research.

## Scientific boundary

Do not use these ticket outcomes to:
- refit player means;
- fit a new model coefficient;
- choose a model feature;
- change a football research source;
- tune a threshold.

Ticket construction may change prospectively because it is a downstream risk /
portfolio process.

Football model changes remain governed by independent historical/prospective
evidence.
