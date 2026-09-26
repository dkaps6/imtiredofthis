# Receiving Vacancy Prospective Audit V1 — Frozen Plan

Date frozen: 2026-09-26
Status: **PREGAME PROSPECTIVE OBSERVATIONAL AUDIT — NO MODEL CHANGE**

Branch: `research-public-intent-week3-prospective-v1`

## Purpose

Evaluate whether production's current handling of definitive WR/TE absences — remove unavailable player first, then generically normalize surviving receiving entitlement — correctly identifies **which surviving receivers absorb the vacated opportunity and how concentrated that absorption becomes**.

This plan is frozen before Week-3 target-game outcomes are read.

It is downstream of the confirmed systems finding:
`AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`

It is not a repair candidate.

## Frozen authority

Current production / pregame authority:
- Full Slate no-live-odds run: `36204768034`
- artifact: `10892728623`
- source main: `f7d2011b73950488ea209124ba895b92c401b2b1`
- sportsbook acquisition: disabled
- target outcomes: unread

Rule-order diagnostic authority:
- run: `36275905038`
- artifact: `10917451964`
- digest: `sha256:340b5992a58ff591baf7a4ccf92c36482c1e22a082fabdc14e2f4e7dbcb1f6fb`

A same-authority output-only replay may persist exact per-player target-entitlement rows. It may not change the cohort, formulas, or methodology.

## Frozen Week-3 cohort

Include **every** current-eligible Week-3 player satisfying:
- position group WR or TE;
- canonical `definitive_unavailable == 1`;
- team remains in the certified current production universe.

No target-share threshold.
No hand-picked teams.
No exclusions based on expected importance.

Frozen unavailable players:
- HOU — Nico Collins — WR
- IND — Ashton Dulin — WR
- LAC — Charlie Kolar — TE
- LAC — Brenen Thompson — WR
- MIA — Caleb Douglas — WR
- NO — Barion Brown — WR
- NYJ — Mason Taylor — TE
- SF — Demarcus Robinson — WR
- WAS — Chig Okonkwo — TE

Nine unavailable receivers across eight teams.

## Baseline to freeze

For every affected team, persist the exact pregame production survivor state after:
1. availability filtering;
2. PlayerForm/Bayesian baseline;
3. simulation rules;
4. M38 WR hierarchy sharpening;
5. TEAM_TARGET_ENTITLEMENT_V1 finite-player allocation semantics.

For every surviving RB/FB/WR/TE receiving option, freeze:
- player identity;
- position;
- rules target share;
- final entitlement target share;
- team modeled-player target mass;
- residual target mass;
- unavailable teammate identity/position;
- unavailable teammate most-recent same-team strict-prior target share when available.

This baseline is descriptive production state, not a candidate.

## Primary postgame questions

After each affected game is final:

1. **Successor identity**
   - Which surviving player(s) produced the largest positive actual-target-share residual versus frozen entitlement?
   - Did the production baseline already rank the eventual largest vacancy absorber near the top of the surviving room?

2. **Successor concentration**
   - Was actual receiving opportunity materially more concentrated or more diffuse than the frozen production entitlement?
   - Compare top-1 share, top-2 share, and Herfindahl concentration across surviving target earners.

3. **Entitlement error**
   - For each surviving receiving option, compute absolute error between frozen target entitlement and actual target share.
   - Report team-level mean and p90 absolute error where sample permits.

4. **Vacancy absorption**
   - Compare unavailable player's frozen strict-prior share with the aggregate positive residual earned by surviving receivers.
   - This is descriptive only; do not assume one-for-one transfer.

## Prespecified descriptive metrics

Per affected team:
- actual team pass attempts and targets where source definitions support them;
- survivor actual targets;
- survivor actual target share;
- frozen entitlement target share;
- residual = actual target share - frozen entitlement;
- absolute residual;
- baseline rank by entitlement;
- actual rank by targets/share;
- top-1 frozen vs actual concentration;
- top-2 frozen vs actual concentration;
- frozen vs actual HHI;
- identity of largest positive-residual survivor.

Across cohort:
- median survivor absolute target-share error;
- mean survivor absolute target-share error;
- percentage of teams where actual top target earner matched frozen top entitlement player;
- percentage where largest positive-residual player was already top-2 frozen entitlement;
- direction of concentration error by team.

No single numeric gate authorizes production.

## Interpretation labels

Possible observational dispositions:
- `GENERIC_NORMALIZATION_DIRECTIONALLY_ADEQUATE`
- `VACANCY_SUCCESSOR_IDENTITY_GAP_SUPPORTED`
- `VACANCY_CONCENTRATION_GAP_SUPPORTED`
- `MIXED_RECEIVING_VACANCY_EVIDENCE`
- `INSUFFICIENT_REALIZED_VACANCY_SIGNAL`

These are descriptive labels only.

## Relationship to RB Vacancy Opportunity V1

RB Vacancy Opportunity V1 is separate and remains frozen.

Postgame order:
1. grade RB Vacancy Opportunity V1 independently;
2. grade DEN/PIT public-intent observation;
3. grade this receiving-vacancy observational audit.

Do not use this receiving result to reinterpret or rescue the RB experiment.

## No-go rules

Do not:
- read Week-3 target-game outcomes before the baseline snapshot is frozen;
- change the nine-player cohort after outcomes;
- exclude low-share unavailable players after seeing results;
- resurrect the legacy WR 60/30/10 formula;
- fit a transfer coefficient;
- tune recipient weights;
- search thresholds/windows after outcomes;
- use sportsbook lines/odds;
- change M38, WR-R15, TE-R5P, ensemble weights, residual cap, or target-volume model;
- promote from one Week-3 observational cohort.

Any future repair must receive its own separately frozen historical/prospective design after this audit is graded.
