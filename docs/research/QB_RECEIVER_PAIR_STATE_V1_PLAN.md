# QB-Receiver Pair State V1 — Frozen Source / Support Audit

**STATUS: FROZEN BEFORE RESULT. SOURCE/SUPPORT ONLY. NO PREDICTIVE CANDIDATE.**

## Purpose

Test whether free nflverse/nflreadpy play-by-play can support a genuinely individual pregame state object:

`passer identity x receiver identity`

The football question is:

> Does the public data let us know, before a game, how much history this exact receiver has accumulated with this exact quarterback — separately from the receiver's generic history, the quarterback's generic history, position, and team environment?

This audit does **not** test whether pair history predicts receiving yards. It establishes whether the object is real, sufficiently covered, and live-deployable.

## Novelty / anti-retest boundary

Adjacent closed/promoted work is protected:

- M38 / WR-R15: receiver entitlement, not QB-receiver pair efficiency/history.
- TE-R5P: TE entitlement, not QB-TE pair history.
- C1/C3: shared QB-receiver target-mass / broad joint state, closed.
- M72 explosive-weapon bridge: aggregates receiving weapons at team level; does not retain passer-receiver pair identity.
- M75: tracking/secondary-quality family, closed.
- R3: same-player residual persistence, closed.
- QB M89/M90: individual QB + team/opponent mean model; do not reopen QB mean.
- WR-CB responsibility work remains source-limited/closed as recorded.

Search of current repo code/docs found no prior `qb_receiver_pair` / passer-receiver pair-history candidate.

A future candidate is not authorized by this audit.

## Sources

Free public only:
- `nflreadpy.load_pbp` for regular seasons 2022-2026;
- `nflreadpy.load_player_stats(..., summary_level="week")` for receiver position identity;
- `nflreadpy.load_schedules` for 2026 Week-5 scheduled teams.

No sportsbook data.

## Prospective 2026 boundary

For 2026:
- only Weeks 1-4 may be read;
- if Week-5-or-later PBP is present, fail closed rather than silently using it;
- no Week-5 outcomes.

## Pair semantic

A pair target event is an official pass-attempt play with:
- non-empty `passer_player_id`;
- non-empty `receiver_player_id`;
- offense team identity.

For every passer-receiver pair, strictly-prior state may include:
- pair games;
- pair targets;
- pair receptions;
- pair receiving yards;
- pair catch rate;
- pair yards per target;
- pair air yards per target;
- pair YAC per reception;
- receiver share of that passer's targets;
- fraction of the receiver's targets delivered by that passer.

These are source fields only. No coefficient or threshold is fit.

## Historical source audit

For each season 2022-2025 report:
- regular-season PBP rows;
- official pass attempts;
- target events;
- passer-ID coverage on target events;
- receiver-ID coverage on target events;
- joint pair-ID coverage;
- distinct passers;
- distinct receivers;
- distinct passer-receiver pairs.

This is source readiness, not outcome scoring.

## Live Week-5 support audit

Use only 2026 Weeks 1-4.

For each Week-5 scheduled offense:
1. define the **current primary passer proxy** as the passer with the most official pass attempts through Week 4;
2. identify every receiver with at least one target from that passer;
3. aggregate pair state through Week 4;
4. attach WR/TE/RB position identity where available.

The current-primary-passer proxy is diagnostic only. It is **not** asserted to be the Week-5 starter and may not be used directly in production without existing starter authority.

Report:
- scheduled teams with a current-primary-passer proxy;
- number of live receiver pairs;
- pair position coverage;
- pair support distribution for >=1, >=5, >=10, >=15 targets;
- pair game support >=1, >=2, >=3 games;
- receiver dependence on primary passer;
- number of receivers targeted by multiple passers in 2026.

## Readiness disposition

`QB_RECEIVER_PAIR_STATE_SOURCE_READY` requires:

1. non-zero regular-season PBP in every 2022-2026 season;
2. 2026 max source week <= 4;
3. joint passer+receiver stable-ID coverage >= 98% on 2026 target events;
4. primary passer proxy available for >=28 of 30 scheduled Week-5 teams;
5. >=100 live current-primary-passer receiver pairs;
6. >=75% position identity coverage on those live pairs;
7. zero Week-5 outcome rows;
8. zero sportsbook inputs;
9. zero candidate models fit;
10. zero production mutations.

If history is present and chronology is clean but one support threshold misses:
`QB_RECEIVER_PAIR_STATE_SOURCE_PARTIAL`.

If chronology or identity safety fails:
`QB_RECEIVER_PAIR_STATE_SOURCE_NOT_READY`.

## What READY would authorize

Only a separately frozen scientific plan.

A future hypothesis would have to test whether pair-specific information adds value **beyond**:
- promoted receiver entitlement;
- receiver own history;
- QB own history;
- team/offense state.

It may not simply replace WR-R15/TE-R5P or retune generic YPT.

No production integration is authorized.

Sportsbook inputs: **0**  
Week-5 outcomes: **0**  
Candidate models: **0**  
Production mutations: **0**
