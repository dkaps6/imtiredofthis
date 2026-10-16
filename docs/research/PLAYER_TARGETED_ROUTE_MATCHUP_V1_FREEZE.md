# Player Targeted-Route × Opponent Route-Family Matchup V1 — Frozen Source / Novelty Plan

Date: 2026-10-14

Status: **FROZEN BEFORE SCIENTIFIC SCORING / RESEARCH ONLY**

## Why this lane exists

The model is an individual-player projection system. Recent work confirmed that named-player history matters for WR/TE uncertainty, while broad personnel buckets (11/12/OTHER) failed as a mean/opportunity mechanism.

The latest high-resolution role-state audit established one free historical player-attributed field that is semantically usable without pretending to have total routes run:

- nflverse participation `route` describes the route label associated with the primary/targeted receiver on a play;
- it does **not** describe all routes run;
- it therefore may support a **targeted-route mix** feature, but never route participation, TPRR, YPRR, or route rate.

This plan tests source readiness and novelty for a distinct player-first matchup hypothesis before any projection candidate is allowed.

## Exact scientific hypothesis

For a named WR/TE, define his strictly-prior distribution of **targeted route families** from only plays on which the source assigns that player the target/primary-receiver route label.

For the target-game opponent, define strictly-prior defensive performance allowed on those same route families.

Question:

> Does matching Player A's own prior targeted-route mix to Opponent B's prior route-family susceptibility contain stable pregame information about Player A's next-game target/receiving efficiency beyond current promoted player means?

This is not:
- defense-v-position;
- total routes run;
- WR-CB coverage assignment;
- generic air-yard depth;
- generic team man/zone;
- 11/12 personnel propensity;
- a sportsbook feature.

## Anti-retest boundary

Do not proceed if repository/source audit shows this exact interaction has already been dispositioned.

Known adjacent closed work does not count as this exact test:
- WR target-depth symmetric widening;
- WR NGS target model R9-R11;
- team coverage man/zone ablation;
- generic matchup-transmission candidates;
- M83 defensive adaptive gameplan;
- broad player personnel-conditioned opportunity V1;
- WR-CB projected alignment archive.

If source semantics collapse route family to only air-yard/depth information with no distinct categorical signal, disposition source-limited and stop.

## Stage A — source contract only

Use public nflverse/nflreadpy historical participation + PBP.

Target source seasons: **2022-2025**.

Required source assertions:
1. participation and PBP join on a stable game/play key;
2. a route-label field is present with non-trivial coverage;
3. the route label can be attributed to the targeted/primary receiver identity for that play without inferring non-target routes;
4. player identity can be mapped to stable GSIS/canonical identity;
5. offense team and defense/opponent are available;
6. target/result fields needed only for later grading are available from PBP;
7. no sportsbook field is read;
8. no 2026 target-game outcome is read.

Required source summaries:
- rows by season;
- route-label coverage by season;
- distinct raw route labels and frequencies;
- targeted-route rows with stable receiver identity;
- distinct WR and TE players with >=4 prior eligible games;
- per-player median eligible targeted-route count;
- per-defense route-family support;
- join failures / ambiguous identity rows.

### Frozen source disposition

`PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY` only if all of:
- each of 2023, 2024, 2025 has >=10,000 valid targeted-route rows;
- each scored season has >=150 distinct eligible WR/TE players;
- >=90% of route-labeled target rows have stable receiver identity;
- at least 5 route families each have >=1,000 pooled rows;
- at least 24 defenses each season have >=100 eligible prior route-family observations by Week 8;
- zero semantic violations where route labels are treated as all-routes participation.

Otherwise:
`PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_LIMITED`.

No scientific projection result may be interpreted if Stage A is source-limited.

## Route-family normalization

Freeze only deterministic label normalization before scoring.

Allowed operations:
- lowercase/strip;
- punctuation/space normalization;
- exact synonym folding only where raw labels are clearly lexical variants.

Do **not** merge route labels based on outcome performance.

If the source already supplies a documented finite route taxonomy, preserve it.

Rare route families with <250 pooled rows may be grouped as `OTHER` for support only. `OTHER` may not be used to rescue a failed candidate.

## Stage B — strictly-prior player × opponent matchup diagnostic

Stage B is authorized only after Stage A = SOURCE_READY.

For target player-game g:

### Player targeted-route mix
Use only the player's completed games strictly before g.
- minimum 4 prior player-games;
- trailing 8 eligible player-games maximum;
- count only source-valid targeted-route rows;
- normalize counts to route-family target-mix weights.

### Opponent route-family susceptibility
For each route family r, use only the target opponent's completed defensive plays strictly before g.
- trailing 8 defensive games maximum;
- minimum 100 total eligible defensive route observations;
- compute route-family allowed target completion rate, yards per target, and receiving yards per target;
- same target game never enters history;
- same-week games never enter another game in that week.

### Player-specific matchup expectation
For each metric m:

`matchup_m = sum_r(player_targeted_route_mix_r * opponent_prior_allowed_m_r)`

Also compute a neutral control using league strictly-prior route-family performance:

`neutral_m = sum_r(player_targeted_route_mix_r * league_prior_allowed_m_r)`

Primary diagnostic signal:

`matchup_delta_m = matchup_m - neutral_m`

This makes the hypothesis explicitly player-specific: two teammates facing the same defense can receive different matchup deltas because their own targeted-route mixes differ.

## Frozen primary outcomes

WR and TE separately:
- next-game receiving yards per target;
- next-game catch rate;
- next-game receiving yards.

Target count itself is diagnostic only in V1 because targeted-route mix has no all-route denominator and must not be presented as route participation.

## Frozen validation

Development/confirmation split:
- 2023 development;
- 2024 independent confirmation;
- 2025 independent confirmation.

Do not tune family mappings or thresholds after 2023 output.

Primary requirements to advance to an integration candidate:
1. source/integrity gates all pass;
2. sign-consistent player-level relationship in 2024 and 2025 for at least one predeclared primary metric;
3. player-clustered 95% CI excludes zero in the same direction pooled across 2024-2025;
4. game-clustered 95% CI excludes zero in the same direction pooled across 2024-2025;
5. signal remains after controlling descriptively for the player's strictly-prior target depth / air-yard state if available, so this is not merely a renamed depth signal;
6. WR and TE are reported separately; one cannot rescue the other;
7. no result may be promoted from pooled position MAE alone.

Player-level reporting must include:
- number of qualifying named players;
- median per-player relationship/delta;
- fraction of qualifying players with the expected sign;
- top/bottom player difficulty slices for diagnosis;
- no position-mean-only conclusion.

## Hard integrity rules

- zero sportsbook inputs;
- zero OddsAPI calls;
- zero production changes;
- zero target-game leakage;
- zero same-week reference contamination;
- zero use of route label as total routes run;
- zero inferred WR-CB assignment;
- stable player identity required.

## Promotion boundary

A Stage-B pass does **not** change production.

It authorizes only a separately frozen full-stack candidate asking whether the route-family matchup delta improves the named WR/TE projection beyond M38/WR-R15 or TE-R5P.

A failure closes this exact route-family interaction. Do not rescue by:
- changing history windows;
- regrouping route families after results;
- selecting only favorable routes/players/defenses;
- replacing targeted-route mix with fake route participation;
- adding sportsbook information.

