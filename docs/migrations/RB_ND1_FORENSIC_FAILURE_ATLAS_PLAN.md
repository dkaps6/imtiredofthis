# RB-ND1 — Forensic Failure Atlas / Reverse-Engineering Audit

## Purpose

Reverse-engineer the 2025 RB rushing-yard misses game by game before proposing another model change. The goal is to identify **why** M94C/M96E missed, quantify which failure mechanisms account for the most error, and map those mechanisms to leakage-safe pregame football information that can be tested in later migrations.

This is a diagnostic migration. It is not a coefficient search and does not change production.

## Frozen parents

- M94C authoritative run `33353485070` / artifact `migration-94c-rb-game-environment`.
- M96E authoritative run `33467630395` / artifact `migration-96e-rb-role-workload-risk-guard`.
- RB market benchmark authoritative run `33499129109` / artifact `rb-market-benchmark-2025`.

Sportsbook lines remain downstream benchmark/diagnostic information only. They may not be used to create a football feature, fit a coefficient, choose a threshold, or route a prediction.

## Core decomposition

For every M94C RB/FB player-game, decompose projected carries as:

`player carries = team rush attempts × player team-rush share`

Use an exact two-factor Shapley decomposition of the difference between projected and actual carries into:

1. **team-volume contribution**, and
2. **player-share / backfield-allocation contribution**.

Then decompose rushing yards as:

`rushing yards = carries × yards per carry`

using the same symmetric two-factor decomposition into:

1. **opportunity/carry contribution**, and
2. **efficiency contribution**.

Actual carries/YPC and target-game play-by-play are postgame forensic facts only. They may never become pregame candidate features from this audit.

## Additional forensic context

Build target-game postgame diagnostics from nflverse PBP where source coverage permits:

- actual offensive plays and team rush volume;
- actual mean score differential / lead-neutral-trail play shares;
- neutral early-down rush rate;
- RB-vs-non-RB rushing competition;
- player max rush and 10+/20+ explosive run counts;
- player rush success rate;
- official target-game injury-report fields are carried only as a timestamped **pregame clue** when nflverse reports the same week, never forward-filled from another week.

M94C's own pregame game-environment projections are retained alongside the postgame facts so game-script misses can be diagnosed without confusing prediction with outcome.

## Precommitted classifications

Carry failure classification is descriptive only:

- `TEAM_VOLUME` when the absolute team-volume contribution is at least 1.25× the player-share contribution;
- `PLAYER_SHARE` when the reverse is true;
- otherwise `MIXED`.

Yard failure classification:

- `OPPORTUNITY` when the absolute opportunity contribution is at least 1.25× the efficiency contribution;
- `EFFICIENCY` when the reverse is true;
- otherwise `MIXED`.

Special forensic flags are reported separately, not used to override the primary class:

- zero/near-zero realized role after a material projection;
- Week-1/new-role initialization clue (very low prior/base carries but meaningful realized workload);
- game-script miss;
- explosive-run shock;
- substantial non-RB/QB rushing competition;
- market-covered large disagreement.

No post-result threshold adjustment is allowed in RB-ND1. If a flag definition proves mechanically impossible because a required source column is absent, the smallest schema-compatible repair may be made and documented; scientific thresholds stay frozen.

## Outputs

1. Full 2025 player-game forensic trace.
2. Large-miss casebook, sorted by absolute rushing-yard error.
3. Error attribution summary by primary opportunity/share/efficiency mechanism.
4. Share of total absolute error attributable to each mechanism.
5. Market-covered summary showing where Vegas's advantage sits by forensic class.
6. M96E-vs-M94C improvement by forensic class on the M96E-authoritative evaluation window.
7. Game-script/explosive/competition summaries.
8. Pregame-proxy map: for each diagnosed mechanism, list the football-only pregame data family that could plausibly address it.

## Decision rule

RB-ND1 does **not** promote a model. It selects the next research family by evidence:

- share/role collapse dominant -> current depth chart, injury/availability, transactions, snaps/participation, competitor-role and rookie/new-team initialization audit;
- team volume/game script dominant -> team play-calling, pace, opponent interaction, game-state/score distribution architecture;
- efficiency dominant -> blocking/OL, defensive front, runner-created efficiency, scheme/concept matchup architecture;
- explosive shock dominant -> distribution/tail model, not mean inflation;
- mixed -> build modular experts only after each component has a leakage-safe pregame proxy.

The next migration must be justified by this atlas rather than by trying arbitrary feature combinations.