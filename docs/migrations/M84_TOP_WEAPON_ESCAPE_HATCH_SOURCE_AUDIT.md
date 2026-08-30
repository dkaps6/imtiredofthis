# Migration 84 — Top Weapon Escape Hatch Source / Feasibility Audit

## Status

`PREREGISTERED / SOURCE AUDIT ONLY`

M84 does **not** fit a QB passing-yards model. It tests whether materially new pregame receiver-vs-defender/responsibility matchup information exists with enough historical and in-season coverage to support a later predictive migration.

## Why M84 exists

M82 established the clean full-stack QB passing benchmark at `56.749517` MAE on 884 canonical-v3 QB games and preserved `TOP_WEAPON_ESCAPE_HATCH` as a source-blocked new-information frontier. M83 then rejected the same-information defensive-adaptation mechanism.

The M84 hypothesis is narrower than prior receiver work:

> A QB may materially exceed an otherwise neutral/bad macro matchup when one WR/TE/RB has an unusually exploitable **pregame individual matchup** and the offense can funnel valuable routes/targets through that player.

Realized receiving yards are not evidence because they are mechanically contained in QB passing yards. M84 therefore audits only information that can exist before kickoff.

## Prior-work anti-retest boundary

M84 may not relabel the following as new:

- M72 aggregate explosive-weapon x defense matchup.
- M75 NGS receiver separation/cushion/aDOT/YACOE.
- M75 PFR secondary aggregate coverage statistics.
- M75 receiver x secondary interactions built from those same aggregates.
- generic target share, route share, WR hierarchy, or receiving efficiency already present in M31-M38.
- postgame realized WR/TE/RB production.

A qualifying source must add a materially new observable such as:

- receiver-to-defender matchup/responsibility identity;
- route/alignment exposure against specific defender responsibilities;
- defender help/coverage responsibility;
- role-specific defensive replacement/injury matchup context;
- another direct pregame observable that identifies an exploitable single-weapon edge beyond aggregate receiver/secondary quality.

## Candidate source families

M84 audits at minimum:

1. **NFL Next Gen Stats Coverage Responsibility**
   - exact defender-to-receiver responsibility/matchup concept is scientifically ideal;
   - audit whether machine-readable historical and in-season public access exists.

2. **NFL Big Data Bowl / PFF scouting slices**
   - exact coverage-assignment and primary/secondary matchup IDs may exist in competition datasets;
   - audit historical span, completeness, reproducibility, and whether the source can extend to target seasons/live use.

3. **nflverse participation / route data**
   - audit route fields, player identity granularity, defender matchup identity, historical coverage, and in-season update status.

4. **public WR/CB matchup tools/reports** such as Fantasy Points/VSiN
   - audit whether a stable machine-readable historical archive and repeatable in-season source contract exists.

5. **PFF WR/CB matchup products**
   - audit only as a source-contract reference; paid/subscription-only access does not qualify for the current free-source research path.

## Frozen qualification contract

A source family can advance only if it satisfies all of the following:

1. **Pregame observability** — no target-game realized outcome/charting.
2. **Material novelty** — not equivalent to M72/M75 aggregate receiver/secondary information.
3. **Receiver-level identity** — identifies the weapon, not only team-level defense quality.
4. **Defender/responsibility specificity** — identifies a defender, coverage responsibility, alignment/route exposure, or equivalent direct matchup mechanism.
5. **Historical development coverage** — sufficient to construct leakage-safe histories for at least 2023-2025 research or another explicitly justified multi-season development window.
6. **In-season deployability** — updates during the season quickly enough to support 2026 pregame use.
7. **Stable/repeatable acquisition** — not dependent on screenshots, manual transcription, one-off editorial prose, or a competition-only slice that cannot be extended.
8. **Free-source contract for this phase** — no paid data dependency is introduced in M84.

A source that is scientifically ideal but unavailable publicly may be labeled `IDEAL_BUT_PROPRIETARY` and does not advance.

## Allowed M84 outputs

- source inventory;
- source URL/status checks;
- field/granularity inventory where machine-readable data are accessible;
- historical-span and in-season-update audit;
- novelty crosswalk to M72/M75;
- final source disposition.

## Prohibited M84 actions

- no QB outcome fitting;
- no 2024/2025 passing-yards selection;
- no Ridge/HGB/XGB/NN modeling;
- no receiver big-game correlation study;
- no sportsbook variables;
- no post-result source substitution to rescue the hypothesis.

## Frozen final dispositions

Exactly one of:

- `QUALIFIED_FOR_M85_PREDICTIVE_DEVELOPMENT`
- `HOLD_SOURCE_BLOCKED_NEW_INFORMATION`
- `CLOSED_NO_MATERIALLY_NEW_SOURCE`

If no free source satisfies the complete historical + live + matchup-specific contract, M84 must stop. A later revisit requires a genuinely new source, not another transformation of M72/M75 data.

## M85 boundary

Only if M84 qualifies a source may M85 test whether a pregame top-weapon edge predicts QB residuals conditional on macro QB matchup and whether it reduces the authoritative M82 full-stack benchmark (`56.749517` MAE), attempts/YPA error, or 100+ yard misses.

`production_actionable = false`
