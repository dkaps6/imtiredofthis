# RB STACK6C / ND4 — Secondary-Back Rotation Source Audit Plan

## Motivation

STACK6B failed its frozen retention gates. The no-fit directional postmortem showed a coherent mechanism: compact/aggregate role information helps when it **contracts** secondary-back workload but expansion calls are harmful, while applying every negative correction creates excessive negative carry bias and still misses the carry-MAE gate.

Therefore do not retune Ridge, search contraction thresholds, or promote a one-sided replay. Audit genuinely new football state first.

The most important unresolved state is **how a backfield actually rotates by drive/series**, not merely aggregate snaps or carries.

## Research question

Can completed prior games provide timestamp-safe rotation information that distinguishes:

- an RB whose aggregate snaps are concentrated in true rushing/series ownership;
- an RB who appears frequently but rotates out of meaningful rushing series;
- a stable secondary reliever from a volatile committee role;
- teams that alternate lead backs by drive from teams that maintain a stable drive hierarchy?

And critically: can a **live-capable PBP touch/drive proxy** preserve enough of the richer historical on-field participation structure to be usable during the 2026 season?

## Source layers

### A. Historical on-field rotation truth

Use nflverse participation joined to nflverse PBP for 2024–2025 regular-season games.

Participation is historical/postseason-release data from 2023 onward. It is permitted only to reconstruct completed prior-game truth and to benchmark the proxy. Target-game participation is forbidden as a pregame feature.

Derive from each offensive play:

- RB/FB identities on field;
- count of RB/FB players on field;
- single-RB vs multi-RB play state;
- game/drive/play order;
- per-player drive presence;
- opening-drive presence;
- player co-presence with another RB/FB;
- drive-level most-present RB and tie rate;
- team drive-leader switching/rotation tendency;
- player share of offensive drives present;
- player share of drives as most-present RB;
- team drive-presence concentration.

### B. Live-capable PBP touch/drive proxy

Using ordinary PBP only for event structure, derive RB rushing/target opportunities by drive:

- rush attempts by drive;
- pass targets by drive where receiver identity is available;
- RB opportunity count = rush attempts + targets;
- drives with an RB opportunity;
- player share of RB opportunities;
- player share of drives with a touch opportunity;
- opening-drive opportunity share;
- most-used RB by touch opportunities on each drive;
- team touch-leader switch rate;
- team touch-opportunity concentration.

For this audit, historical participation IDs may be used only as a position-identity bridge to define the RB/FB universe. The future live implementation must use the canonical production identity/roster/depth bridge, not target-game participation.

## Lagged feature contract

Build player/team prior-game features only:

- prior-1 same-team value;
- prior-3 same-team rolling mean where available;
- feature source order must be strictly less than target season-week order.

No target-game on-field or target-game touch state may enter a pregame feature.

## Predeclared source gates

Infrastructure gates:

1. participation/PBP play-key join rate >= `0.95`;
2. offense player/position array alignment >= `0.95`;
3. drive identifier coverage on joined regular-season offensive plays >= `0.95`;
4. strict-prior lag leakage pass rate = `1.00`.

Live-proxy evidence gates, evaluated descriptively against historical participation truth:

5. player-game touch opportunity share vs player-game RB on-field presence share Pearson correlation >= `0.60`;
6. player-game touch-drive share vs on-field drive-presence share correlation >= `0.60`;
7. team-game top-RB identity agreement between touch-opportunity share and on-field presence share >= `0.70`;
8. 2025 non-null prior-3 coverage for core touch/drive proxy features >= `0.75` among player-games with at least one earlier team game and a known RB/FB identity.

A gate miss does not justify threshold tuning. It identifies which proxy family is not trustworthy enough.

## Forbidden

- no rushing-yard or carry outcome model fit;
- no Ridge/logistic/tree fit;
- no sportsbook data;
- no feature search;
- no threshold search;
- no weight search;
- no production changes;
- no target-game participation as an input feature;
- no claiming delayed participation itself is live-2026 capable.

## Dispositions

If all infrastructure gates and at least three of the four live-proxy evidence gates pass:

`GO_STACK6C_ROTATION_PROXY_BUILD`

Otherwise:

`ROTATION_PROXY_INSUFFICIENT_FIND_NEW_LIVE_SOURCE`

A GO only authorizes freezing a separate model architecture. It does not retain a point-model change.

## Parallel availability boundary

Exact game-day active/inactive competitor identity remains a separate source-qualification problem. Weekly roster/injury datasets may be audited, but no exact inactive indicator can be used until its historical timing is demonstrably pre-kickoff. Target-game participation cannot substitute for that proof.
