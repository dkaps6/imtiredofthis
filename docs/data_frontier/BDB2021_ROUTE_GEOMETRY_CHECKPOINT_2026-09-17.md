# BDB 2021 Route / Nearest-Defender Geometry Checkpoint — 2026-09-17

## Scope

Isolated NFL data-frontier engineering only. No predictive experiment, no production-science change, no sportsbook change, no Issue #535 change.

- Branch: `data-frontier-phase0-bdb-contact-v1`
- Official source: Kaggle `nfl-big-data-bowl-2021`
- Source-manifest SHA-256: `55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4`
- Canonical source-audit run: `35265684261`
- Canonical route-contract run: `35265867949`
- Canonical Week-1 route-geometry run: `35266104143`
- Canonical all-weeks route-geometry run: `35266277669`
- Raw competition files uploaded: **NO**

## Official corpus

All expected files were present:

- `games.csv`: 253 rows
- `players.csv`: 1,303 rows
- `plays.csv`: 19,239 rows
- `week1.csv` through `week17.csv`

Weekly tracking files contain 19 columns including:

`time`, `x`, `y`, `s`, `a`, `dis`, `o`, `dir`, `event`, `nflId`, `displayName`, `jerseyNumber`, `position`, `frameId`, `team`, `gameId`, `playId`, `playDirection`, `route`.

The play table has no direct defender-responsibility assignment field. It provides play context such as formation, personnel, defenders in box, pass rushers, dropback type, pass result and EPA.

## Route-label contract

Across Weeks 1-17:

- tracked player-plays: 263,173
- route-labeled player-plays: **78,343**
- player-plays with more than one non-null route label: **0**
- plays missing football tracking: **0**
- route labels appearing on the non-offense tracking side: **0** in the all-weeks geometry contract

Observed route taxonomy:

- `GO`: 14,890
- `HITCH`: 11,857
- `FLAT`: 11,310
- `OUT`: 8,057
- `CROSS`: 7,449
- `IN`: 5,656
- `POST`: 5,169
- `SLANT`: 4,349
- `SCREEN`: 3,221
- `ANGLE`: 2,994
- `CORNER`: 2,854
- `WHEEL`: 430
- `undefined`: 107

Route-labeled player-play positions are overwhelmingly receiving/offensive skill positions:

- WR: 46,762
- TE: 16,935
- RB: 13,637
- FB: 466
- HB: 477
- QB: 52
- a very small number of defensive/DL-position anomalies are present in the source labels and are not silently rewritten.

## Event coverage

All weekly files include football tracking. Route geometry is anchored to explicit tracking events only.

Across the full corpus the route geometry run produced:

- snap nearest-defender geometry rows: **78,312**
- throw nearest-defender geometry rows: **77,833**
- arrival nearest-defender geometry rows: **64,879**

The difference between route-label count and event-specific geometry count reflects source event/frame availability and a very small number of event-frame defensive snapshot omissions. The pipeline does not synthesize a coverage defender when no valid geometric snapshot exists.

## Geometry semantics

The geometry lab determines the nearest and second-nearest defender by Euclidean x/y distance at a specified event frame.

**This is proximity only. It MUST NOT be interpreted as true coverage responsibility, matchup assignment, shadow assignment, bracket responsibility, banjo/exchange responsibility, or man/zone responsibility.**

That semantic distinction is mandatory because the project already established in M84 that nearest defender is not an honest substitute for exact responsibility.

## Full-season geometry

Across all available route runners:

| Event | Rows | Nearest defender p10 | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snap | 78,312 | 2.225 | 3.096 | **5.250** | 7.255 | 9.234 | 12.300 |
| Throw | 77,833 | 0.938 | 1.487 | **2.898** | 4.929 | 7.317 | 13.279 |
| Arrival | 64,879 | 0.914 | 1.486 | **2.889** | 5.122 | 7.817 | 14.350 |

Median second-nearest defender distance:

- snap: 8.500 yd
- throw: 7.931 yd
- arrival: 8.581 yd

## Throw-frame nearest-defender geometry by route

Median nearest-defender distance at the throw frame:

| Route | Rows | Median distance (yd) |
|---|---:|---:|
| GO | 14,762 | 1.927 |
| POST | 5,157 | 2.255 |
| SLANT | 4,312 | 2.370 |
| CORNER | 2,845 | 2.421 |
| IN | 5,644 | 2.460 |
| HITCH | 11,825 | 2.643 |
| OUT | 8,034 | 2.685 |
| CROSS | 7,371 | 2.970 |
| WHEEL | 430 | 3.145 |
| FLAT | 11,209 | 5.181 |
| ANGLE | 2,968 | 5.424 |
| SCREEN | 3,170 | 6.099 |

These descriptive route differences are not predictive findings and are not evidence of route quality or coverage responsibility.

## Disposition

**BDB2021_ROUTE_NEAREST_DEFENDER_GEOMETRY_LAB_VALIDATED**

What is now established:

1. exact public route labels are stable at player-play grain across the full 2018 regular-season passing corpus;
2. offense/defense side can be resolved deterministically from possession team + game metadata;
3. snap/throw/arrival nearest-defender geometry can be reconstructed at essentially complete coverage where the relevant event exists;
4. route-specific geometric distributions can be generated without using post-hoc assignment guesses.

What is *not* established:

- true defender responsibility;
- man/zone assignment;
- targeted receiver identity from a semantic target field;
- predictive value;
- production/live deployability.

This lab is suitable for developing route-trajectory, spacing, leverage, congestion and target-candidate methods, provided all responsibility language remains separate from nearest-defender geometry.
