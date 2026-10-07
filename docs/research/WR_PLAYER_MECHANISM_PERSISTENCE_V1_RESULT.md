# WR Player Mechanism Persistence V1 — Result

**STATUS: COMPLETE / WR_PLAYER_PERSISTENCE_MIXED_MECHANISM**

Frozen plan:
`docs/research/WR_PLAYER_MECHANISM_PERSISTENCE_V1_PLAN.md`

Authority:
- branch `research-wr-player-mechanism-persistence-v1`
- run `37634044900` — SUCCESS
- source SHA `0c9db93a8edd381d70ddc72f5a22d971c88c73d4`
- artifact `11487359185`
- digest `sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3`

Parent promoted WR authority:
- run `34238301577`
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- exact WR1 M38 + WR2+ WR-R15 candidate rows: 4,193

Final disposition:

`WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`

## Scoreable panel

Strictly-prior same-player history:
- latest up to 8 prior promoted-authority WR games
- minimum 4 prior
- 3,224 scoreable rows
- 211 distinct WRs
- 2023: 1,357 rows / 160 players
- 2024: 1,867 rows / 183 players
- support PASS
- same/future violations 0

Exact accounting decomposition max gap:
- `2.842170943040401e-14`

## Opportunity mechanism

### O1 signed opportunity persistence — PASS

2023:
- Spearman `0.175841`

2024:
- `0.164967`

Pooled:
- `0.169932`

Player-cluster bootstrap:
- P(positive) = `1.000`
- 95% CI `[0.1269, 0.2055]`

### O2 opportunity difficulty — PASS

2023:
- `0.334931`

2024:
- `0.304749`

Pooled:
- `0.318305`

Bootstrap:
- P(positive) = `1.000`
- 95% CI `[0.2672, 0.3658]`

Interpretation:
Even after M38/WR-R15 opportunity science, individual WRs retain stable same-player target/opportunity miss structure.

## Efficiency mechanism

### E1 signed efficiency persistence — FAIL

2023:
- `-0.057648`

2024:
- `-0.045106`

Pooled:
- `-0.051067`

Bootstrap:
- P(positive) = `0.0014`
- 95% CI `[-0.0828, -0.0188]`

There is no stable same-direction efficiency bias suitable for a simple YPT correction.

### E2 efficiency difficulty — PASS

2023:
- `0.325364`

2024:
- `0.250127`

Pooled:
- `0.283308`

Bootstrap:
- P(positive) = `1.000`
- 95% CI `[0.2341, 0.3257]`

Interpretation:
Some individual WRs are persistently harder to translate from targets into receiving yards, even though the direction of the efficiency miss is not stable.

## Error mass

Pooled:
- mean absolute total error: `23.2455` yards
- mean absolute opportunity component: `17.1408`
- mean absolute efficiency component: `18.0871`
- normalized opportunity mass: `48.48%`
- normalized efficiency mass: `51.52%`
- component signs oppose in `50.68%` of rows

The WR receiving-yard problem, like TE, is approximately half opportunity and half efficiency in absolute component mass.

## Role diagnostics

These were predeclared diagnostics only, not rescue gates.

WR1:
- O1 `0.1417`
- O2 `0.1260`
- E1 `-0.0056`
- E2 `0.0996`

WR2+:
- O1 `0.1907`
- O2 `0.3014`
- E1 `-0.0784`
- E2 `0.2814`

The player-level persistence is visibly stronger among WR2+ where WR-R15 is actively reallocating the room, but the pooled scientific disposition does not depend on this subgroup.

## Cross-position interpretation

WR and TE now independently show the same broad pattern on their promoted individualized-entitlement architectures:

1. individual opportunity error persists in direction and difficulty;
2. individual efficiency difficulty persists;
3. signed efficiency error does not persist;
4. opportunity and efficiency contribute roughly equally to total receiving-yard error.

This is evidence that the remaining pass-catcher problem is not well described as a single WR-level or TE-level positional correction.

The next valid research object is a **player-specific pregame opportunity state** that explains why certain players earn more or fewer targets than their participation-based entitlement expects.

Do not feed residual error directly back into projections.

No production change is authorized.
No sportsbook input.
No 2026 outcomes.
Models fit: 0.
