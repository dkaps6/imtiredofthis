# RB Opponent-Defender Injury Readiness V1 — Result

**STATUS: SOURCE READY / NO MODEL CANDIDATE SCORED**

Frozen plan:
`docs/research/RB_OPPONENT_DEFENDER_INJURY_READINESS_V1_PLAN.md`

Authority:
- branch: `research-rb-opponent-defender-injury-readiness-v1`
- run: `37534490584` — **SUCCESS**
- source SHA: `d9b0101499a7e8ef2e320b45d062623112428b17`
- artifact: `11446405827`
- digest: `sha256:01b4124be06a435da3048e931517cff81dbc036d3a23ad00c4cf9d49ed58e97b`

Final disposition:

`RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY`

## Frozen readiness gates

All required source seasons are present:

- injury rows:
  - 2024: 6,215
  - 2025: 6,068
  - 2026: 1,052
- defensive snap rows:
  - 2023: 26,540
  - 2024: 26,615
  - 2025: 26,613
  - 2026: 5,970
- live 2026 defensive injury teams represented: **32**

Strictly-prior defensive snap coverage:

- all defensive injury rows: **97.4439%** (6,690 rows)
- OUT/DOUBTFUL defensive rows: **95.5901%** (1,542 rows)
- OUT/DOUBTFUL FRONT7 rows: **95.4076%** (871 rows)

Identity / chronology:

- stable or roster-mediated identity share among matched rows: **100%**
- same/future snap violations: **0**
- unresolved identity collisions: **0**

Research boundary:

- sportsbook inputs used: **0**
- target-game outcomes read: **0**
- candidate variants constructed: **0**
- candidate variants scored: **0**
- parameters fit: **0**
- production mutations: **0**

## Interpretation

The source question is resolved positively.

Free nflverse/nflreadpy injury, weekly-roster identity, and defensive-snap data can support a leakage-safe opponent-defender availability surface at the frozen readiness standard. This means target-week defensive personnel loss can be represented with materially new pregame information rather than inferred from generic aggregate defense history.

This result does **not** establish predictive value and does not authorize a production change.

The next scientific action is governed by the already-frozen RB-PD2 / mean-information activation boundary. Do not fit or score an injury candidate until that boundary explicitly permits it.

No paid OddsAPI or paid external source is required.
