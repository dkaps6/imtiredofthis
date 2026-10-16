# WR Player Efficiency-Difficulty Trust — Week 6 Pregame Lock V1 — Result

Date: 2026-10-15

Status: **COMPLETE / WEEK6_WR_EFFICIENCY_TRUST_PREGAME_LOCK_FROZEN / RESEARCH ONLY**

Frozen contract:
- `docs/research/WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_FREEZE.md`

Historical parent:
- disposition `WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38826418315`
- artifact `12015303274`
- digest `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`.

Prospective Week-6 authority:
- workflow: `Research WR Player Efficiency Difficulty Trust Week6 Pregame Lock V1`
- run: `38826804491` — **SUCCESS**
- job: `116736044870`
- exact head: `4f7443ed21273040c91dd29593a47654acad61be`
- artifact: `12015596169`
- digest: `sha256:6fc91bee57067219c69f6817950afa87ac07588a391275fe9ffe0c04f58b05dd`

## Frozen Week-6 cohort

- current WR cohort: **152**
- supported: **141**
- unsupported: **11**
- HIGH_EFFICIENCY_TRUST: **35**
- MIDDLE: **70**
- LOW_EFFICIENCY_TRUST: **36**
- minimum prior games: **4**
- reconstructed 2026 W1-W5 WR rows: **717**
- Week-6 efficiency pool MAE: **11.964726 yd-equivalent**

Immutable row digest:
- `sha256:0bf72b516548fed6ba6c48d464674f011ef357e09388695b28ca5dd3ae9cf20f`

## Prospective integrity

The lock reused the already-frozen Week-6 WR opportunity-trust pregame
reconstruction and added only settled W1-W5 receiving-yard labels from the
football-only/no-odds Week-6 Full Slate artifact.

Boundary checks:
- 2026 history source max week: **5**
- Week-6 outcomes read: **0**
- actual-target parity max gap: **0**
- efficiency decomposition max gap:
  **8.01e-12**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- models fit: **0**
- projection/distribution values changed: **0**
- production changed: **false**

216 zero-stat player-weeks absent from the compact player log were explicitly
filled as zero only when the already-frozen target count was also zero.

## Interpretation

This is a prospective **reliability label only** around the unchanged WR-R15
projection.

It is deliberately separate from the already-frozen Week-6 WR opportunity-trust
label.

The lock does **not**:
- combine opportunity and efficiency trust;
- move the receiving-yard mean;
- change distribution width or shape;
- choose an over/under;
- choose a bet/fade;
- use sportsbook information.

Prospective scoring must occur only after Week 6 settles. The frozen labels must
not be altered using target-week outcomes.

TE receives no analogous efficiency-trust lock because the historical TE V1
failed its independent-confirmation clustered gates.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
