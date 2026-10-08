# Football Matchup Transmission V1 — Integration Candidate Result

Status: **COMPLETE — ALL THREE FROZEN V1 INTEGRATION CANDIDATES FAILED CLOSED — NO PRODUCTION CHANGE**

Branch: `research-football-matchup-transmission-v1`

Canonical run:
- run `37519246784`
- artifact `11440195795`
- artifact digest `sha256:4e2fac59ad262de477c3cd9acbaeae623db619aba2b54e5a88606177a8f01c38`
- head `0c7c1b81907fc2b213cd57dcc2c45f62f2c73c06`

Frozen candidate contract:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATES_FREEZE.md`

## Boundary

- evaluation seasons: 2024, 2025
- Weeks 2-18
- 2026 outcomes used: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- target-game usage: false
- candidate coefficients fit: 0
- threshold searches: 0
- production change authorized: false

## Result

All three independently frozen candidates received:

`HISTORICAL_INTEGRATION_FAIL_CLOSED`

### RB — opponent pass-rate-faced

Candidate:
`FMT-INT-RB-DEF-PASS-RATE-FACED-V1`

Primary cohort:
RB rushing yards.

Primary RB rushing MAE moved only trivially:
- 2024: 21.4109 -> 21.4095
- 2025: 20.6801 -> 20.6637

The candidate did not achieve supported primary improvement and produced
statistically supported collateral harm in:
- RB rush+receiving yards, 2024 and 2025;
- RB receiving yards, 2024 and 2025.

Disposition:
`HISTORICAL_INTEGRATION_FAIL_CLOSED`

Do not rescue with a new coefficient or role carveout.

### WR — offensive true PROE

Candidate:
`FMT-INT-WR-TRUE-PROE-V1`

Primary cohort:
WR receiving yards.

Primary WR rec-yards MAE:
- 2024: 22.4612 -> 22.4548
- 2025: 21.6298 -> 21.5711

Direction was slightly favorable, but neither season cleared the frozen
game-cluster + player-cluster improvement gate, and primary RMSE was not
non-worse in both seasons.

Disposition:
`HISTORICAL_INTEGRATION_FAIL_CLOSED`

The architecture defect remains real—generic skill simulation still shadows
offensive pass tendency with a 0.57 rules pass rate—but this simple replacement
is not historically qualified.

### TE — opponent pass-success allowed

Candidate:
`FMT-INT-TE-DEF-PASS-SUCCESS-V1`

Primary cohort:
TE receiving yards.

Primary TE rec-yards MAE:
- 2024: 16.4441 -> 16.4225
- 2025: 15.5712 -> 15.5258

The 2025 improvement was supported; 2024 was not. The frozen requirement was
replication in both seasons.

Disposition:
`HISTORICAL_INTEGRATION_FAIL_CLOSED`

Do not promote a 2025-only matchup multiplier.

## Interpretation

This result does **not** mean matchup context is irrelevant.

Phase A established that production collects richer football context than the
generic RB/WR/TE stack consumes. Phase B/C then found several residual matchup
signals that replicate descriptively.

What failed was the exact simple V1 transmission mechanisms.

Therefore:
- do not add arbitrary defense-vs-position multipliers;
- do not globally boost players against weak defenses;
- do not retune these three candidates after outcomes;
- preserve the architectural finding that player-specific matchup transmission
  is incomplete;
- future work must localize the individual player mechanism rather than
  repackage generic team/position matchup stats.

This is directly consistent with the later 2026 W1-4 all-player replay:
individual player workload/share separation is a larger unresolved source than
generic position-level matchup boosting.
