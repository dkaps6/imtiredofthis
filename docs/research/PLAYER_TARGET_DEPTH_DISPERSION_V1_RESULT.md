# Player Target Depth Dispersion V1 — Result

**STATUS: COMPLETE / PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED**

Frozen plan:
`docs/research/PLAYER_TARGET_DEPTH_DISPERSION_V1_PLAN.md`

Authority:
- branch `research-player-target-depth-dispersion-v1`
- run `37655282486` — **SUCCESS**
- source SHA `7607eb068bb4dad92ec5b0d8497ba5069349ae4d`
- artifact `11497703984`
- digest `sha256:39110b5d8d344b5beb51f5eab02ba931ffddae41932f8d7dcf5ca5da66824f55`

Final disposition:
`PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED`

## Frozen player-state feature

For each exact promoted-authority WR/TE player-game:
- use only strictly-prior target events;
- latest up to 8 completed receiver-games;
- require at least 4 prior receiver-games;
- require at least 10 finite air-yard target events;
- calculate population SD of `air_yards`.

Target is the absolute receiving-yard **efficiency component** error from the exact promoted WR/TE authority.

This is an uncertainty/difficulty signal, not a signed YPT-mean signal.

## Support and integrity

- 6,497 scoreable player-games
- 350 individual receivers
- WR: 219
- TE: 131
- identity mapping coverage WR 100%, TE 100%
- same/future feature violations 0
- fitted models 0
- sportsbook inputs 0
- 2026 outcomes 0
- production changes 0

## WR replication

2023:
- 1,816 rows / 167 WRs
- rho = **0.07713**

2024:
- 1,910 rows / 186 WRs
- rho = **0.09278**

Pooled WR:
- 3,726 rows / 219 WRs
- rho = **0.08499**
- player-cluster bootstrap P(rho > 0) = **0.9990**
- 95% CI = **[0.0333, 0.1349]**

Typical WR target-depth SD:
- P10 6.93
- median 10.03
- P90 13.68 air yards

## TE replication

2024:
- 942 rows / 93 TEs
- rho = **0.04696**

2025:
- 985 rows / 105 TEs
- rho = **0.04930**

Pooled TE:
- 2,771 rows / 131 TEs
- rho = **0.05414**
- player-cluster bootstrap P(rho > 0) = **0.9782**
- 95% CI = **[0.0017, 0.1046]**

Typical TE target-depth SD:
- P10 4.68
- median 6.74
- P90 9.00 air yards

## Combined

- 6,497 rows / 350 receivers
- rho = **0.18412**
- cluster-bootstrap P(rho > 0) = **1.000**
- 95% CI = **[0.1436, 0.2230]**

All frozen confirmation gates passed.

## Interpretation

The previously confirmed player-specific WR/TE efficiency-difficulty signal has a genuine football-state correlate:

> Receivers whose own target depths are more dispersed are systematically harder to translate from target opportunity into receiving-yard outcomes.

Because signed efficiency bias did not persist in the mechanism audits, this does **not** authorize a YPT mean shift.

The correct downstream lane is a separately frozen **mean-neutral player-specific distribution/uncertainty shadow** that preserves:
- receiving-yard means;
- target entitlement;
- team target/pass volume;
- M38 / WR-R15 / TE-R5P;
- QB science;
- sportsbook separation.

Protected closures remain closed:
- WR R7 signed-YPR trait work;
- M72;
- M75;
- WR-R3 residual-width calibration;
- TE Width V2;
- QB-receiver pair YPT.

No production change is authorized.
