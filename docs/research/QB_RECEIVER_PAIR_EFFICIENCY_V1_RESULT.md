# QB-Receiver Pair Efficiency V1 — Result

**STATUS: COMPLETE / SIGNAL CLOSED**

Frozen contract:
`docs/research/QB_RECEIVER_PAIR_EFFICIENCY_V1_CONTRACT.md`

Authority:
- branch: `research-qb-receiver-pair-efficiency-v1`
- optimized certified run: `37628838585` — SUCCESS
- source SHA: `e7fa5b0064462aa1c3e4fc783c24d5d0ff95f689`
- artifact: `11485911410`
- digest: `sha256:f260bd78a5f5d4d81b4688bc020ab59d7a52ff88e274d8d4b5ea2e7f71202682`
- pre-optimization equivalent run: `37628251417` — SUCCESS
- equivalent result artifact: `11485197160`
- equivalent digest: `sha256:1f7652676208563010b3531890c3b9edf806d563d071b8bdb2efce36ceac615f`

Both runs produced the same scientific result.

Final disposition:

`QB_RECEIVER_PAIR_EFFICIENCY_SIGNAL_CLOSED`

## Design

Control:
- same receiver's strictly-prior YPT over latest eight receiver games.

Challenger:
- same receiver's strictly-prior YPT with the exact pregame passer proxy inside the same eight-game horizon.

Support:
- >=10 prior receiver targets;
- >=5 prior pair targets.

Target:
- target-game YPT.

No fitted coefficient.
No shrinkage.
No sportsbook input.
No 2026 outcome.
No production change.

## Season results

### 2023
Rows: 2,014 / 208 games

Control YPT MAE:
- `4.517338`

Pair YPT MAE:
- `4.620825` — worse by `0.103486`

Control RMSE:
- `6.265697`

Pair RMSE:
- `6.395197`

Actual-target-held-fixed yard MAE:
- `18.189188 -> 18.726285` worse.

### 2024
Rows: 2,062 / 208 games

Control MAE:
- `4.351838`

Pair MAE:
- `4.495346` — worse by `0.143507`

Control RMSE:
- `6.162470`

Pair RMSE:
- `6.330586`

Actual-target-held-fixed yard MAE:
- `17.222906 -> 17.917031` worse.

### 2025
Rows: 1,999 / 208 games

Control MAE:
- `4.437507`

Pair MAE:
- `4.536059` — worse by `0.098552`

Control RMSE:
- `6.281649`

Pair RMSE:
- `6.386095`

Actual-target-held-fixed yard MAE:
- `16.578115 -> 17.139520` worse.

## Pooled WR + TE

Rows:
- 6,075

Players:
- 364

Games:
- 624

Control:
- YPT MAE `4.434895`
- RMSE `6.236134`
- actual-target-held-fixed yard MAE `17.331080`
- 20+ YPT misses `71`

Pair:
- YPT MAE `4.550342`
- RMSE `6.370336`
- actual-target-held-fixed yard MAE `17.929474`
- 20+ YPT misses `74`

Game-cluster bootstrap:
- 10,000 reps
- P(pair improves MAE): `0.0`
- 95% CI of control-AE minus pair-AE:
  `[-0.146272, -0.084608]`

The interval is entirely negative, favoring the receiver-own-history control.

## Position detail

WR:
- control MAE `4.608256`
- pair MAE `4.732118` — worse
- held-fixed yard MAE `19.305707 -> 20.016140`

TE:
- control MAE `4.074193`
- pair MAE `4.172132` — worse
- held-fixed yard MAE `13.222613 -> 13.587898`

Every scientific gate except support failed.

## Interpretation

Exact QB-receiver pair state is a valid and useful data object, but the receiver's own recent efficiency is more predictive than raw pair-specific YPT.

This is a useful player-level finding:

> Individualization does not mean splitting a player's history into ever-smaller identity cells. Some identity-conditioned histories become noisier and less predictive than the player's own broader history.

The source remains valid for future genuinely distinct hypotheses, especially transition/continuity questions, but raw pair YPT replacement is closed.

Do not rescue with:
- alternate windows;
- pair-target thresholds;
- WR/TE carveouts;
- actual starter identity;
- pair catch/YAC/air blends;
- fitted shrinkage.

No production change is authorized.
