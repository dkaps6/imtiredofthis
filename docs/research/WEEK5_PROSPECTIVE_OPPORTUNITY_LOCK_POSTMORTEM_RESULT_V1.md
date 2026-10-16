# WEEK 5 PROSPECTIVE PLAYER-OPPORTUNITY LOCK POSTMORTEM V1 — RESULT

Date: 2026-10-13

Status: **LOCK #1 DIAGNOSTIC COMPLETE. NO FINAL PROMOTION DISPOSITIONS AUTHORIZED.**

Canonical authority:
- run `38396065511` SUCCESS
- exact head `3f1097d2303eab5dce7813c03c875002d1838794`
- artifact `11812261490`
- digest `sha256:56a589505cb17cdf71b90a13b2817ca7fd776a5f9a43e7a757a655128075f071`

Week 5 is prospective lock number one.
Production changed: false.

## RB carry/snap allocation

Frozen shadow:
- 50% recent carry share
- 50% recent snap fraction

Scoreable:
- 83 players
- 25 teams

Control recent-carry-share:
- MAE **0.11947**
- RMSE **0.16517**
- top-back accuracy **84%**
- mean room rank rho **0.5702**

50/50 carry/snap shadow:
- MAE **0.14013**
- RMSE **0.17505**
- top-back accuracy **84%**
- mean room rank rho **0.5091**

Mean player absolute-error improvement:
**-0.02065 share points** — shadow was worse.

Team comparison:
- control better: 16 teams
- shadow better: 9 teams
- ties: 0

Examples where snap blending helped materially:
- Bucky Irving
- Omarion Hampton
- Chase Brown / Samaje Perine
- Alvin Kamara
- Detroit Gibbs/Vaki room

Examples where it hurt materially:
- J.K. Dobbins
- Emanuel Wilson
- James Cook
- Christian McCaffrey
- Javonte Williams
- several FB/H-back players with snaps but little/no realized carry role

Interpretation:
raw snap participation is not interchangeable with rushing opportunity. The
50/50 formula appears capable of allocating carry mass to blocking/receiving
participants who do not receive carries.

Status:
`WEEK5_DIAGNOSTIC_ONLY__NO_FINAL_V1_DISPOSITION`

Do not retune the frozen shadow from this one week. Preserve the preregistered
multi-week gate unless a separate stop rule authorizes closure.

## RB receiving-room share

Scoreable:
- 82 players
- 25 teams

Candidate absolute realized-share diagnostics:
- MAE **0.18879**
- RMSE **0.23488**
- room-share bias approximately zero by construction

But the immutable Week-5 lock did not persist the required current canonical
baseline receiving-room share.

Therefore:
- baseline comparison unavailable;
- no candidate-vs-baseline Week-5 improvement claim is legal.

Evidence defect:
`LOCK_CONTRACT_REQUIRED_BASELINE_BUT_LOCK_CONSTRUCTOR_DID_NOT_PERSIST_IT`

Status:
`WEEK5_CANDIDATE_ABSOLUTE_DIAGNOSTIC_ONLY`

Required forward repair:
future locks must persist both control and candidate receiving-room shares at
lock time. Do not reconstruct a favorable Week-5 baseline post hoc.

## WR/TE target-share trajectory

Locked / scoreable:
- 277 players
- 60 rooms
- identity-unscoreable rows: 0

All WR+TE:
- baseline MAE **0.058365**
- shadow MAE **0.058051**
- improvement **+0.000313**
- baseline RMSE 0.073558
- shadow RMSE 0.073295
- room rank rho 0.5130 -> **0.5243**
- top-target accuracy 70.0% -> **68.3%**

WR only:
- 167 players
- baseline MAE **0.064765**
- shadow MAE **0.064860**
- change **-0.000095** — essentially flat/slightly worse

TE only:
- 110 players
- baseline MAE **0.048647**
- shadow MAE **0.047714**
- improvement **+0.000934**

This is small Week-5 evidence in favor of TE trajectory and essentially neutral
for WR. It is nowhere near sufficient for final promotion.

Status:
`WEEK5_DIAGNOSTIC_ONLY__NO_FINAL_V1_DISPOSITION`

## Conclusion

The Week-5 prospective evidence does NOT support a broad claim that every
opportunity shadow improved.

- RB 50/50 carry/snap: **negative first lock**
- RB receiving: **cannot legally compare because of frozen-evidence defect**
- WR trajectory: **flat/slightly negative**
- TE trajectory: **small positive first lock**

Keep the original prospective accumulation rules. Repair evidence persistence
forward, not Week-5 outcomes backward.
