# Receiver Room Targets-Per-Play V1 — 2024-2025 Confirmation Result

Date: 2026-09-25

Disposition: **RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED**

This is a scientific failure under the frozen confirmation gates.

## Authority

- branch: `research-receiver-room-targets-per-play-v1-confirm-2024-2025`
- authoritative corrected run: `36174077739`
- job: `108200327879`
- tested head: `c1b48b90364e2d6f82b66271065cc7b359d725ca`
- artifact: `10881825507`
- digest: `sha256:3956a8ed9cc2191714787ecab235831b5f9ce39c59f3e08ba260ee2918f2de9c`
- parameters fit: `0`
- candidate variants scored: `1`
- sportsbook inputs: `0`
- target-game outcomes upstream: `0`
- WR-R15 2025 applications: `0`
- WR same/future participation violations: `0`
- specialist conservation max gap: `1.11e-16`

## Mechanical lineage

Initial run `36173485673` stopped before science with:

`KeyError: 'prior_history_plays'`

Cause:
the confirmation script accidentally imported the prior targets-per-dropback
history/rate helpers.

Bounded repair commit:
`c1b48b90364e2d6f82b66271065cc7b359d725ca`

The repair restored the already-frozen targets-per-play helper implementation.
No formula, history horizon, cohort, specialist ordering, gate or threshold
changed.

Only corrected run `36174077739` is scientifically authoritative.

## 2024

- WR MAE: `4.561046 -> 4.724467` — worse
- WR p90: `9.687580 -> 9.763043`
- TE MAE: `2.982040 -> 2.881030`
- TE p90: `6.106685 -> 6.300518`
- RB_FB MAE: `2.389736 -> 2.169780`
- RB_FB p90: `4.835043 -> 4.499055`
- macro MAE: `3.310940 -> 3.258426`
- macro abs bias: `0.536497 -> 0.506666`

## 2025

- WR MAE: `4.510085 -> 4.440251`
- WR p90: `9.165220 -> 9.242950`
- TE MAE: `2.768463 -> 2.728961`
- TE p90: `5.527355 -> 5.777204`
- RB_FB MAE: `2.302348 -> 2.176637`
- RB_FB p90: `4.643633 -> 4.426546`
- macro MAE: `3.193632 -> 3.115283`
- macro abs bias: `0.506332 -> 0.297785`

## Pooled 2024-2025

- WR MAE: `4.535565 -> 4.582359` — worse
- TE MAE: `2.875251 -> 2.804996`
- RB_FB MAE: `2.346042 -> 2.173209`
- macro MAE: `3.252286 -> 3.186854`
- macro p90: `6.646801 -> 6.651926` — worse
- macro abs bias: `0.404433 -> 0.325998`
- all-room candidate closer rate: `52.2672%`

Summed-room:
- MAE: `6.127841 -> 5.983952`
- abs bias: `1.213299 -> 0.977995`
- p90: `12.704084 -> 12.049116`

## Frozen gate failures

Failed:
- WR room MAE improves in 2024
- pooled WR room MAE improves
- pooled macro p90 nonworse

All other frozen gates passed.

## Interpretation

The 2022-2023 discovery signal did **not** confirm unchanged in 2024-2025.

The exact formula still improved:
- macro room MAE in 2024;
- macro room MAE in 2025;
- TE pooled MAE;
- RB_FB pooled MAE;
- summed-room MAE/bias/p90;
- pooled absolute bias.

But the WR room was unstable:
- 2024 WR worsened materially;
- 2025 WR improved;
- pooled WR therefore worsened.

The tiny pooled macro p90 regression independently failed the frozen protection
contract.

Therefore the exact cumulative targets-per-play formulation is closed.

## No rescue

Do not:
- exclude 2024;
- make WR use a different rule;
- add recency/shrinkage/window search;
- blend with fixed57;
- add a WR multiplier or bias offset;
- exempt WR1/Q4;
- route by season;
- alter specialist ordering;
- fit 2026;
- use sportsbook information.

## Next legitimate question

Use the failure diagnostically, not as a rescue target.

The next read-only question should explain **why WR room opportunity is
non-stationary across eras while TE/RB_FB remain directionally stable**.

A valid diagnostic may compare 2022/2023/2024/2025 WR room-rate errors against
strictly pregame structural change variables such as:
- roster continuity / top-target turnover;
- QB change;
- offensive coordinator / play-caller change if a reliable source already
  exists in the repo;
- prior-season-to-current-season WR room share drift;
- speed at which within-season room rates adapt.

Do not score candidate variants during that diagnostic.

No player/full-stack integration is authorized from this failed confirmation.
