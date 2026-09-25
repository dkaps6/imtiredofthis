# Receiver Room Targetable-Rate V1 — Temporal Screen Result

Date: 2026-09-25

Disposition: **RECEIVER_ROOM_TARGETABLE_RATE_V1_FAILED_CLOSED**

This is a scientific failure under the frozen gate table, not a mechanical
failure.

## Authority

- branch: `research-receiver-room-targetable-rate-v1`
- run: `36171462750`
- job: `108191753797`
- tested head: `8a81a74ebd660d467711f048413adce461301252`
- artifact: `10880311081`
- digest: `sha256:d6db6f5bcbd8a599a362c9e4c05aec0cd0c74b9f953e2133146fef28a616bb22`
- parameters fit: 0
- candidate variants: 1
- sportsbook inputs: 0
- target-game outcomes upstream: 0

## Exact candidate

For room g in WR / TE / RB_FB:

`R_g = sum(strict-prior room targets) / sum(strict-prior team dropbacks)`

`candidate room targets = projected dropbacks * R_g`

2022-2023 baseline used only the historically legitimate pre-specialist/M38
entitlement stack. No TE-R5P / WR-R15 backcast.

## 2022

- WR MAE: `4.761250 -> 4.564125`
- WR p90: `9.519272 -> 9.412887`
- TE MAE: `2.828880 -> 2.600674`
- TE p90: `5.479486 -> 5.106522`
- RB_FB MAE: `2.730621 -> 2.548742`
- RB_FB p90: `5.589022 -> 5.101199`
- macro MAE: `3.440250 -> 3.237847`

## 2023

- WR MAE: `4.538430 -> 4.474989`
- WR p90: `9.506628 -> 9.628632`
- TE MAE: `2.800682 -> 2.659486`
- TE p90: `5.553391 -> 5.497953`
- RB_FB MAE: `2.368523 -> 2.264151`
- RB_FB p90: `4.855590 -> 4.724783`
- macro MAE: `3.235879 -> 3.132876`

## Pooled

- WR MAE: `4.649635 -> 4.519475`
- TE MAE: `2.814755 -> 2.630134`
- RB_FB MAE: `2.549238 -> 2.406185`
- macro MAE: `3.337876 -> 3.185265`
- macro p90: `6.736290 -> 6.559403`
- candidate closer rate: `52.1793%`

All room MAE gates, season gates, tail gates and source/integrity gates passed.

## Decisive frozen failure

Pooled macro absolute bias worsened:

- baseline macro abs bias: `0.323767`
- candidate macro abs bias: `0.800346`

Candidate pooled signed room biases:
- WR: `-1.466160` targets
- TE: `-0.555442`
- RB_FB: `-0.379436`

Therefore the candidate is failed closed despite broad MAE/tail improvement.

## Upstream attribution discovered after disposition

The bias is not caused by an omitted residual/OTHER receiver class.

On the exact 1,086 aligned 2022-2023 team-games:
- actual targets outside WR/TE/RB_FB average only ~`0.0295` per game;
- candidate team target pool minus summed candidate room pools averages only
  ~`0.0322` targets/game;
- summed room-candidate bias is `-2.4010` targets/game;
- the already-qualified team targetable candidate bias is `-2.3983`
  targets/game.

So the room candidates faithfully inherit the team candidate's negative bias.

The upstream fixed 57% projected-dropback anchor is materially low versus actual
dropbacks:
- 2022 signed bias: approximately `-2.8504` dropbacks/game;
- 2023 signed bias: approximately `-3.4041` dropbacks/game.

This explains how a correct targetability/room-rate correction can improve MAE
and tails while still underpredicting volume after multiplying an already-low
dropback forecast.

## Interpretation

This result closes the exact two-stage formulation:

`fixed 57% projected dropbacks * cumulative room targets/dropback`

It does not invalidate:
- the four-season team targetable-dropback signal;
- the room-level evidence that direct historical room opportunity materially
  improves WR/TE/RB_FB MAE;
- the compensation-audit finding that uniform player thinning is wrong.

The next genuinely distinct question is whether receiver opportunity should be
forecast directly per offensive play, avoiding multiplication by the biased
fixed-dropback anchor.

No V1 rescue, bias offset, blend, room multiplier, cap/floor, or gate relaxation
is authorized.
