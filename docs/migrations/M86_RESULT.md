# Migration 86 — Authoritative Result

## Disposition

`ERROR_FLOOR_AUDIT_COMPLETE_LOW_CHAOS_FRONTIER_IDENTIFIED`

Migration 86 completed the frozen forensic audit of the authoritative M82 full-stack QB trace. It did not fit a predictive model and did not use postgame variables as pregame features.

## Authoritative run

- GitHub Actions workflow: `Migration 86 QB Error-Floor Recoverability Audit`
- Run: `33323402907` (Run #1)
- Artifact: `m86-qb-error-floor-recoverability`
- Artifact ID: `9735538162`
- Artifact SHA256: `1bae4df7949e408e4234bef4bc803c9eb65661c8572a7b5af2907c1af3df1a26`
- frozen M82 rows: `884`
- frozen M82 OOS-ensemble MAE: `56.749517`
- frozen 100+ yard misses: `123`
- sportsbook features used: `False`
- postgame event features used for prediction: `False`
- production actionable: `False`

## Primary finding

Of the 123 catastrophic 100+ yard ensemble misses:

- `85` (`69.11%`) were `HIGH_EVENT_CHAOS` under the preregistered postgame forensic markers;
- `38` (`30.89%`) were `LOW_EVENT_CHAOS`.

The high-event-chaos marker was also common in non-tail games (`60.32%`), versus `69.11%` in tail games. Therefore the marker is **not** a deployable classifier and M86 does not claim that chaos alone explains catastrophic error.

However, high-event-chaos games carried `64.73%` of total absolute passing-yard error across all 884 games. Their descriptive MAE was about `59.70` yards, while games not flagged high-event-chaos had MAE `52.035143`.

This is forensic attribution only; it does not imply that the high-chaos games can be identified or excluded pregame.

## Catastrophic component decomposition

Across all 123 tails:

| Component class | Underprojected | Overprojected | Total |
|---|---:|---:|---:|
| `VOLUME_DOMINANT` | 51 | 6 | **57** |
| `EFFICIENCY_DOMINANT` | 34 | 16 | **50** |
| `MIXED` | 11 | 5 | **16** |
| **Total** | **96** | **27** | **123** |

The catastrophic problem remains strongly asymmetric: `96/123` (`78.05%`) tails are underprojections.

## Low-event-chaos research subset

The 38 catastrophic misses with no frozen high-chaos marker are the highest-value remaining pregame research population:

- `EFFICIENCY_DOMINANT`: `17`
- `VOLUME_DOMINANT`: `16`
- `MIXED`: `5`

Their mean absolute errors were:

- efficiency-dominant: `118.646636`
- volume-dominant: `132.224864`
- mixed: `125.505719`

This is important: after removing games with obvious extreme-event context, the residual catastrophic frontier does **not** collapse to only attempts or only YPA. Volume and efficiency surprises remain almost evenly represented.

## High-event-chaos tails

Among the 85 high-chaos tails:

- volume-dominant: `41`
- efficiency-dominant: `33`
- mixed: `11`

The volume-dominant high-chaos group had a mean actual-minus-predicted attempt residual of `+14.858972` attempts. The efficiency-dominant high-chaos group had mean YPA residual `+2.572363` yards/attempt. These are enormous target-game deviations and explain why many catastrophic underprojections are difficult to solve with ordinary historical averages.

## Event prevalence caution

Tail games had somewhat larger longest completions on average (`41.72` yards vs `36.88` for non-tails), but sacks, interceptions, scrambles and fourth-down attempts were not dramatically separated in simple unconditional averages.

Therefore M86 does **not** support a simplistic conclusion that one observable postgame event category explains all misses. The event-chaos flag is a forensic partition, not a causal or predictive model.

## Error-floor interpretation

Two diagnostics now bracket the problem:

1. Current deployable clean full-stack OOS ensemble: `56.749517` MAE.
2. Existing-model hindsight oracle: `41.103131` MAE, nondeployable.

Games outside the broad high-chaos forensic bucket had `52.035143` MAE, showing that substantial error remains even in comparatively ordinary-looking games. Therefore the project should **not** conclude that the remaining ~15.6 yards of oracle headroom is simply irreducible randomness.

At the same time, 69% of catastrophic misses occurring in high-chaos contexts and 64.7% of total absolute error being carried by those games are strong evidence that target-game event variance materially limits point prediction.

## Strategic conclusion

The next research target should be **the 38 low-event-chaos catastrophic misses**, not all 884 games and not all 123 tails at once.

The key question becomes:

> What pregame information distinguishes ordinary-looking games in which the full-stack ensemble still misses by 100+ yards?

Because the 38-game subset is nearly balanced between volume and efficiency dominance, future work should treat these as two distinct sub-frontiers rather than forcing one correction across both:

- low-chaos volume surprise (`16` games);
- low-chaos efficiency surprise (`17` games);
- low-chaos mixed (`5` games).

A next migration may audit the pregame characteristics of those low-chaos groups **without fitting a selector on the same outcomes**. Any predictive candidate must be preregistered and validated on an untouched temporal sample before promotion.

## Anti-loop consequence

Do not:

- use the M86 postgame chaos markers as predictive features;
- tune the chaos thresholds after seeing these results;
- discard high-chaos games from normal model evaluation;
- claim `52.035143` is a deployable MAE;
- reopen M83/M84/M85 same-information mechanisms;
- train an unrestricted classifier directly on these 38 labels and report in-sample performance.

M86 narrows the scientific target; it does not provide a new production model.
