# WR-ND3 — Dynamic Target Entitlement Result

## Canonical run

- Branch: `research-wr-nd3-dynamic-target-entitlement`
- Workflow: `WR-ND3 Dynamic Target Entitlement`
- Run: `34046039215`
- Job: `101521139008`
- Tested SHA: `0ce6ea4f5dfa7a2b6b337d467a04d211cac20a1d`
- Artifact: `9993252191`
- Artifact digest: `sha256:0731475c4f3ed70f5927ee988b5e9fe50a0deb691f7dca3d2a986c500a5647fa`
- Conclusion: `success`

## Integrity

The run reconstructed exact M38 before evaluating WR-ND3:

- M38 rec-yards rows: `4647`
- M38 rec-yards MAE: `17.099904733366`
- RMSE: `25.196099510685915`
- bias: `-5.238640833494836`
- correlation: `0.5679458508349821`

The WR entitlement casebook also reproduced the known post-M38 target-opportunity error:

- evaluation rows: `2130`
- target MAE: `2.076010432545868`
- factorization anomalies excluded from evaluation: `1`
- anomaly: Isaiah Bond, CLE, 2025 W16 — `0` recorded targets with `+21` receiving yards
- historical logs remained untouched for prior construction
- sportsbook inputs used: `false`
- model fitting used: `false`
- production changed: `false`

## Source audit discovered inside the canonical run

The historical pipeline does **not** currently provide the dynamic participation state needed for a stronger entitlement model:

- historical routes non-null rate: `0.0`
- historical route-rate non-null rate: `0.0`
- target-week pregame role nonempty rate: `0.0`

Therefore routes, route-rate trajectory, and historical depth-role movement are source-blocked in the current canonical historical information set. They may not be silently imputed from outcomes.

## Frozen scientific disposition

`NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`

No candidate passed the predeclared gate. Thresholds remain frozen and are not relaxed after seeing the result.

## Candidate results

### `RECENCY_ACCEL_2V8`

Definition: last-2 prior same-team target-share mean minus last-8 prior same-team target-share mean.

- coverage: `0.9464788732394366`
- Spearman vs allocation residual: `0.06013468907887743`
- high-vs-low allocation-residual gap: `0.024742037067688773`
- frozen 10+ target / under-by-3 tail enrichment: `1.3333333333333335x`
- W2-18 gap: `0.02586374549266885`
- W13-18 gap: `0.027108718745584036`
- WR1 gap: `0.023137117853676024`
- WR2 gap: `0.012019846755785685`
- WR3 gap: `0.00025483877706581476`
- positive WR1/WR2/WR3 count: `3`
- frozen quartiles: low `-0.026783955091436773`, high `0.021256443556178263`
- gate: `FAIL`

Interpretation: this was the strongest and directionally coherent candidate. It passed coverage, tail enrichment, late-season direction, and all three WR-role direction checks, but it failed the frozen Spearman threshold (`0.0601 < 0.08`) and narrowly missed the frozen allocation-gap threshold (`0.024742 < 0.025`). It is **promising but insufficient**, not actionable.

Do not lower the thresholds. Do not test nearby 1v4 / 3v8 / alternative windows simply because 2v8 nearly passed; that would be post-result window hunting on the same information family.

### `HIGHER_WR_ABSENT_COUNT`

- coverage: `0.9131455399061033`
- Spearman: `-0.01897140734925638`
- allocation gap: `-0.0033543041407648492`
- tail enrichment: `0.27646166966739316x`
- W2-18 gap: `-0.003211399025127871`
- W13-18 gap: `0.007619654598198986`
- WR1 gap: `-0.036698658966384516`
- WR2 gap: `-0.022336560530022684`
- WR3 gap: `-0.025603781584718637`
- gate: `FAIL`

Disposition: rejected. Do not repackage simple higher-usage WR absence counts as a new target-entitlement feature.

### `VACATED_SHARE_ABOVE_PLAYER`

- coverage: `0.9131455399061033`
- Spearman: `-0.0018748497547069452`
- allocation gap: `0.00101723341991074`
- tail enrichment: `0.18036536737658806x`
- W2-18 gap: `0.0013033233731988302`
- W13-18 gap: `0.00880039870003724`
- WR1 gap: `-0.032084471568981685`
- WR2 gap: `-0.03493293647921607`
- WR3 gap: `-0.012292956035384859`
- gate: `FAIL`

Disposition: rejected. Simple vacated-target mass does not explain the remaining within-WR entitlement residual and should not be layered on top of the existing alpha-vacancy rule.

## Durable interpretation

The remaining post-M38 opportunity problem is **not solved by simple teammate absence/vacated-target heuristics**.

Recent target-share acceleration contains some real-looking directional information, particularly in the frozen high-entitlement miss tail, but the current aggregate target-history representation is not strong enough to clear the frozen action gate.

The most defensible next step is therefore **new information acquisition**, not threshold/window tuning: audit whether timestamp-safe historical offensive participation, snaps/routes, and/or real pregame depth-role state can be recovered at sufficient coverage to support a separate future entitlement test.

## QB / explosive lane remains separate

The user hypothesis that validated WR ceiling information may help explain QB explosive passing-yard games remains preserved for later testing. M72 already rejected aggregate explosive-weapon × defense proxies, so any future QB bridge must use materially new, validated player-level pregame WR ceiling information and test QB upper-tail misses specifically. Realized target-week WR explosions may never be used upstream.
