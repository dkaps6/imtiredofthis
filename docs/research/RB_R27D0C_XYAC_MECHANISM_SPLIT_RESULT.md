# RB R27D0C — xYAC Mechanism Split Result

Status: `R27D0C_XYAC_MECHANISM_SPLIT_COMPLETE`

Diagnostic-only evidence. No model fit, no candidate, no sportsbook input, no production/R26/R22 change.

## Canonical authority

- Protected production authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Parent R27D0B source-extension result commit: `4e19b549dad2141ba3d55b77c563947342426461`
- Frozen plan commit: `21db47af29f121e27c137c6e21773dd557094395`
- Frozen plan blob: `6a01722bd0291805a44d6be37dec1e228c3adbd0`
- Original implementation lock: `99f3347abc6af3ac300b4a78ac1237454678a6e1`
- Preserved Run1 mechanical failure: run `34432353557`, job `102730315801`
- Run1 failure: actual YAC used all completed catches while expected YAC/YACOE used only xYAC-observed completed catches, creating a `0.0509294626941692` yard/reception algebra gap.
- Frozen repair record: `docs/research/RB_R27D0C_RUN1_XYAC_OBSERVATION_SET_MECHANICAL_REPAIR.md`
- Repaired script blob: `8428aa5cfe675efcf3c366a48e80d705ffdf5aea`
- Repaired workflow blob: `23fcc7018ab560782556c7fd316308ec90154bbe`
- Repair lock / valid head: `6ee566b67abd87f0f33b37111aeb83011419b43d`
- First valid run: `34432497854`
- Job: `102730752292`
- Artifact: `10134994377`
- Artifact name: `rb-r27d0c-xyac-mechanism-split`
- Artifact digest: `sha256:2c3d6043dc25298b8806e82abe7e4a97e9c2c066408fc38e215eae0bdbc1c882`
- Exact R27B V2 parent artifact: `10134023092`
- Exact R27B V2 parent digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Integrity

PASS.

- 8,429 preserved parent rows.
- 2023 vacancy-RB1 xYAC reception coverage: `1.0000`.
- non-2023 vacancy-RB1 xYAC reception coverage: `0.9944518717`.
- same-observation decomposition gap: `1.7763568394002505e-15`.
- no model fit.
- no candidate projection.
- no sportsbook.
- production/R26/R22 unchanged.

## Primary finding — 2023 compression is execution below xYAC expectation, not low expected-YAC target context

Vacancy RB1 incumbents:

### 2023
- actual YAC/reception on xYAC-observed catches: `7.3395100069`
- expected YAC/reception (`xyac_mean_yardage`): `7.9250716963`
- YACOE/reception: `-0.5855616894`

### Non-2023
- actual YAC/reception: `8.1176100563`
- expected YAC/reception: `7.6519937717`
- YACOE/reception: `+0.4656162846`

### 2023 minus non-2023
- actual YAC difference: `-0.7781000494`
- expected-YAC difference: `+0.2730779246`
- YACOE difference: `-1.0511779740`

The catch/play context in 2023 was not inherently lower-YAC according to nflverse xYAC; expected YAC was actually about 0.27 yards/reception higher. The entire observed compression is therefore explained by substantially worse YAC relative to expectation. In signed decomposition terms, YACOE accounts for roughly 135% of the negative observed difference while expected-YAC context offsets about 35% of it.

This narrows the predictive question materially: the next study should test whether **strict-prior persistence/context of YAC above or below expectation** is pregame-identifiable, rather than simply predicting lower expected YAC from target design.

## Tail reference — explosive failures are strongly positive YACOE

RB1 rows that V2 moved from <30-yard AE under B1 to >=30-yard AE under C1, n=4:
- actual YAC/reception: `22.5`
- expected YAC/reception: `6.9425139086`
- YACOE/reception: `+15.5574860914`

RB1 rows 30+ AE under both B1 and C1, n=44:
- actual YAC/reception: `11.8935633811`
- expected YAC/reception: `7.8978036361`
- YACOE/reception: `+3.9957597449`

Thus catastrophic upside is dominated by realized post-catch production far above contextual expectation, reinforcing the architecture in which repeatable mean context and stochastic right-tail behavior remain separate. R22 remains the tail authority.

## Scientific implication

The viable next lane is not another generic YAC/YPR/YPT history model and not a broad expected-YAC target-design correction. It is a separately frozen strict-prior study of whether residual **YAC-over-expected state** has stable pregame predictors at player/offense/opponent level and whether that information can repair R27 RB1 mean translation without damaging RB2+, aggregate MAE, p90, Week1, or season stability.

This result authorizes only the writing/freezing of that predictive study. It does not authorize production or integration.
