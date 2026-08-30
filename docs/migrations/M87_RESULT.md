# Migration 87 — Authoritative Result

## Disposition

`STABLE_PREGAME_DIFFERENTIATORS_FOUND`

Migration 87 completed the preregistered low-chaos catastrophic pregame atlas on the exact 38 M86 low-event-chaos 100+ yard QB passing misses.

M87 did not fit a predictive model. It compared the two primary low-chaos failure families with same-season, low-chaos, well-projected matched controls using only frozen model state and pregame/strictly-prior football information.

## Authoritative run

- GitHub Actions workflow: `Migration 87 QB Low-Chaos Pregame Atlas`
- Run: `33324313891` (Run #1)
- Conclusion: `success`
- Artifact: `m87-low-chaos-pregame-atlas`
- Artifact ID: `9735784411`
- Artifact SHA256: `4cc17715a719d7d40c3290c3ec676bfce607b3c06f6d0d03440ec0f34860ff5a`
- Repo CI: Run `33324313869`, success
- Historical Input Validation: Run `33324313861`, success
- Rows: `884`
- Full-stack 100+ misses: `123`
- Low-event-chaos 100+ misses: `38`
- Volume-dominant low-chaos: `16`
- Efficiency-dominant low-chaos: `17`
- Mixed low-chaos: `5`
- Predictive model fit: `False`
- Postgame event features used as atlas features: `False`
- Sportsbook features used: `False`
- Production actionable: `False`

## Stable preregistered differentiators

Four features cleared every frozen M87 gate: >=85% coverage, absolute combined SMD >=0.50, same sign in 2024 and 2025, and absolute SMD >=0.20 in each season.

| Failure family | Pregame feature | Target mean | Matched-control mean | Combined SMD | 2024 SMD | 2025 SMD |
|---|---|---:|---:|---:|---:|---:|
| Volume-dominant | Opponent defense prior pass rate faced | 0.626307 | 0.586122 | **+0.877921** | +0.793369 | +0.996710 |
| Volume-dominant | Target offense prior deep-attempt rate | 0.175916 | 0.194194 | **-0.630125** | -0.648922 | -0.586843 |
| Efficiency-dominant | Opponent defense prior success rate allowed | 0.415498 | 0.434224 | **-0.599075** | -0.304423 | -0.814846 |
| Efficiency-dominant | Opponent defense prior YPA allowed | 6.238596 | 6.615183 | **-0.573535** | -0.293564 | -0.783869 |

These are forensic differentiators, not yet predictive features.

## Volume interpretation

The cleanest M87 volume fingerprint is a **pass-funnel / short-intermediate volume environment**:

- the opposing defense had previously faced a materially higher pass rate than matched controls;
- the target offense had a materially lower recent deep-attempt rate.

The directional secondary atlas makes the mechanism more coherent. Thirteen of the 16 volume-dominant low-chaos catastrophes were **underprojections**. In those 13 rows:

- target offense prior pass rate: `0.619170` vs `0.585499` controls, SMD `+0.799580`;
- opponent defense prior pass rate faced: `0.623048` vs `0.585882`, SMD `+0.793937`;
- target offense prior deep-attempt rate: `0.173670` vs `0.193881`, SMD `-0.660479`;
- opponent offense prior success rate: `0.446769` vs `0.422916`, SMD `+0.685252`.

This suggests a plausible failure regime in which the model underestimates **sustained pass volume** in games whose pregame environment points toward repeated short/intermediate passing rather than deep-shot efficiency.

The directional atlas is exploratory and is not advancement-eligible by itself.

## Efficiency interpretation

The two stable efficiency differentiators both indicate that low-chaos efficiency catastrophes occurred against opponent defenses that had recently been **stronger against passing efficiency** than matched controls:

- lower success rate allowed;
- lower YPA allowed.

Ten of the 17 efficiency-dominant low-chaos catastrophes were **overprojections**. Their directional fingerprint was especially strong:

- opponent defense YPA allowed: `6.044615` vs `6.660607`, SMD `-0.984505`;
- opponent defense pass EPA allowed: `-0.043153` vs `+0.044259`, SMD `-0.676014`;
- opponent defense explosive-20 completion rate allowed: `0.076152` vs `0.086099`, SMD `-0.626881`;
- opponent defense success rate allowed: `0.411001` vs `0.431975`, SMD `-0.625230`.

This is consistent with an efficiency-suppression regime: a subset of strong pass defenses appears to produce ordinary-looking games in which the full-stack QB projection remains too optimistic even without a frozen M86 chaos event explaining the miss.

The seven efficiency underprojections had a different exploratory profile, including higher target-offense neutral pass rate and a large negative ML-minus-MC differential. Those small directional results are descriptive only and cannot promote a hypothesis.

## Existing-model rescue result

M87 found **no `MODEL_REPRESENTATION_CLUE`** under the preregistered gate.

### Volume-dominant low-chaos tails

- N: `16`
- ensemble MAE: `132.224864`
- hindsight best MC/ML/State MAE: `105.823828`
- oracle gain: `26.401036` yards
- at least one component below 75 yards error: `25%`
- at least one component below 50 yards error: `0%`
- hindsight best counts: MC `7`, ML `4`, State `5`
- largest best-model share: MC `43.75%`

### Efficiency-dominant low-chaos tails

- N: `17`
- ensemble MAE: `118.646636`
- hindsight best MC/ML/State MAE: `104.710368`
- oracle gain: `13.936267` yards
- at least one component below 75 yards error: `0%`
- at least one component below 50 yards error: `0%`
- hindsight best counts: MC `6`, ML `2`, State `9`
- largest best-model share: State `52.94%`

Therefore the M82 `41.103131` overall hindsight library oracle is **not** explained by the existing component library having a hidden good answer for these low-chaos catastrophic failures. The component models generally fail together on this subset.

## Scientific interpretation

M87 materially narrows the research problem:

1. Low-chaos catastrophic **volume** misses are mostly underprojections and show a repeatable high-pass-funnel / lower-deep-attempt pregame fingerprint.
2. Low-chaos catastrophic **efficiency** misses frequently include overprojections against defenses with stronger recent passing-efficiency suppression than matched controls.
3. Existing MC/ML/State model selection is not sufficient to solve these games.

This does **not** authorize simply adding static defensive pass rate, YPA allowed, or success rate to another Ridge/HGB/XGB model. Those information families overlap prior defensive-context research, especially M56 and related game-script work.

The new research question is whether the **conditional regime** exposed by M87 is architecturally unresolved — e.g. pass-funnel x short/intermediate offense for volume, and extreme defensive efficiency suppression for directional overprojection — rather than whether the raw static features have generic predictive value.

## M88 boundary

M88 should not tune on the same 2024/2025 M87 target outcomes.

Before any correction model is allowed, M88 should:

1. crosswalk the four M87 differentiators and the two coherent conditional regimes against M1-M87 to prove the proposed interaction/regime is not a relabeled M56/M67-M72 retest;
2. freeze the exact regime construction before scoring another season;
3. use a season not involved in M87 feature selection for confirmation, preferably a clean current-stack 2023 walk-forward reconstruction using 2022 prior history, if the historical pipeline can reproduce a valid cohort;
4. test regime **enrichment / direction** first, not optimize a pass-yard correction immediately.

Only a replicated untouched-season regime signal should graduate to a later full-stack predictive correction test.
