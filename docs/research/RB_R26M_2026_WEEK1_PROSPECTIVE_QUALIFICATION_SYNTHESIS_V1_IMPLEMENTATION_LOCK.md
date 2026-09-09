# RB R26M 2026 Week-1 Prospective Qualification Synthesis V1 — Implementation Lock

Status: LOCKED BEFORE SYNTHESIS EXECUTION
Date: 2026-09-09

Frozen plan:
`docs/research/RB_R26M_2026_WEEK1_PROSPECTIVE_QUALIFICATION_SYNTHESIS_V1_FROZEN_PLAN.md`

Evaluator:
`scripts/backtest/synthesize_rb_r26m_2026_week1_prospective_qualification_v1.py`

Frozen-plan commit:
`4201985b23ba96b58e81a1b78a37bd91edc66cb4`

Evaluator implementation commit:
`fbdad82591e038d89235f70d6b9f4e0c638c3504`

Production comparison base:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

## Immutable parent contract

R26:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

Latest R26E:
- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`

R26J:
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`

R26K:
- run `34376961740`
- artifact `10114261724`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`

R26L:
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`

The workflow must verify all five digests before evaluator execution.

## Locked evaluator behavior

The evaluator may only:
- recursively locate exactly one frozen disposition JSON from each downloaded immutable parent artifact;
- compare inherited categorical disposition/gate/integrity fields to the exact rules in the frozen plan;
- write the two required R26M synthesis outputs.

It may not:
- import a prediction model;
- load current or historical football source tables beyond the immutable parent disposition JSONs;
- regenerate R26 predictions;
- fit/refit R9 or any model;
- use 2026 outcomes or participation;
- use sportsbook data;
- use same-week depth;
- invent or optimize a threshold;
- delete, reweight, or exempt 2020;
- alter R22, receiving-yard means, or production.

## Latest R26E authority lock

R26M must use run `34368268224` / artifact `10110785184` as the R26E authority. The older R26E execution represented elsewhere in the research tree is historical lineage only and may not substitute for this pinned parent.

The inherited R26E scientific pattern is locked as:
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`;
- exactly 20 frozen gates;
- exactly 19 true;
- sole false gate `14_no_w1_season_worsens_more_than_5pct`;
- exactly five improved Week-1 seasons;
- 2020 remains the harmful Week-1 season.

## Qualification ceiling

Even if every R26M inheritance condition passes, R26M authorizes only:

`r26n_design_authorized = true`

This means a separately frozen R26N-style prospective 2026 Week-1 candidate build/structural audit may be designed.

R26M always requires:
- `prospective_shadow_activation_authorized = false`;
- `production_promotion_authorized = false`;
- `exclude_2020_authorized = false`;
- `new_historical_router_authorized = false`;
- `r9_refit = false`;
- `predictions_regenerated = false`;
- `2026_outcomes_used = 0`;
- `sportsbook_football_inputs_used = 0`;
- `same_week_depth_used = false`;
- `production_parameters_changed = false`;
- `r22_changed = false`;
- `receiving_yard_means_changed = false`.

No scientific condition may be changed after R26M execution begins.