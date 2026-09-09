# RB R26 Receptions Production Integration V1 — Implementation Lock

**Locked before qualification execution/results.**

## Governing frozen contract

The governing plan is V2:
- file `docs/production/RB_R26_RECEPTIONS_PRODUCTION_INTEGRATION_V2_FROZEN_PLAN.md`
- plan commit `ca5b641f077964fad58df5f6d1a9c5a4188fcbf1`

V1 plan commit `fd591a54a5a62169148c102fefef3040d2e9cb60` is preserved as superseded-before-run lineage because its original Gate 27 would have bypassed the existing receptions ensemble. No qualification result existed when V2 corrected that contract.

## Locked implementation

Pinned R26L Week-1 room-state asset:
- file `data/models/rb_r26_production_v1/rb_r26_week1_room_state_v1.csv`
- introduction commit `b148de8686eb2df1281ab0bf0153a88658bb422f`
- expected exact file SHA-256 `27ad7bad8fcdfa6b0b1090994e0c0d2bdc4c0e45209d2409abc9574b5d354258`
- exactly 31 frozen vacancy teams; CIN absent as sole non-vacancy control.

R26 production adapter:
- file `scripts/modeling/rb_r26_receptions_production_adapter_v1.py`
- implementation commit `d3a9a25e86513c674ff494cfb6dc236f34aca960`

Full Slate V5 candidate entrypoint:
- file `scripts/run_pricing_with_full_roster_universe_v5_production.py`
- implementation commit `8c2770b8aedbdb6cfbddf778a12ba6f2cfa20389`

Frozen qualification evaluator:
- file `scripts/validate_rb_r26_week1_production_integration_v1.py`
- implementation commit `83619f2cdedac95fcacbb81c59feb9beda16845b`

Exact previously proven mechanical identity-staging helper:
- file `scripts/backtest/stage_r26n_production_identity_key_repair_v1.py`
- branch introduction commit `138b4206b3f2b8b11de900885dff6445d5a64489`
- blob SHA `e9fce50aad9e2f200bd4f9991f7f089f109f2b6c`, exactly matching the helper used by the successful Week-1 RB operational-readiness run.

## Parent input locks

Protected production / V4 parent:
- protected code head `f8417f55b04ce0e19baf260e9d532765034c47f1`
- run `34317211395`
- artifact `10090547415`
- name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`

Exact downstream market snapshot for fixed control-vs-candidate pricing comparison:
- R26R run `34401814588`
- artifact `10124274040`
- name `rb-r26r-2026-week1-prospective-observation-snapshot-v1`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- this artifact supplies the already-captured sportsbook offers only; it does not feed football generation.

Pinned R19/R9 identity model:
- repo file `data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json`
- expected exact file SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- training season 2025
- no fit/refit permitted.

## Locked execution design

Qualification must:
1. checkout this branch from a clean runner;
2. prove the V2 plan and all locked implementation files are unchanged from their pinned commits;
3. prove pre-existing production files remain byte-identical to protected head `f8417...`;
4. verify exact parent artifact digests/heads;
5. download the exact protected Full Slate parent artifact and exact R26R market snapshot;
6. apply only the previously proven exact one-to-one `player_clean_key` staging repair to the downloaded production parent, leaving the immutable parent untouched;
7. stage those exact football inputs into the clean checkout and verify repo-pinned R19/R26 assets;
8. stage the exact R26R `props_pricing_offers.csv` as the fixed downstream `outputs/props_raw.csv` offer board;
9. rebuild deterministic `metrics_ready.csv` from those fixed offers + protected football state without any new sportsbook fetch;
10. run V4 control pricing and preserve its exact priced output;
11. run V5 R26 candidate pricing on the exact same inputs;
12. evaluate all 35 frozen V2 gates;
13. upload the complete gate/disposition/control/candidate/adapter/lineage evidence.

No current/future outcomes, same-week results, OddsAPI fetch, tuning, router search, blend search, or scientific-parameter changes are permitted during qualification.

## Authority ceiling

A 35/35 PASS may authorize the already user-approved production promotion. It does not authorize unrelated changes to P3, R22, QB, WR, TE, ensemble weights, or sportsbook boundary semantics.

Expected PASS disposition:
`RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`

Expected FAIL disposition:
`RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_FAIL_NO_PROMOTION`
