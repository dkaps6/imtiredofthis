# WR Read Priority Control Feasibility V3 — Result

**STATUS: SOURCE/CONTROL SCREEN PASS ONLY. NO WR RECEIVING-YARD OUTCOME TEST. NO PRODUCTION CHANGE.**

## Canonical lineage

- branch: `research-wr-read-priority-source-audit-v1`
- V2 semantic/source result: `docs/research/WR_READ_PRIORITY_SOURCE_REDUNDANCY_V2_RESULT.md`
- workflow head: `06cc7c10a14bc9f2db79b8b59b06af51fff55192`
- workflow run: `34913439449`
- artifact: `10375002416`
- artifact digest: `sha256:80b82ccb910e643868da019e0ea7941b8a9e03ac281b15476f52554230734fe1`
- exact WR-R15 authority run: `34238301577`
- exact WR-R15 authority artifact: `10061328722`
- exact WR-R15 authority digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

## Purpose

Claude's adversarial V2 review passed the primary source/novelty screen conditionally on two controls being available before an R20 outcome plan is frozen:

1. `TEAM_EARLY_NO_EXTENDED_SHARE8` to separate receiver tendency from team scheme tendency;
2. `MEAN_AIR_YARDS_PER_TARGET8` to separate progression state from target-depth adjacency to WR-R17.

This audit tested those controls without loading WR receiving-yard outcomes.

## Frozen control mechanics used in this audit

Primary feature remains unchanged:

`EARLY_NO_EXTENDED_SHARE8`

Team control:
- same target offense;
- last 8 strictly-prior team target-bearing games;
- V2 classifiable progression events only;
- minimum 40 classifiable team targets.

Target-depth control:
- same resolved receiver GSIS ID;
- last 8 strictly-prior receiver target-bearing games;
- mean PBP `air_yards` across non-null receiver targets;
- minimum 12 valid air-yard targets.

No alternate control construction was searched.

## Result

Disposition: **`READ_PRIORITY_R20_CONTROL_SET_ELIGIBLE`**

- V2 primary-supported rows: **1,578**
- team-control coverage among primary-supported rows: **100.0%**
- air-yards-control coverage among primary-supported rows: **100.0%**
- Spearman(`EARLY_NO_EXTENDED_SHARE8`, `TEAM_EARLY_NO_EXTENDED_SHARE8`): **+0.231844**
- Spearman(`EARLY_NO_EXTENDED_SHARE8`, `MEAN_AIR_YARDS_PER_TARGET8`): **-0.512643**
- expanded R² explaining receiver progression share from `entitlement_tgt_share + pred_targets + WR1 + team progression share + mean air yards/target`: **0.348071**

Interpretation:
- team scheme contributes some structure but does not dominate the receiver feature;
- target depth is meaningfully related, as expected, but the feature is not reducible to target depth under the frozen redundancy screen;
- after including both required controls, roughly 65% of the feature variance remains unexplained by the tested opportunity/scheme/depth structure.

This does **not** establish outcome predictiveness. It only clears the control-feasibility requirement for a separately frozen R20 football-outcome experiment.

## Guards

PASS:
- exact WR-R15 authority lineage;
- only 2023 non-outcome authority fields materialized;
- no WR receiving-yard outcome fields loaded or present in the panel;
- no 2024 WR-R15 projection/outcome fields parsed;
- sportsbook inputs = 0;
- no production change.

## What this authorizes

A separate R20 plan may now be frozen prospectively using:
- primary signal `EARLY_NO_EXTENDED_SHARE8` exactly as V2 defines it;
- team scheme control `TEAM_EARLY_NO_EXTENDED_SHARE8`;
- target-depth control `MEAN_AIR_YARDS_PER_TARGET8`;
- unchanged WR-R15 opportunity/role controls.

No WR outcome run is authorized by this document itself.
