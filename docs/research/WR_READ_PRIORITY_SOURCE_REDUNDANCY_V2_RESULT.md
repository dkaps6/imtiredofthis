# WR Read Priority Source/Redundancy Audit V2 — Result

**STATUS: SOURCE/REDUNDANCY SCREEN PASS ONLY. NO WR OUTCOME TEST. NO PRODUCTION CHANGE.**

## Canonical lineage

- branch: `research-wr-read-priority-source-audit-v1`
- semantic contract: `docs/research/WR_READ_PRIORITY_SEMANTIC_RESOLUTION_V2.md`
- workflow head: `b33f7fa315953fb06cb6abd736e6fcd8f45e1d9e`
- workflow run: `34912772828`
- artifact: `10375186236`
- artifact digest: `sha256:2632b456d166d517f293090637388e39466a1ee4c3600929ef76a0adec6293cb`
- exact WR-R15 authority run: `34238301577`
- exact WR-R15 authority artifact: `10061328722`
- exact WR-R15 authority digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

## Frozen V2 feature semantics

`EARLY_NO_EXTENDED_SHARE8`

- 2022 early/no-extended: `NA` or `DES`
- 2023+ early/no-extended: `0` or `DES`
- all-season extended progression: `1` or `2`
- `CHK`, `SD`, and unknown codes excluded from the binary denominator
- last 8 strictly-prior target-bearing games selected before progression filtering
- minimum 4 prior target-bearing games
- minimum 16 classifiable progression targets

This is not labeled first-read share. It is a broader receiver-specific progression-state feature.

## Source/novelty result

Disposition: **`READ_PRIORITY_R20_PLAN_ELIGIBLE_V2`**

- supported 2023 rows / WR-R15 authority coverage: **76.0116%**
- Spearman vs `entitlement_tgt_share`: **-0.0393056**
- Spearman vs `pred_targets`: **-0.0385956**
- R² explaining the feature from `entitlement_tgt_share + pred_targets + WR1 indicator`: **0.0181922**

Therefore the V2 feature is materially nonredundant with the existing WR-R15 opportunity/entitlement structure under the frozen screen.

## Guards

PASS:
- exact WR-R15 artifact lineage
- source seasons 2022/2023/2024 present
- source-only workflow
- no WR receiving-yard outcomes loaded
- no outcome fields present in the feature panel
- only 2023 authority opportunity/identity fields materialized
- no 2024 WR-R15 projection/outcome fields parsed
- sportsbook inputs = 0
- no production change

## What this does and does not authorize

This result **does not** establish that read priority predicts receiving-yard residuals.

It establishes only that:
1. the semantic contract is source-defensible under the maintainer-approved FTN key;
2. the V2 progression feature has enough supported historical coverage;
3. it is not simply a repackaging of WR-R15 target entitlement or projected target volume.

A separate R20 football-outcome plan must be frozen and adversarially reviewed before any 2023 receiving-yard outcome is opened. The 2024 WR holdout remains sealed.