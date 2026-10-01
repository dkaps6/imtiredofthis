# WR-CB Provider-ID Reuse Guard V1 Result — 2026-09-30

**Disposition: FAIL-CLOSED SOURCE REPAIR PASSED. SOURCE/MODEL GATE REMAINS CLOSED.**

## Why this was required

Claude's independent provider-ID census found that some FantasyAlarm provider IDs are reused for different people. Anchored GSIS collision counts alone cannot catch a reused ID when the second person's normal roster join is unresolved.

Concrete reproduced defect from the preserved source artifact:
- FantasyAlarm WR provider ID `300936` appears for Marquise Goodwin and later Allen Robinson II.
- Goodwin anchored to GSIS `00-0030068`.
- The old bridge propagated `00-0030068` onto **nine 2022 Allen Robinson II rows**.
- Six of those rows had previously counted in the 4,063 provisional source-quality-ready diagnostic.

This is a source-identity bug, not football-model science.

## Repair

Commit `abd83eea9823848ae1eb5d798bf9e44b912c4d62`:
- pregame/schedule anchor rule preserved;
- provider IDs spanning demonstrably different/ambiguous person-name families are a **negative veto** before bridging;
- known benign aliases/suffix variants are normalized only for the veto screen;
- composite projected corner strings (for example a slash-separated two-CB assignment) cannot certify one stable player identity;
- independently resolved weekly/season roster identities remain untouched;
- regression demonstrates why the old unique-GSIS anchor rule missed the Goodwin/Robinson case.

Validation:
- Repo CI `36734473138` = **SUCCESS**.
- WR-CB source audit `36734472417` = **SUCCESS**.
- Exact new artifact `11106407106`, digest `sha256:a6a1bfde25f3587a235e63f7c78a9d051c7a690429306ab731c7be6a66db2077`.

## Exact source effect

- rows: **4,459** unchanged;
- former provisional source-quality-ready: **4,063**;
- repaired provisional source-quality-ready: **4,034**;
- **29 rows** lost provisional eligibility because they depended on uncertain/reused provider-ID bridging;
- WR uncertain/reused provider IDs quarantined: **9**;
- CB uncertain/reused provider IDs quarantined: **3**;
- bridge rows blocked by reuse: WR **31**, CB **3**;
- remaining WR bridge rows: **96**; CB bridge rows: **373**;
- Allen Robinson II rows using provider ID `300936`: now all **UNRESOLVED**, none source-quality-ready.

Content-version limitation after this repair:
- among the 4,034 provisional-ready rows, **3,558** are only `METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF`;
- **476** provisional-ready rows still come from pages with reported post-kickoff modification;
- independently verified historical pregame content snapshots remain **0**.

Therefore 4,034 is still **not model-ready data**.

## Week-3 settlement clarification from the same coordination cycle

Claude separately inferred six additional Week-3 DNP voids from absence in the weekly player-stat table. That inference is invalid under the canonical grading contract. The grader explicitly uses PFR snap-count participation because weekly player stats can omit a player who played but recorded zero box-score usage. The canonical rows for Elijah Arroyo, Erick All and Blake Whiteheart are roster `ACT`, snap-participated `True`, and `snap_confirmed_verified_zero`. No Week-3 regrade is authorized.

No production/model/weight/threshold change. No paid data.