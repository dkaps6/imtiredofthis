# QB First-Down Choice Economics Source Audit V1 — Result

## Canonical execution

- Branch: `research-qb-first-down-choice-economics-source-v1`
- Frozen plan commit: `dd9f0d645256eb5f5345c4f674966c4f3d165d66`
- Evaluator commit: `579892e5708ee40f9c4e15d0dc407d5b08a61dc0`
- Canonical head: `39cb4ab85d009760d29a84535a242a8c63bc5ae4`
- Run: `34544776077`
- Job: `103094922496`
- Artifact: `10178612429`
- Artifact name: `qb-first-down-choice-economics-source-v1`
- Digest: `sha256:5a4928458d079f79edbaedfb1ef1ccf3f47f2d1bb76dbb39d3dfb01fa6fbe48a`
- Disposition: `FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_QUALIFIED`
- Production actionable: `false`

## Result

Every frozen source-eligibility gate passed. The family is qualified for exactly one separately frozen predictive development screen.

Target schedule coverage:

- 2023: 272 games / 544 team-weeks expected, 544 recovered;
- 2024: 272 / 544, 544 recovered;
- 2025: 272 / 544, 544 recovered.

All eight strictly-prior primitive quantities had 100% finite coverage in every target season. All four predeclared pass-minus-run difference descriptors also had 100% finite coverage in every target season.

## Historical source density

Across pooled 2023-2025 target team-weeks:

- median prior offense games: 8; p10: 8;
- median prior opponent-defense games: 8; p10: 8;
- median prior offense first-down pass-origin plays: 107; p10: 89;
- median prior offense first-down designed-run plays: 107; p10: 88;
- median prior opponent-defense first-down pass-origin plays faced: 108; p10: 89;
- median prior opponent-defense first-down designed-run plays faced: 107; p10: 89.

Every density share gate was 100%:

- >=4 prior offense games;
- >=4 prior opponent-defense games;
- >=40 prior offense first-down pass-origin plays;
- >=30 prior offense first-down designed-run plays;
- >=40 prior opponent-defense first-down pass-origin plays;
- >=30 prior opponent-defense first-down designed-run plays.

## Raw source audit

Eligible first-down plays:

- 2022: 14,754 = 7,323 pass-origin + 7,431 designed-run;
- 2023: 14,701 = 7,530 pass-origin + 7,171 designed-run;
- 2024: 14,635 = 7,242 pass-origin + 7,393 designed-run;
- 2025: 14,267 = 7,100 pass-origin + 7,167 designed-run.

EPA non-null coverage: `1.000`.

Success non-null coverage: `1.000`.

Pass-origin and designed-run choices were mutually exclusive under corrected M89-compatible semantics.

Every target feature used only games strictly before its target `(season, week)`.

## Scientific boundary

This result says only that the first-down choice-economics information object is historically reconstructable, dense, leakage-safe, and compatible with in-season nflverse updates.

No parent first-down residual, QB residual, WR residual, passing-yard outcome, sportsbook field, or predictive model was used in this migration.

The source result therefore authorizes exactly one separately preregistered 2023 predictive development screen. It does not authorize production integration.
