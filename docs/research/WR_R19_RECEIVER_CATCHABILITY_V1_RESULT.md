# WR-R19 Receiver-Specific Catchability V1 — Result

**STATUS: CLOSED. RESEARCH ONLY. NO PRODUCTION CHANGE. 2024 HOLDOUT REMAINS SEALED.**

## Authority and execution

- Branch: `research-wr-r19-receiver-catchability-v1`
- Frozen plan: `8d2318aca2d5a48906e23ad503641c3a06f70406`
- Original evaluator: `d8c81d0aeeb730982d186294d7bfe0308349addd`
- Original synthetic suite: `d70c54d68902b81a4fb7a9175385cea3c920f390`
- Pre-outcome synthetic workflow head: `19f800f8da4b5265edde255f66c83f1a856d3f20`
- Initial real-run head: `091368525ea353118fb303980dd018f35d308aa0`
- Initial real run: `34909243233` — `MECHANICALLY_INVALID` before any valid football output because the exact FTN/PBP merge retained colliding `season/week` columns and the evaluator referenced a nonexistent unsuffixed `season` field.
- v1b source fix: `acccab8794d0077d0c318c04c1c95e25dea34505`
- v1b regression tests: `4c249f70c34cee9bf0029cdf825ac4cd7ffce30b`
- Canonical v1b workflow head: `a750b72370262df41e34197ade0e76e85b2ea184`
- Canonical Stage-A run: `34909421222` — SUCCESS
- Job: `104193360578`
- Artifact: `10373597805`
- Artifact digest: `sha256:6bca6db9d099f430a882ea714eeec3d58306fe2e8b71510fed59923b664d3be2`

WR-R15 authority was pinned exactly:
- run `34238301577`
- artifact `10061328722`
- name `wr-r15-wr1-anchor-participation-v1`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

## Mechanical review

Claude independently returned `IMPLEMENTATION_REVIEW_PASS` on the pre-result implementation, then after the first real-run source collision independently re-reviewed v1b and returned `V1B_MECHANICAL_REVIEW_PASS` in Issue #535 comment `5672355081`.

The v1b review confirmed:
- FTN and PBP `season/week` are collision-safe and parity-asserted after exact game/play joins;
- only 2022/2023 FTN/PBP feature sources are loaded;
- the 2024 authority season is counted/key-audited for artifact parity but its projection/outcome fields are not parsed;
- only the two source loaders changed; frozen signal, support floors, raw gates, mediation gates, and football calculations stayed unchanged;
- both synthetic suites passed independently;
- the canonical job's explicit 2024/source/target-population guard passed.

## Frozen hypothesis

Primary candidate only:

`WR_TARGET_CATCHABLE_RATE8`

= arithmetic mean FTN `is_catchable_ball` across all official receiver target events in the last 8 strictly-prior target-bearing games, with the last 8 games selected before event aggregation.

Support:
- >=4 prior target-bearing games
- >=16 valid catchability target events
- fixed positive direction

Frozen raw Stage-A gates:
1. coverage >=60%
2. Spearman >= +0.08
3. Q4-Q1 receiving-yard residual gap >= +5.0 yd
4. actual 100+ yard ratio >=1.20 OR +30-yard underprojection ratio >=1.20
5. WR1 and WR2+ Q4-Q1 gaps positive if slice n>=150
6. source/identity/leakage audits clean

Mediation would run only after a raw pass. Because raw Stage A failed, mediation was **not executed**. The result field `mediation_supported: false` must not be interpreted as a separately scored mediation failure.

## Canonical 2023 Stage-A result

| Gate / metric | Result | Frozen requirement | Disposition |
|---|---:|---:|---|
| N | 1,667 | descriptive | — |
| Coverage | 80.30% | >=60% | PASS |
| Spearman | -0.000404 | >=+0.08 | FAIL |
| Q4-Q1 residual gap | -0.685 yd | >=+5.0 yd | FAIL |
| 100+ yard ratio | 1.110x | >=1.20 if used | FAIL |
| +30-yard underprojection ratio | 0.861x | >=1.20 if used | FAIL |
| WR1 slice | n=495, +1.161 yd | positive | PASS direction only |
| WR2+ slice | n=1,172, -1.530 yd | positive | FAIL |
| Source/identity/leakage audits | clean | clean | PASS |

Formal experiment disposition:

`NO_ACTIONABLE_WR_RECEIVER_CATCHABILITY_SIGNAL`

Maps to methodology label:

`FAIL`

Scientific evidence disposition:

`NO_DIRECTIONAL_EVIDENCE`

Rationale: the pooled correlation is effectively zero, the high-vs-low residual gap is slightly opposite the preregistered direction, neither tail ratio clears the frozen threshold, and the large WR2+ slice moves opposite the hypothesized direction. The small positive WR1 gap (+1.16 yd) is not large enough to constitute meaningful partial directional evidence against the otherwise near-null/incoherent full result.

## Descriptive quartiles

| Quartile | n | MAE | Signed residual bias | Actual 100+ rate | +30-yard underprojection rate |
|---|---:|---:|---:|---:|---:|
| Q1 | 417 | 22.984 | +6.790 | 6.71% | 20.38% |
| Q2 | 417 | 26.974 | +9.451 | 10.07% | 23.98% |
| Q3 | 416 | 26.382 | +10.042 | 12.50% | 23.56% |
| Q4 | 417 | 22.238 | +6.150 | 7.43% | 17.51% |

There is no monotonic improvement in the hypothesized direction. The middle quartiles, not the highest-catchability quartile, carry the larger positive residual bias and tail rates.

## Source and identity audit

Canonical artifact audit:
- development rows: 2,076 / expected 2,076
- identity mode: 2,055 stable ID; 21 unmatched
- identity source: 2,055 strictly-prior weekly roster; 21 exact-name PBP fallback attempts
- team-disambiguated rows: 16
- rows with >=4 prior target games: 1,872
- rows with >=16 valid catchability targets: 1,679
- rows with valid final signal: 1,667
- team control available: 2,076 / 2,076
- target-game leakage rows: 0
- team-control target-game leakage rows: 0
- sportsbook inputs: 0
- holdout 2024 scored: false

Source rows:
- 2022: 17,306 official receiver-target rows, exact FTN->PBP join rate 100%, catchability coverage 100%, 11,605 completions / 5,701 incompletions, season/week parity PASS
- 2023: 17,483 official receiver-target rows, exact FTN->PBP join rate 100%, catchability coverage 100%, 11,808 completions / 5,675 incompletions, season/week parity PASS

Resolved-receiver target-population audit:
- 18,976 target events
- 18,976 non-null catchability events
- 12,041 completions
- 6,935 incompletions
- both completions and incompletions present: true

## Positive findings worth preserving

- The source path is unusually clean: exact game/play FTN->PBP joins, direct stable receiver IDs, full catchability population, and no target-week leakage.
- The v1b loader now provides a stronger holdout discipline than the original implementation by not parsing sealed 2024 projection/outcome fields.
- WR1 direction was mildly positive (+1.16 yd), but this is descriptive only and does not justify a WR1-only rescue because that subgroup was not the primary frozen candidate and the full evidence is near-null.

## Negative findings / closed rescue paths

This result closes the receiver-specific historical catchability-rate lane under the frozen R19 information set.

Do **not** rescue R19 by:
- changing the trailing window;
- lowering support floors;
- selecting a catchability threshold after seeing 2023;
- switching to WR1-only;
- replacing Spearman with a tail-only criterion;
- adding drops, contested-ball, read-progression, created-reception, or other FTN fields post hoc;
- inspecting 2024 to search for a better shape;
- combining R19 with R18 CPOE or R17 target-depth shape after the fact.

Any future use of another FTN field must be a materially new football mechanism, prospectively frozen before untouched evidence is exposed, and must survive the existing anti-retest map.

## Holdout status

2024 remains sealed for WR-R19. Because raw 2023 Stage A did not pass, the frozen contract does not authorize Stage B.

## Production eligibility

`NO`

No production science, model weight, threshold, Full Slate behavior, or sportsbook process changes are authorized by WR-R19.
