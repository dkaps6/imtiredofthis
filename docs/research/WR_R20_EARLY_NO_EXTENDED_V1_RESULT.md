# WR-R20 Early / No-Extended Progression V1 — Canonical 2023 Stage-A Result

## Status

Formal experiment disposition: **`NO_ACTIONABLE_WR_EARLY_NO_EXTENDED_SIGNAL`**.

The frozen 2023 development test failed the raw Stage-A gates. Mediation / robustness was **not evaluated** because the raw test failed. The 2024 holdout remained sealed and no 2024 WR-R15 projection/outcome cells were parsed.

This result closes WR-R20 under its frozen V1 contract. No threshold, window, WR1-only, tail-only, `CHK` rebucketing, direction flip, interaction, or other rescue is authorized.

Scientific-evidence taxonomy label is intentionally left pending the independent Claude result audit requested in Issue #535 comment `5673369352`; the formal experiment disposition above is final regardless of whether the evidence label is ultimately `NO_DIRECTIONAL_EVIDENCE` or `DIRECTIONAL_CONTRADICTION`.

## Canonical execution lineage

- branch: `research-wr-r20-early-no-extended-v1`
- frozen plan commit: `0f83ec22f22cdb7e35fcab7ef056225a7c806f54`
- pre-implementation CHK amendment commit: `b6ca20f89a017208579e533bab1874895a1d1bbe`
- evaluator commit: `697af0167754fdcf323e710af3ad36bc8e15bf4c`
- synthetic suite commit: `ceca396c2edf48c1ecc8d830f8260b543a788bf5`
- synthetic workflow head: `655aed55a31b82a3f9570ebab5f82e86f561beb5`
- synthetic run: `34917337515` — SUCCESS
- Claude implementation review: Issue #535 comment `5673308051` — `IMPLEMENTATION_REVIEW_PASS`
- Stage-A workflow head: `5feacb5798e6b2f8523451264fbe8601e097c537`
- canonical Stage-A run: **`34917971041` — SUCCESS**
- canonical job: **`104219624185`**
- artifact: **`10376188752`**
- artifact name: `wr-r20-early-no-extended-stage-a-v1`
- artifact digest: **`sha256:95b9d68fb338f1fc9034df6f1833ce3d39a23133481ebe64776fbb62d9e2ce20`**
- artifact size: `181749` bytes

Exact WR-R15 authority:

- run: `34238301577`
- artifact: `10061328722`
- artifact name: `wr-r15-wr1-anchor-participation-v1`
- digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- variant: `WR_R15_WR1_ANCHORED_PARTICIPATION`
- 2023 development rows: `2076`
- 2024 sealed holdout rows: `2117`

## Frozen signal

`EARLY_NO_EXTENDED_SHARE8`

- 2022 EARLY/NO-EXTENDED = FTN `NA | DES`
- 2023+ EARLY/NO-EXTENDED = FTN `0 | DES`
- EXTENDED = FTN `1 | 2`
- `CHK`, `SD`, missing/unknown excluded from the binary denominator
- last 8 strictly-prior target-bearing games selected before progression classification/filtering
- minimum 4 prior target-bearing games
- minimum 16 classifiable progression targets
- frozen expected direction: **negative**

## Raw Stage-A result

| Metric | Frozen gate | 2023 result | Status |
|---|---:|---:|---|
| supported rows | — | 1,578 | — |
| coverage | >= 0.60 | **0.7601156069** | PASS |
| Spearman(signal, yard residual) | <= -0.08 | **+0.0472021392** | FAIL / opposite sign |
| Q4-Q1 yard-residual gap | <= -5.0 yd | **+3.7135138 yd** | FAIL / opposite sign |
| Q1/Q4 actual 100+ ratio | >= 1.20 OR other tail gate | **0.8485499x** | FAIL / opposite orientation |
| Q4/Q1 residual <= -30 ratio | >= 1.20 OR other tail gate | **0.8838608x** | FAIL / opposite orientation |
| WR1 Q4-Q1 gap | negative when n>=150 | **+2.4127144 yd** (n=494) | FAIL / opposite sign |
| WR2+ Q4-Q1 gap | negative when n>=150 | **+3.9487939 yd** (n=1,084) | FAIL / opposite sign |

Raw development supported: **false**.

Mediation evaluated: **false**.

Do **not** call the robustness step a mediation failure; it was never run.

## Quartile descriptives

| Quartile | n | residual MAE | signed residual bias | actual 100+ | residual <= -30 |
|---|---:|---:|---:|---:|---:|
| Q1 | 395 | 24.1709 | +4.5916 | 7.34% | 8.10% |
| Q2 | 394 | 26.7529 | +9.1232 | 11.17% | 7.61% |
| Q3 | 394 | 25.7337 | +11.0763 | 11.17% | 6.09% |
| Q4 | 395 | 24.0542 | +8.4506 | 8.86% | 7.09% |

The quartiles are not monotonic in the frozen negative direction. Q2/Q3 carry the largest positive bias and 100+ rates, while Q1/Q4 are closer in MAE. This does not support a threshold rescue: all such post-result reshaping is forbidden by the frozen contract.

## Identity / source / leakage audit

- authority development rows: `2076 / 2076`
- stable-ID resolved: `2055`
- unmatched: `21`
- weekly-roster identity rows: `2055`
- PBP exact-name fallback attempts: `21`
- team-disambiguated rows: `16`
- rows with >=4 prior target games: `1872`
- rows with >=16 classifiable progression targets: `1590`
- rows with valid primary signal: `1578`
- team control available: `2076`
- air control available: `1799`
- authority display-key mismatches: `0`
- target-game leakage: `0`
- team-control leakage: `0`
- air-control leakage: `0`
- sportsbook inputs: `0`
- 2024 scored: `false`
- 2024 projection/outcome fields parsed: `false`

FTN/PBP exact join rate:

- 2022: `1.0`
- 2023: `1.0`

Unknown progression codes:

- 2022: `0`
- 2023: `0`

Selected but denominator-excluded states under the frozen contract:

- `CHK`: `2705` target events
- `SD`: `5188` target events
- missing progression: `0`

## Source hashes

2022:

- FTN: `sha256:2846ef9bcc5f7298a5c783fb9a5edc85b3820c0b4943bf60a1217398e412985b`
- PBP: `sha256:931121d8897779d7944e2a293e92ed8799c8e5cceef84096ac42339003fedc09`

2023:

- FTN: `sha256:162f0f48a8fca23f86a334c78e27f8daa7443e6b383beada0416b65c593b2c0e`
- PBP: `sha256:bd3484731408def6b0ec93225bba2bd7b2c65769ca707a2b9444d891abdc6776`

## Scientific interpretation

The source and identity mechanics are healthy; coverage comfortably cleared the frozen minimum. The test failed because the football relationship did not follow the preregistered negative direction.

Every frozen endpoint-level directional diagnostic leaned positive rather than negative: correlation, Q4-Q1 residual gap, both role slices, and both frozen tail orientations. The reverse magnitudes are modest and do not authorize flipping the hypothesis or creating a positive-direction R20 rescue.

The program-level implication is therefore the jointly preregistered roadmap trigger, not a new downstream feature:

1. close R20;
2. keep 2024 sealed;
3. stop the R17-R20 downstream target-quality / post-opportunity feature hunt;
4. move to the authority-exact M34/M35-informed opportunity-attribution audit;
5. do not build a challenger until that diagnostic isolates a material structured layer and a genuinely fresh validation route exists.

## Production eligibility

None.

- no production change
- no direct model change
- no sportsbook feature selection
- no paid Full Slate
- no RB work
