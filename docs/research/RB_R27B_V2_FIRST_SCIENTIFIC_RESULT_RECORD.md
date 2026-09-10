# RB R27B V2 Novel Efficiency Context — First Scientific Result Record

Status: `IMMUTABLE FIRST VALID SCIENTIFIC RESULT / NO INTEGRATION`

## Authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen plan commit: `bbaa0e2bbf32b182ca768555fa17548f8493be18`
- Frozen plan blob: `88c1377acc2e5389df092ec74ab876eaed468cdd`
- Implementation lock / workflow head: `2c86520c84fbdd5a18aad3f4b88373e5f8c17051`
- First valid scientific workflow run: `34428917229`
- Job: `102720004328`
- Artifact: `10134023092`
- Artifact name: `rb-r27b-v2-novel-efficiency-context`
- Artifact digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Scientific disposition

`R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`

- Gate count: 31
- Gate pass count: 24
- Integrity gates 1–15: ALL PASS
- Scientific gates: MIXED / FAIL
- Production changed: false
- R22 changed: false
- Sportsbook inputs upstream: 0

Failed gates:
- 16 vacancy MAE improve >=0.50% vs B1
- 19 vacancy RB1 MAE improve >=1.50% vs B1
- 20 vacancy RB1 MAE non-worse vs B0
- 22 vacancy RMSE non-worse vs B1
- 24 vacancy 30+ yard miss rate non-worse vs B1
- 26 2023 vacancy MAE improve >=2.00% vs B1
- 27 2023 vacancy MAE non-worse vs B0

## Key immutable metrics

### Vacancy-active, n=1761
- B0 MAE: 11.283018191924986
- B1/R27 MAE: 11.15926768815846
- C1/V2 MAE: 11.126700786821461
- C1 vs B1 MAE: -0.2918372625074398%
- C1 vs B0 MAE: -1.3854219008119515%
- B1 p90 AE: 25.374682142656468
- C1 p90 AE: 24.976930952354753
- C1 p90 vs B1: -1.5675120108522265%
- B1 RMSE: 16.281970690860362
- C1 RMSE: 16.292710282111756
- B1 30+ miss rate: 0.07211811470755253
- C1 30+ miss rate: 0.07268597387847814

### Vacancy RB1 incumbent, n=503
- B0 MAE: 14.305919424496233
- B1 MAE: 14.709394943903904
- C1 MAE: 14.666495644919056
- C1 vs B1: -0.2916455717481893%
- C1 vs B0: +2.5204686935773117%

### Vacancy RB2+ incumbent, n=941
- B1 MAE: 9.978918246636526
- C1 MAE: 9.945147839979008
- C1 vs B1: -0.33841751002320747%

### 2023 vacancy, n=259
- B0 MAE: 10.546563803678103
- B1 MAE: 11.065013055572372
- C1 MAE: 11.199725906086258
- C1 vs B1: +1.2174667109501847%
- C1 vs B0: +6.193127112930993%

### Season vacancy C1 vs B1 MAE
- 2020: -1.711203785332074%
- 2021: +0.4482619190292193%
- 2022: -0.08087978023553602%
- 2023: +1.2174667109501847%
- 2024: -0.7184060525174603%
- 2025: -0.6232932582709466%

### All RB
- B0 MAE: 10.993465478898194
- B1 MAE: 10.967611328093492
- C1 MAE: 10.960807399602038
- C1 vs B1: -0.06203655734978339%

### Week 1
- B1 MAE: 10.830435454603549
- C1 MAE: 10.809187904998515
- C1 vs B1: -0.19618370557762527%

## Scientific interpretation boundary

This result must not be described as a promotion candidate. C1 does not satisfy the frozen integration gates.

However, a failed integrated candidate does not erase component information. The following are valid hypothesis-generating observations from the immutable first result:

1. Novel contextual target-quality information contains some signal: pooled vacancy MAE and all-RB MAE improved modestly versus B1.
2. V2 repaired the R27 vacancy p90 weakness, lowering p90 absolute error by about 1.57% versus B1.
3. RB2+ vacancy incumbents improved modestly.
4. Four of six outer seasons improved versus B1 and no season worsened by more than 2%.
5. The dominant unresolved failure remains vacancy RB1 translation; C1 improved slightly versus B1 but remained materially worse than B0.
6. 2023 remains a distinct failure cohort and merits decomposition.
7. The 30+ miss-rate failure means the p90 repair did not uniformly repair catastrophic misses.

These observations may guide a NEW pre-specified forensic study. They may not be cherry-picked into production, used to alter V2 after seeing results, or treated as proof that any particular sub-feature should be promoted.

## Next authorized step

Perform a forensic-only decomposition of the preserved first-result evidence, focused on why R26 opportunity translation behaves differently for vacancy RB1 versus RB2+ and why 2023 diverges. The forensic study must create no new production candidate and must not mutate R26, R22, production YPT, or protected production files.
