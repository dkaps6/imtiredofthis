# QB First-Down Score-State Decomposition V1 — Result

## Canonical execution

- Branch: `research-qb-first-down-score-state-decomp-v1`
- Frozen evaluator head before mechanical repair: `fa38cc3379157da5c5f6b5edc4cc314a621d81e2`
- Mechanical parity repair commit: `735a248fa76f0a11004350f2580c6515dfd354ca`
- Canonical execution head: `c874f5a5743f069714965d37ff794845629f1986`
- Canonical run: `34548863668`
- Job: `103107307133`
- Artifact ID: `10180050013`
- Artifact name: `qb-first-down-score-state-decomp-v1`
- Artifact digest: `sha256:969b3a3a6c435c08034d5f63dc88ca44991aa99e27b0eca54698cba174beaef9`
- Disposition: `FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC`
- Production actionable: `false`

## Mechanical lineage note

The first execution (`34546404935`) was not scientifically interpretable because the child evaluator excluded kneels while the immutable parent first-down universe retained them. The frozen hypothesis, score-state definitions, routing thresholds, reference construction, and scientific gates were not changed. The repair restored target-universe parity only. All scientific conclusions below come exclusively from canonical run `34548863668`.

## Integrity

Every frozen integrity gate passed after the mechanical parity repair.

- Exact parent rows: 884, split 444 in 2024 / 440 in 2025.
- Exact shared receiver cohorts: 440 WR-target and 884 pooled WR-reception rows.
- Score-state coverage: `1.000` in 2024, `1.000` in 2025, `1.000` pooled.
- Eligible/decomposable first-down plays: 14,953 / 14,953 in 2023; 14,886 / 14,886 in 2024; 14,517 / 14,517 in 2025.
- All strictly-prior-reference, uniqueness, shared-join, zero-model-fitting, zero-sportsbook, zero-production-change, and target-game-PBP-diagnostic-only gates passed.
- Maximum reconciliation errors were numerical noise only: all <= `2.220446049250313e-16`, far inside the frozen `1e-10` tolerance.

## Frozen routing result

### WITHIN_SCORE_STATE_PASS_PROPENSITY — PRIMARY

- pooled mean absolute parent-scaled contribution: `0.03574055185869413`
- 2024 mean absolute: `0.03555504299231492`
- 2025 mean absolute: `0.03592774716931315`
- largest pooled: yes
- >=20% pooled lead over second-largest: yes
- season stability: yes
- 2025 WR-target Spearman: `0.3388118552453079`
- pooled WR-reception Spearman: `0.3225090455074344`

It cleared all five frozen PRIMARY conditions.

### SCORE_STATE_OCCUPANCY — secondary, not primary

- pooled mean absolute: `0.02354084633667748`
- 2024 mean absolute: `0.02319004359599425`
- 2025 mean absolute: `0.023894838193185105`
- pooled WR-reception Spearman: `0.14083619665091407`
- 2025 WR-target Spearman: `0.28931498049623655`

Occupancy retained some shared signal but failed the frozen primary routing requirements and was materially smaller than within-score-state propensity.

### SCORE_STATE_REFERENCE_LEVEL — negligible

- pooled mean absolute: `0.002200406937696577`
- 2024 mean absolute: `0.00229176363068335`
- 2025 mean absolute: `0.002108219729319014`
- pooled WR-reception Spearman: `-0.02660748607336955`
- 2025 WR-target Spearman: `0.04305732232839294`

## Scientific conclusion

The surviving first-down play-choice miss is not mainly explained by where teams find themselves on the field, by realized score-state occupancy, or by a misspecified score-state reference level. The dominant shared QB/receiver opportunity-error component persists **after first down, field-position zone, and score state are held constant**.

This is strong evidence that the remaining mechanism is week-specific first-down pass/run choice uncertainty inside otherwise comparable football states.

The result does **not** authorize a correction, predictive model, or production promotion.

## Anti-reinvention / stopping rule

- Do not reopen generic M64/M65 score-state predictive families.
- Do not reinterpret M89 catastrophic-completion/tail work as new evidence.
- Do not respond by adding more postgame PBP transforms such as quarter/clock/state slicing merely because they are available.
- The next authorized step is a source audit for **genuinely new pregame target-game intent information** capable of explaining week-specific first-down pass/run choice.
- The already-tested first-down relative pass-vs-run efficiency-economics family is closed after D1 failure and cannot be retuned or rescued.
- Prior M67/M68 opening-script, broad offensive-intent, personnel/continuity, verified-playcaller, and competitive-leverage families must be treated as prior art during the source audit rather than rediscovered.
- Sportsbook/game-market information remains downstream only.
- Production remains unchanged.
