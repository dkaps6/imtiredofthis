# RB Workhorse Transition Gate V1 — Final Result

**STATUS: TERMINAL / CLOSED — `RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED`**

Research only. No production change. No rescue tuning authorized.

## Canonical final evidence

- Branch: `research-rb-workhorse-transition-gate-v1`
- Final evaluated head: `bc56c80664a93b8691baf932312b9ccb6405749c`
- Canonical final workflow run: `35253048097`
- Final one-shot artifact: `10513292695`
- Artifact digest: `sha256:e8d632b2b602ae108090e74fe08ceb0d3bd0ec8bf88fef4b12af079b67e7a78c`
- Issue #535 final Claude checkpoint: `5719134779`
- Frozen plan lineage: `7fac65c960422bd87d40f45cff85ba5f4fc55c85` -> `bbdd521d4aafd2c2046102dfbba526b2b43f13c1` -> `1e69934743f5d4db79b5914ac84d6092af136aa6`, plus schedule-domain amendment `e4c8520d22bb89bd94ec82062a78e5f1a8d3bd75` and pre-run audit fix head `bc56c80664a93b8691baf932312b9ccb6405749c`.

## Mechanical / pre-outcome lineage preserved

The final scientific result must not erase the earlier implementation history.

1. Run `35237128188`, head `57f3dbb8...`: mechanical failure from incorrectly attempting 2019 Week 18.
2. Run `35239506651`, head `b855181a...`: season-length repair succeeded; evaluation then stopped on missing caller-side `name_key` normalization.
3. Run `35243854637`, head `e8f5532e06377872003c3c7fd7feaa21cd3b021d`: full pre-outcome construction ran and fail-closed on 5/655 events. Artifact audit proved all 5 were non-game calendar-domain rows (`2019 W18 NO`, `2020 W18 BUF`, `2020 W18 NO`, `2020 W18 LAR`, `2020 W5 DEN`), not genuine football events. No outcomes were opened.
4. Schedule-domain correction at `e4c8520d22bb89bd94ec82062a78e5f1a8d3bd75` intersected the frozen scored transition population with canonical scheduled target-game team-weeks, preserving excluded rows with `NO_SCHEDULED_TARGET_GAME`.
5. Final run `35253048097` then completed end to end under the unchanged frozen science.

These are mechanical/data-integrity repairs only. The classifier family, feature contract, target, cutoff grid, confirmation gates and no-rescue rules were not altered after scientific results were exposed.

## Final pre-outcome integrity

- Gate 0.2 injury/status integrity: PASS.
- Gate 0.3 roster structural/event checks: PASS.
- Raw scored transition rows before schedule-domain correction: `655`.
- Scheduled real-game events retained: `650`.
- Non-game rows excluded: `5`, exactly matching the prior audit.
- Feature construction: `WORKHORSE_GATE_V1_FEATURES_CONSTRUCTIBLE`.
- Every retained event produced a complete finite 13-feature row.
- Sportsbook inputs used: `0`.
- Outcomes were attached only after all pre-outcome integrity checks passed.

## Frozen two-rotation result

### Rotation A

Chronology: fit 2019-2020 -> cutoff selection on 2021 -> untouched confirmation on 2022.

- Fit n: `247`
- Fit positives: `49`
- Selected cutoff: `0.80`
- Cutoff-year F0.5 at selected cutoff: `0.379`
- Cutoff-year predicted positives: `8`
- 2022 confirmation n: `113`
- 2022 positives: `25`
- 2022 predicted positives at frozen cutoff: `0`
- Disposition: `NOT_QUALIFIED`
- Detail: `PRECISION_UNDEFINED_NO_PREDICTED_POSITIVES`

### Rotation B

Chronology: fit 2019-2021 -> cutoff selection on 2022 -> untouched confirmation on 2023.

- Fit n: `414`
- Fit positives: `83`
- Selected cutoff: `0.50`
- Cutoff-year F0.5: `0.355`
- 2023 confirmation n: `123`
- 2023 positives: `23`
- Precision: `0.255`
- Recall: `0.565`
- ROC-AUC: `0.637`
- PR-AUC: `0.289`
- `precision_vs_prevalence_margin`: FAIL
- `precision_floor >= 0.60`: FAIL
- `recall_floor >= 0.25`: PASS
- `roc_auc_floor > 0.60`: PASS
- `pr_auc_vs_prevalence`: PASS
- `sportsbook_inputs_zero`: PASS
- Disposition: `NOT_QUALIFIED`
- Detail: `gate_failure`

## Final disposition

**`RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED`**

Neither independent historical rotation confirmed. This is not `INSUFFICIENT_EVIDENCE`; sample adequacy passed and the classifier executed normally.

The one-time 2024-2025 transport through the frozen Lane-A V2 router was correctly **NOT EXECUTED**, because both historical rotations had to confirm before transport was authorized.

## Scientific interpretation

The exact V1 architecture is closed. There is descriptive evidence of pregame discrimination in Rotation B (ROC-AUC `0.637`, PR-AUC above prevalence, useful recall), but it did not produce a stable high-precision activation rule. Rotation A became too selective and fired zero times in confirmation; Rotation B fired more often but precision was only `0.255`, far below the preregistered `0.60` safety floor.

This supports only a hypothesis-generation statement: some strictly pregame RB-role/transition information may contain predictive structure about future concentrated workloads. It does **not** validate a production gate and does **not** authorize lowering the precision floor, changing the cutoff grid, switching classifier family, adding interactions, selecting only Rotation B, changing the target, or otherwise rescuing V1 on the same exposed evidence.

## Permanent stop rule for this family

Do not reopen `RB_WORKHORSE_TRANSITION_GATE_V1` by:

- changing the `>=20 carries` target after seeing these results;
- lowering confirmation precision/recall/AUC gates;
- searching new probability cutoffs on 2019-2023;
- changing logistic hyperparameters/model family on the same exposed chronology;
- adding/removing features to rescue this exact gate;
- using only the better-looking Rotation B;
- blending Rotation A and B;
- executing 2024-2025 V2 transport despite failed historical confirmation.

Any future RB workload/concentration study must be a separately justified family that first passes the repository-wide RB anti-retest audit and demonstrates a genuinely different target, mechanism, data source, or evidence regime rather than a post-result V1 repair.
