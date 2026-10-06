# Football Matchup Transmission V1 — Candidate Parity Repair

Status: **FROZEN BEFORE REPAIRED CANDIDATE SCORING — RESEARCH ONLY**

Parent candidate contract:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATE_CONTRACT.md`

## Why this repair exists

Candidate run `37527077327` correctly failed closed before scoring because an independently rebuilt 2024/2025 football-only baseline did not numerically match the frozen right-tail authority.

Observed fail-closed evidence:

- run: `37527077327`
- candidate: `FMT-RB1`
- maximum baseline gap: `18.893955186846213` yards
- scoring never began
- candidate coefficients were never fit
- no production state changed
- no sportsbook data was used

The repository projection code shared by the frozen right-tail branch and the current candidate branch is unchanged; only research-layer files differ between those snapshots. The exact cause of the regenerated historical-input drift is therefore not assumed or hand-waved. It may reflect historical source revision or reconstruction-environment drift. The repair must not tune around it.

## Frozen authority repair

For the repaired score, use direct frozen authorities wherever an exact parent authority already exists.

### 2022 / 2023

Use the preserved leakage-safe candidate-preparation artifact from run `37527077327`:

- artifact: `11444106835`
- name: `football-matchup-integration-candidate-inputs-37527077327`
- digest: `sha256:6e2efdd9649bfa8c3043b74b07866cb08364409d3a7b2e72340d6fcca2957ec8`

Use only:

- 2022 baseline projection rows for coefficient training;
- 2023 baseline projection rows for primary confirmation;
- 2021-2025 historical player metadata as identity/position support;
- 2022/2023 reconstructed M89/M90-corrected team context for matchup features.

### 2024 / 2025 baseline means

Do **not** use the independently rebuilt 2024/2025 means from run `37527077327`.

Use the exact frozen right-tail baseline authority directly:

- run: `37493425352`
- artifact: `11427113291`
- digest: `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`
- file: `distribution_right_tail_asymmetry_detail.csv`
- baseline column: `final_mean`

The repaired scorer must prove one-to-one identity coverage for the three candidate cohorts before using these rows.

### 2024 / 2025 matchup features

Do **not** reconstruct the discovery-era 2024/2025 candidate features again.

Use the exact frozen Phase B/C feature authority directly:

- run: `37514137803`
- artifact: `11436786668`
- digest: `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- file: `football_matchup_phase_bc_team_features.csv`

Use only the already-frozen candidate features:

- `def_pass_rate_faced`
- `off_true_proe`
- `def_pass_success_allowed`
- their frozen weekly z-scores

## What does not change

The candidate science remains exactly frozen:

- candidates: `FMT-RB1`, `FMT-WR1`, `FMT-TE1` only;
- coefficient training: 2022 Weeks 2-18 only;
- coefficient form: zero-intercept OLS on `actual - baseline`;
- primary confirmation: 2023 Weeks 2-18;
- secondary consistency: 2024 and 2025 with the unchanged 2022 coefficient;
- all original pass/fail gates remain unchanged;
- no combined candidate;
- no thresholds or post-hoc cohort carving;
- no target-game usage;
- no sportsbook features;
- no production promotion.

## Required reporting

The repaired result must explicitly report:

- `secondary_baseline_method = DIRECT_FROZEN_RIGHT_TAIL_AUTHORITY`;
- `secondary_feature_method = DIRECT_FROZEN_PHASE_BC_FEATURE_AUTHORITY`;
- the independent rebuild drift audit separately;
- `sportsbook_inputs_used = 0`;
- `production_changed = false`.

Direct frozen authority use is not described as a successful independent rebuild. The failed independent parity attempt remains part of the evidence trail.
