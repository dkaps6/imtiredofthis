# Football Matchup Transmission V1 — Candidate Baseline Execution Note

Status: **EXECUTION CORRECTION ONLY — FROZEN SCIENCE UNCHANGED**

Parent contract:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATE_CONTRACT.md`

## Why this note exists

Candidate run `37527077327` correctly failed closed at the required 2024/2025
baseline-parity gate. The rebuilt 2024/2025 ensemble means drifted from the frozen
right-tail parent authority (FMT-RB1 maximum observed gap: 18.893955 yards).

The cause was not target/future leakage and was not a model-code change. The
candidate workflow supplied a shared 2021-2025 player-log history to every
walk-forward season. ML v2 and State v2 intentionally train on all rows before
the target cutoff, so this gave 2024/2025 extra old seasons that were not present
when the parent right-tail baseline was frozen.

The frozen right-tail run `37493425352` used one shared 2023-2025 log/history
window for its 2024/2025 targets. Therefore:
- 2024 could see 2023 pregame history;
- 2025 could see 2023-2024 pregame history;
- no later season was visible at either target.

## Corrected baseline execution

No candidate definition, coefficient rule, metric, threshold, gate, season, sign,
or source semantic changes.

### 2022/2023 training and primary confirmation

Build the canonical football-only baseline with the analogous shared 2021-2023
history window:

- 2022 target: only 2021 plus strict-prior 2022 rows are visible;
- 2023 target: 2021-2022 plus strict-prior 2023 rows are visible;
- 2,000 MC iterations;
- frozen `data/model_ensemble_weights.csv`;
- no sportsbook inputs.

This preserves the same historical-harness semantics as the parent authority.

### 2024/2025 secondary consistency

Do not reconstruct these means again.

Use the frozen right-tail authority directly:
- run `37493425352`
- artifact `11427113291`
- digest `sha256:5051111ebc20b79a334cfd1dc08ec8d8d52107e59acb023f7f5fefe4e5b771dd`
- `distribution_right_tail_asymmetry_detail.csv::final_mean`

The candidate scorer must still prove exact identity and numerical parity against
that authority before interpreting 2024/2025.

## Preserved non-science inputs

Run `37527077327` preserved reusable canonical inputs before the parity failure:

- artifact `11444106835`
- digest `sha256:6e2efdd9649bfa8c3043b74b07866cb08364409d3a7b2e72340d6fcca2957ec8`

The corrected runner may reuse its:
- strict historical player logs;
- corrected matchup TeamForm history;
- schedule history.

It must rebuild only the 2022/2023 football-only projection baselines with the
correct shared 2021-2023 scope.

## Governance

This note enforces the already-frozen baseline-authority requirement. It does
not authorize:
- a new candidate;
- coefficient retuning;
- threshold search;
- combined-candidate scoring;
- production changes;
- sportsbook data;
- reopening M95A/M95B, M56/M83, Rush Pool, Opportunity Authority, or Bayes tuning.
