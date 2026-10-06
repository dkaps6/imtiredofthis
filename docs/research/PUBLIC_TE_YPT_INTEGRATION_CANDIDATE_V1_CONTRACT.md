# Public TE YPT Integration Candidate V1 — Frozen Contract

**STATUS: FROZEN BEFORE CANDIDATE SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Branch:
`research-public-te-ypt-integration-candidate-v1`

## 1. Parent evidence

Football Matchup Transmission Phase B/C authority:
- run `37514137803` — SUCCESS
- artifact `11436786668`
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- 2024/2025 discovery only
- zero sportsbook inputs
- zero candidate models fit

The specification:

`TE_REC x def_te_ypt_allowed`

replicated across both 2024 and 2025 in the expected positive direction, but was dispositioned:

`REPLICATED_DIAGNOSTIC_SOURCE_PARITY_BLOCKED`

because the historical official-stat field had not been proven deployable with the same semantics as the then-live Sharp field.

Public source-parity authority:
- run `37545711420` — SUCCESS
- artifact `11451030931`
- digest `sha256:7a52455276e99efa18b4b6b9e22fd248d50038e2da9f3cae4e4fddf3e4c329d3`
- result: `PUBLIC_POSITION_YPT_EXACT_PARITY_READY`
- exact historical identity coverage: 100%
- TE max absolute reproduction gap: `1.7763568394002505e-15`
- live 2026 Week-5 TE coverage: 30 / 30 scheduled defenses
- source weeks used live: Weeks 1-4 only
- chronology violations: 0

The public source is therefore a **new separately named field**. It does not overwrite or claim equality with Sharp.

## 2. Only candidate authorized

Candidate ID:

`PUB-TEY1`

Cohort:
- TE only
- market `rec_yards`
- regular-season Weeks 2-18

Feature:

`public_def_te_ypt_allowed`

Definition:
- free `nflreadpy.load_player_stats(..., summary_level="week")`;
- canonical weekly normalization;
- opponent from regular-season schedule;
- same-season source weeks strictly before target week;
- latest eight distinct prior source weeks;
- TE receiving yards allowed divided by TE targets faced.

Favorable direction:
- higher opponent public TE YPT allowed = more favorable TE receiving-yard environment.

Oriented feature:

`weakness_z = week_z(public_def_te_ypt_allowed)`

No WR or RB YPT candidate is authorized. Source readiness does not override their failed Phase B/C replication gate.

## 3. Functional form

For each eligible TE player-game:

`candidate_projection = baseline_projection + beta_train * weakness_z`

Rules:
1. zero-intercept fit;
2. one coefficient only;
3. no threshold, clipping, rank, TE1 carveout, target-share interaction, route interaction, opponent subgroup, or second feature;
4. no Sharp field enters the candidate;
5. no sportsbook field enters the candidate;
6. if `beta_train <= 0`, the sign gate fails and the candidate closes;
7. coefficient never changes after the 2022 fit;
8. production code remains untouched.

## 4. Temporal design

The source signal was selected using 2024/2025 residual evidence. Those seasons cannot be treated as independent confirmation.

### Training
Fit `beta_train` on:
- 2022 Weeks 2-18 only.

The public YPT feature uses only completed 2022 weeks strictly before each target week.

Baseline:
- corrected parent-analogous 2022 football-only historical baseline from run `37531333007`;
- corrected-input artifact `11444039829`;
- digest `sha256:fb7213ec0589dcd4e8f0a2bdcd25bcd25ccd3ce6d3e52395670b61f11411ec13`;
- no sportsbook data.

### Primary independent confirmation
Apply the frozen 2022 coefficient to:
- 2023 Weeks 2-18 only.

2023 was not used to select this TE YPT signal.

### Secondary consistency only
Without refitting, replay on:
- 2024 Weeks 2-18;
- 2025 Weeks 2-18.

For 2024/2025:
- baseline means come from the same frozen composite authority used by the corrected Football Matchup candidate run;
- public TE YPT values come from the frozen Phase B/C team-feature authority, whose public-stat identity was proven exact by run `37545711420`.

2024/2025 can never rescue a failed 2023 primary confirmation.

## 5. Training support and estimator

2022 training requires:
- >=200 TE receiving-yard player-games;
- >=50 distinct games.

Estimator:

`beta_train = sum(weakness_z * (actual - baseline)) / sum(weakness_z^2)`

No intercept, regularization, hyperparameter search, sign flip, clipping, or tuning.

## 6. Frozen evaluation metrics

For each season report:
- n
- distinct games
- distinct players
- MAE
- RMSE
- signed bias
- median absolute error
- Pearson correlation
- 75+ yard absolute-error count
- 100+ yard absolute-error count
- mean absolute candidate adjustment
- 95th percentile absolute adjustment
- maximum absolute adjustment
- residual Spearman correlation to `weakness_z`, before and after correction.

Paired uncertainty:
- 5,000 game-cluster bootstrap replicates
- seed `20261006`
- statistic `MAE_baseline - MAE_candidate`

## 7. Frozen PASS gate

`PUB_TEY1_CONFIRMED` requires ALL:

Training:
1. 2022 support passes;
2. `beta_train > 0`.

Primary 2023:
3. candidate MAE < baseline MAE;
4. candidate RMSE <= baseline RMSE;
5. game-cluster bootstrap probability of MAE improvement >= 0.80;
6. 75+ yard AE count does not increase;
7. 100+ yard AE count does not increase;
8. absolute residual Spearman relationship to `weakness_z` is smaller after correction.

Secondary 2024/2025:
9. coefficient remains exactly the 2022 coefficient;
10. MAE is non-worse in both 2024 and 2025;
11. pooled 2024+2025 MAE improves;
12. pooled 75+ and 100+ AE counts do not increase.

Integrity:
13. 2022/2023 feature chronology violations = 0;
14. 2024/2025 feature authority is the frozen Phase B/C field;
15. sportsbook inputs used = 0;
16. 2026 outcomes read = 0;
17. production changed = false.

Failure of any required gate yields:

`PUB_TEY1_CLOSED`

No rescue after scoring.

## 8. Explicit anti-rescue list

A failure may not be rescued by:
- TE1 only;
- high-target TEs;
- minimum target denominator;
- high/low YPT opponent buckets;
- 4-game / 6-game / season-long alternate windows;
- raw rather than z-scored feature;
- caps or nonlinear transforms;
- combining with `def_pass_success_allowed`;
- combining with Sharp;
- route/snap subgroups;
- side or sportsbook line;
- 2024/2025 refit;
- Week-4 or Week-5 2026 outcomes.

## 9. Promotion boundary

Even a PASS is research qualification only.

A confirmed candidate would require a separate integration/shadow contract that explicitly chooses the production seam and proves:
- ordering;
- receiving conservation;
- unaffected-market invariance;
- live public-source freshness;
- prospective 2026 evidence.

No paid OddsAPI pull is required or authorized.

Candidates scored before this contract: **0**  
2026 outcomes authorized: **0**  
Sportsbook inputs authorized: **0**  
Production mutations authorized: **0**
