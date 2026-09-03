# RB Final Qualification Results

## Authoritative execution

- Branch: `research-rb-final-qualification`
- Run: `33719205615`
- Job: `100534683157`
- SHA: `0584a87e24e011cbd791be89b9fd4caa967349fa`
- Artifact: `9879617371`
- Artifact name: `rb-final-qualification`
- Artifact SHA256: `14c0f9e4e0b9bed0798ba24c8443134b1949520c37e82c18db5fd8d9d5564d48`
- Workflow conclusion: `success`

The run used only the exact inherited STACK3 football casebook and STACK5 market casebook frozen in `RB_FINAL_QUALIFICATION_PLAN.md`. There was no fitting, feature search, model-family search, weight search, threshold optimization, or sportsbook input upstream.

## Integrity

All frozen identity checks passed:

- football rows: `1393`
- market rows: `899`
- football duplicate player-games: `0`
- market duplicate player-games: `0`
- P3 all-RB yard MAE: `19.949523978340356`
- P3 all-RB yard RMSE: `28.866519286368135`
- P3 exact-market MAE: `24.315798244183124`
- Vegas exact-market MAE: `23.701890989988875`
- P3 exact-market RMSE: `33.34291378702183`
- Vegas exact-market RMSE: `32.493543467503315`
- exact reconstructed P3 yards vs inherited STACK5 P3 parent max abs diff: `0.0`
- exact reconstructed P3 carries vs inherited STACK5 P3 parent max abs diff: `0.0`

This is a valid scientific result, not a source or integration failure.

## Overall football performance

### Carries

| Arm | MAE | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| M94C | 3.411003 | 4.590691 | -0.026222 | 0.742664 |
| Full stack | 3.482576 | 4.741740 | -0.862725 | 0.735396 |
| STACK2 enriched | 3.376055 | 4.536147 | -0.026222 | 0.749914 |
| **P3** | **3.357494** | **4.500432** | **-0.079500** | **0.753643** |

P3 is the strongest overall carry arm in this frozen comparison.

### Rushing yards

| Arm | MAE | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| M94C | 21.031150 | 29.861488 | +0.708378 | 0.601602 |
| Full stack | 20.424163 | 30.069806 | -5.537128 | 0.616847 |
| STACK2 enriched | 20.047102 | 28.990310 | -2.302435 | 0.626204 |
| **P3** | **19.949524** | **28.866519** | **-2.496944** | **0.631266** |

P3 improves M94C by:

- rushing-yard MAE: **+1.081626 yards**
- rushing-yard RMSE: **+0.994969 yards**
- rushing-yard correlation: **+0.029664**
- carry MAE: **+0.053509 carries**
- carry RMSE: **+0.090259 carries**

Late-season W13–18 rushing-yard MAE also improves:

- M94C: `20.707936`
- P3: `19.846754`
- gain: **+0.861183 yards**

## Depth-role performance

The exact casebook had `100%` finite depth-rank coverage.

P3 improves M94C rushing-yard MAE at every predeclared depth level:

- RB1: `25.620708 -> 24.155252` (**+1.465456**)
- RB2: `20.404111 -> 19.411194` (**+0.992917**)
- secondary depth 3+: `15.432903 -> 14.776903` (**+0.656000**)

P3 also improves carry MAE at all three depth levels:

- RB1: `3.782030 -> 3.744471`
- RB2: `3.409651 -> 3.388256`
- secondary: `2.900535 -> 2.787328`

Therefore the final production blocker is not a generic RB1/RB2/secondary allocation regression.

## Pregame diagnostic states

The retained pregame diagnostics do not show a simple population in which M94C should replace P3:

### M95F risk ON

- M94C yard MAE: `30.195360`
- P3 yard MAE: `28.327541`

### M95F risk OFF

- M94C yard MAE: `16.743575`
- P3 yard MAE: `16.029777`

### Vacancy ON

- M94C yard MAE: `22.752446`
- P3 yard MAE: `22.236478`

### Vacancy OFF

- M94C yard MAE: `20.874791`
- P3 yard MAE: `19.741782`

### M95I tail state ON

- M94C yard MAE: `36.482137`
- P3 yard MAE: `31.271360`

P3 therefore remains better than M94C inside the available pregame tail/risk states, despite the actual-outcome tail failure below. This is evidence against a simple pregame routing back to M94C.

## Frozen production gates

| Gate | Value | Pass |
|---|---:|---:|
| P3 yard MAE gain >= 0.50 | +1.081626 | PASS |
| P3 yard RMSE gain > 0 | +0.994969 | PASS |
| P3 yard corr delta >= -0.01 | +0.029664 | PASS |
| P3 carry MAE gain >= -0.10 | +0.053509 | PASS |
| P3 carry RMSE gain >= -0.10 | +0.090259 | PASS |
| P3 W13–18 yard MAE better | +0.861183 | PASS |
| Not both frozen tail slices worse | -9.392580 combined recovery | **FAIL** |

P3 passes **6 of 7** football gates.

## Exact blocker: high-end outcome compression

### Games with actual carries >=20

- n: `98`
- M94C yard MAE: `40.005137`
- P3 yard MAE: `43.741417`
- P3 is worse by **3.736280 yards**
- M94C bias: `-35.627337`
- P3 bias: `-41.588897`

At actual carries >=25:

- n: `24`
- M94C yard MAE: `49.310484`
- P3 yard MAE: `57.364680`

### Games with actual rushing yards >=100

- n: `95`
- M94C yard MAE: `66.439634`
- P3 yard MAE: `72.095934`
- P3 is worse by **5.656300 yards**
- both models underproject every member of this actual-outcome slice.

This is a ceiling/tail compression problem. It does not justify post-hoc routing by actual workload, which is unknowable pregame, and the already-retained M95F/M95I pregame states do not identify a simple M94C routing population. The previous M95T stop on retrospective carry-tail candidate retuning remains in force.

## Downstream Vegas benchmark

Exact 899-row market universe:

| Arm | MAE | RMSE | Bias | Corr | Median AE | P90 AE |
|---|---:|---:|---:|---:|---:|---:|
| **P3** | **24.315798** | **33.342914** | -4.888748 | 0.492092 | 18.569001 | 52.461266 |
| **Vegas** | **23.701891** | **32.493543** | -4.327030 | 0.529751 | 17.500000 | 50.000000 |

Aggregate MAE gap:

- Vegas advantage = **0.613907 yards**.

P3 is strictly closer to the realized outcome in `433 / 899 = 48.1646%` of rows; Vegas is closer in `466 / 899 = 51.8354%`. There were no absolute-error ties under the exact continuous P3 projections.

P3-vs-Vegas directional disagreement was correct in `53.3927%` of rows, but this is an exposed retrospective sample and is not a validated betting edge.

### Frozen disagreement bins

| Absolute P3-Vegas disagreement | n | P3 MAE | Vegas MAE | Vegas MAE - P3 MAE | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| <2.5 | 169 | 24.385488 | 24.396450 | **+0.010962** | 52.07% |
| 2.5–5 | 164 | 23.989945 | 23.661585 | -0.328359 | 50.00% |
| 5–7.5 | 158 | 22.133385 | 22.018987 | -0.114397 | 54.43% |
| 7.5–10 | 115 | 26.182128 | 25.834783 | -0.347345 | 53.04% |
| >=10 | 293 | 24.902337 | 23.394198 | **-1.508139** | 55.63% |

There is no retrospective disagreement bin that justifies declaring a validated market edge. The >=10-yard disagreement group is particularly important: directional sign accuracy is above 50%, but point-estimate MAE is materially worse than Vegas. Do not optimize or select a new betting threshold from these exposed results.

### Market time slices

- W1: P3 `23.709666`, Vegas `23.198113`
- W2–12: P3 `24.681676`, Vegas `23.894689`
- W13–18: P3 `23.756984`, Vegas `23.440000`

Vegas remains lower-MAE in every frozen time slice.

## Formal disposition

### Football status

**`FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED`**

P3 is the best overall football point model in the frozen comparison, but the precommitted tail gate failed. No waiver is granted. Production promotion is **not authorized** from this run.

### Market status

**`AGGREGATE_VEGAS_NOT_CLEARED`**

The football model still trails the exact downstream Vegas benchmark by approximately `0.614` rushing yards MAE on the 899-row 2025 market universe.

Market status does not change football qualification and sportsbook information remains downstream only.

## RB research disposition going forward

1. Preserve P3 as the RB **research champion / shadow candidate**.
2. Preserve M95F workload-tail and M95I vacancy/tail states as diagnostics; do not route the P3 point mean through them based on this exposed result.
3. Do not reopen STACK6 team-rush context slicing. `RB_STACK6_FINAL_STOPPING_EVIDENCE.md` remains authoritative.
4. Do not launch another retrospective carry-tail tuning family; the M95T stop remains authoritative.
5. The unresolved RB issue is now specifically **pregame ceiling calibration / high-end outcome compression**, not broad opportunity, depth allocation, or generic team-rush modeling.
6. Use the 2026 regular season as genuine forward/shadow evidence for P3 and its tail diagnostics, including exact downstream comparison to contemporaneous sportsbook lines.
7. Primary retrospective research can now move to WR without discarding RB: RB remains frozen in shadow evaluation and can be revisited only with genuinely new pregame information or forward evidence.
