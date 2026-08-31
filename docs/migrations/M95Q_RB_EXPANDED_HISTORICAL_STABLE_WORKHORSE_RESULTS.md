# M95Q — Expanded Historical Stable-Workhorse Backtest Reconstruction Results

## Status

**Execution success. Expanded exact historical panel is ready for the next research migration.**

M95Q was a reconstruction/source-comparability migration, not a new model candidate. It made no production change, used no sportsbook input, and performed no feature or coefficient search.

## Authoritative run

- Workflow: `M95Q RB Expanded Historical Stable Workhorse`
- Run: **`33450395426`**
- Run number: **7**
- Downstream results job: **`99680228542`**
- Branch: `research-rb-m95q-expanded-historical-stable-workhorse`
- Tested SHA: **`92bd39f624fca6ab79f328bd856c11197d4894e0`**
- Started: `2026-08-31T23:22:18Z`
- Completed: `2026-08-31T23:32:57Z`
- Execution: **success**
- Artifact: `migration-95q-rb-expanded-historical-stable-workhorse`
- Artifact ID: **`9779790912`**
- Artifact SHA256: **`ed2ae370f9832ac8509b8a85a7359334c66329eb57c726818cdbe8776a227062`**
- Artifact files: 13
- Disposition: **`M95Q_EXPANDED_PANEL_READY`**

## Mechanical repair history

The M95Q reconstruction initially exposed historical-calendar mechanics, not model failures.

1. Pre-2021 NFL regular seasons ended at Week 17, while the first generalized M91 rebuild requested Weeks 1-18 for every season.
2. nflverse weekly player data can expose postseason observations under Week 18 for pre-2021 seasons; those rows cannot be joined to the REG-only historical schedule.
3. The repair was deliberately mechanical:
   - target seasons 2019-2020 rebuild Weeks 1-17;
   - target seasons 2021+ rebuild Weeks 1-18;
   - `historical_player_logs_m95q.py` derives each season's maximum regular-season week from the already-built REG-only schedule and drops provider observations outside that season-specific calendar before opponent attachment.
4. No candidate model logic, feature set, calibration family, stable-workhorse rule, comparability gate, or production logic was changed.

Run #7 confirmed that all six M91 reconstruction matrix jobs (2019-2024) completed successfully.

## 2024 M91-universe parity — perfect

The reconstructed 2024 M91 RB universe matched the authoritative frozen M91 universe exactly:

| Metric | Result |
|---|---:|
| Exact authoritative RB rows | 1,394 |
| Rebuilt RB rows | 1,394 |
| Overlap | 1,394 |
| Overlap rate | **100.000%** |
| Carry-truth MAE | **0.000000** |
| Exact-only rows | 0 |
| Rebuild-only rows | 0 |
| M91 universe parity gate | **PASS** |

This is the strongest mechanical control in M95Q: the generalized historical M91 rotation reproduces the canonical 2024 player-week population and carry truth exactly.

## Downstream 2024 M95F/M95G parity — pass

The generalized role/tail reconstruction was compared with authoritative frozen M95G 2024 W13-18.

| Metric | Result | Precommitted gate |
|---|---:|---:|
| Exact rows | 479 | — |
| Reconstructed rows | 479 | — |
| Overlap rows | 466 | — |
| Overlap vs exact | **97.2860%** | >=95% |
| Workhorse-role agreement | **100.000%** | >=98% |
| Stable-workhorse mask agreement | **100.000%** | >=95% |
| M95F 20+ probability correlation | **0.962355** | >=0.90 |
| M95F 20+ probability MAE | **0.017481** | <=0.05 |
| M95F 25+ probability correlation | **0.988349** | audit |
| M95F 25+ probability MAE | **0.002663** | audit |
| Exact-only rows | 13 | — |
| Reconstructed-only rows | 13 | — |
| Downstream parity gate | **PASS** | PASS |

The 13/13 nonoverlap is not being hidden or waived. The precommitted row-overlap requirement was >=95%, the exact M91 universe itself is 100% reproduced, and on the downstream overlap the role and stable masks agree exactly while tail probabilities show strong parity.

## New comparable historical stable-workhorse seasons

M95Q attempted target seasons **2020, 2021, 2022**. All three passed every comparability requirement.

| Season | Roster join | Raw 20+ features | Feature ratio vs 2024 | Stable n | Comparable |
|---|---:|---:|---:|---:|---:|
| 2020 | **100%** | 43 | **1.00** | 44 | **YES** |
| 2021 | **100%** | 43 | **1.00** | 61 | **YES** |
| 2022 | **100%** | 43 | **1.00** | 72 | **YES** |

Disposition counts:

- new exact years attempted: **3**
- new exact years comparable: **3**
- 2024 downstream parity pass: **1**
- 2024 M91-universe parity pass: **1**

## Historical M95F stable-workhorse baseline

These are baseline diagnostics, not a newly selected candidate.

### 2020 W13-17

- stable observations: **44**
- unique players: **19**
- actual 20+ rate: **27.27%** (12 events)
- actual 25+ rate: **11.36%** (5 events)
- mean carries: **16.16**
- M95F 20+ mean probability: **25.15%**
- M95F 20+ AUC: **0.776042**
- M95F 20+ Brier: **0.156351**
- 20+ calibration gap, actual minus predicted: **+2.13 pp**
- M95F 25+ AUC: **0.702564**
- M95F 25+ Brier: **0.087603**

### 2021 W13-18

- stable observations: **61**
- unique players: **23**
- actual 20+ rate: **36.07%** (22 events)
- actual 25+ rate: **13.11%** (8 events)
- mean carries: **16.87**
- M95F 20+ mean probability: **31.72%**
- M95F 20+ AUC: **0.497669**
- M95F 20+ Brier: **0.259780**
- 20+ calibration gap: **+4.34 pp**
- M95F 25+ AUC: **0.679245**
- M95F 25+ Brier: **0.110340**

### 2022 W13-18

- stable observations: **72**
- unique players: **26**
- actual 20+ rate: **29.17%** (21 events)
- actual 25+ rate: **2.78%** (2 events)
- mean carries: **15.49**
- M95F 20+ mean probability: **26.71%**
- M95F 20+ AUC: **0.690009**
- M95F 20+ Brier: **0.188409**
- 20+ calibration gap: **+2.46 pp**
- M95F 25+ AUC: **0.971429**
- M95F 25+ Brier: **0.026677**

The 2022 25+ AUC must be treated cautiously because there were only **2** positive 25+ events.

### 2024 parity/control W13-18

- stable observations: **79**
- unique players: **26**
- actual 20+ rate: **34.18%** (27 events)
- actual 25+ rate: **15.19%** (12 events)
- mean carries: **16.54**
- M95F 20+ mean probability: **29.03%**
- M95F 20+ AUC: **0.701567**
- M95F 20+ Brier: **0.199431**
- 20+ calibration gap: **+5.14 pp**
- M95F 25+ AUC: **0.568408**
- M95F 25+ Brier: **0.133166**

## Scientific interpretation

M95Q answers its primary question **yes**: exact, leakage-safe stable-workhorse temporal depth can be extended backward beyond 2023-2025 without replacing the M95F/M95G semantics with the broad M95P lead-RB proxy.

The important result is not that one historical season has a spectacular AUC. The important result is that we now have **three additional source-comparable exact seasons** with a validated 2024 reconstruction control. That gives the next candidate materially more temporal diversity.

The baseline diagnostics also reinforce the prior conclusion that tail behavior is not stationary:

- M95F 20+ ranking is strong in 2020 (`.776`), nearly random in 2021 (`.498`), and moderate in 2022/2024 (`.690` / `.702`).
- 20+ calibration gaps vary even within these newly reconstructed seasons.
- 25+ remains rare enough that single-season AUCs can be misleading; 2022 is the clearest example.

Therefore the expanded panel should be used to test whether a **pregame population/workload prior** can adapt the stable-workhorse tail probability scale or expert weighting across environments. It should not be used to tune one universal historical-ceiling coefficient until the old failures disappear.

## What M95Q does not prove

- It does not rescue or promote M95K.
- It does not validate M95O's fixed gate.
- It does not establish a production dynamic-prior formula.
- It does not make 25+ a sufficiently dense per-season target for unrestricted tuning.
- It does not provide new pristine confirmation data; these reconstructed seasons become research/development data once used.

## Next migration

**M95R — Expanded-Panel Dynamic Population-Prior Candidate**

Precommit the dynamic-prior/gating hypothesis on the now-expanded exact panel before fitting. The candidate should use only pregame workload-population signals, preserve M94C central carries, benchmark against M95F, preserve M95I vacancy separation, and be evaluated with strict temporal rotations rather than a pooled random split.

No sportsbook input. No production change. No retuning M95K against opened 2023 labels.
