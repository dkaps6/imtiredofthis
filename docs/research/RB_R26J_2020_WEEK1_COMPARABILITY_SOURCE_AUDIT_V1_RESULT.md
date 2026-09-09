# RB R26J — 2020 Week-1 Comparability Source Audit V1 Result

Status: COMPLETE — SOURCE REGIME DISTINCT FOR MECHANISM FOLLOW-UP
Date: 2026-09-09

## Canonical execution

- Branch: `research-rb-r26j-2020-week1-comparability-source-audit-v1`
- Frozen-plan commit: `ff7e2c535a2283c02e0bb1c4db19becb721e3de7`
- Initial launch head: `48f685523528f43c5453b97195c9622c8ad864f5`
- Initial run: `34370743209` — mechanical failure before science because the R26C exited-player source artifact did not expose `player_key` under the assumed name.
- Mechanical repair commit: `1dc136253bc0b7b27acb72014d7fbffb8f65900c`
- Valid run: `34374987828`
- Job: `102545404859`
- Artifact: `10113466373`
- Artifact digest: `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- Valid code SHA256: `e6b8d4d861de43075ededb5ee43975d4bdb4af3e4b7452cd88c022caf88244b5`
- Frozen plan SHA256: `10ba409484a880e55f734e3ad7e73261e4510c5842977d44fbc1f5c2dd448603`
- Implementation lock SHA256: `e5ca6529f7ec67dd8dade535ddeba21da19bd9aeb7782f9066e8dc8dbac4fc4c`
- Disposition: `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`

## Mechanical repair

The first run failed at exited-player aggregation because R26J attempted `exit_player_rows=("player_key", "count")` while the immutable R26C `r26c_exited_player_source_state.csv` did not guarantee an identity column named `player_key` after the source feature frame was materialized.

The repair changed only the row-count implementation to `groupby(keys).size()` and merged that count back into the same team-week aggregation. This preserves the exact frozen semantic quantity — number of exited-player source rows — while removing an unnecessary identity-column dependency.

No frozen feature definition, comparability threshold, cohort, source contract, disposition rule, R9 logic, or production file was changed.

## Integrity

All 10 frozen integrity gates passed.

- Parent artifact digests verified by workflow: PASS
- Week-1 vacancy population only: PASS
- Selected `actual_*` fields: `[]`
- Target-game outcome/participation features used: `0`
- Sportsbook inputs used: `0`
- Same-week historical depth used: `false`
- Canonical ACT/INA roster-source contract: PASS
- Room keys unique: PASS
- R26C vacancy reconstruction rate: `1.0`
- Protected production files clean: PASS
- Predictions regenerated: `false`
- R9 refit: `false`
- Production parameters changed: `false`

Population:
- 170 Week-1 vacancy rooms total
- 2020: 28
- 2021: 29
- 2022: 29
- 2023: 28
- 2024: 30
- 2025: 26

## Frozen comparability result

The frozen disposition required at least three independent structurally distinct pregame dimensions spanning at least two of Sections A–D. R26J found:

- **10 independent structurally distinct A–D dimensions**
- spanning **all 4 Sections A–D**

Therefore the source-only audit supports treating 2020 Week 1 as a materially different pregame RB-room/offseason-transition regime for the purpose of a follow-up mechanism study.

This result does **not** authorize excluding 2020 from qualification.

## Distinct 2020 pregame dimensions

### A — Room continuity / turnover structure

1. `current_room_n`
   - 2020 mean: `4.4642857143`
   - 2021–25 mean: `3.9986623721`
   - 2020 exceeded the entire 2021–25 seasonal-mean range (`3.8214` to `4.2414`).

2. `continuing_n`
   - 2020: `2.25`
   - 2021–25: `2.0026310471`
   - 2020 exceeded the entire later-season range.

3. `entrants_n`
   - 2020: `2.2142857143`
   - 2021–25: `1.9960313250`
   - 2020 exceeded the entire later-season range.

Interpretation: 2020 Week-1 vacancy rooms were larger and contained both more incumbents and more entrants. This is not simply a generic higher-churn story; the whole room was more populated.

### B — Entrant composition

4. `veteran_entry_n`
   - 2020: `0.8571428571`
   - 2021–25: `0.5598206391`
   - ~53.1% higher than the later-period mean and above the entire later-season range.

5. `veteran_entry_share`
   - 2020: `0.1869897959`
   - 2021–25: `0.1424184245`
   - ~31.3% higher and above the entire later-season range.

Interpretation: the 2020 Week-1 transition environment had materially more established-veteran entrants in vacancy rooms.

### C — Exited-role / strict-prior source state

6. `exit_history_coverage`
   - 2020: `0.9523809524`
   - 2021–25: `0.9118750789`
   - above the entire later-season range.

7. `sum_exit_last8_targets_pg`
   - 2020: `3.5864197531`
   - 2021–25: `3.3128236110`
   - above the entire later-season range.

Interpretation: 2020 did not look worse because exited-player history was missing. If anything, source history was more complete, and the exited backs collectively carried more recent receiving work.

### D — Returning-room / baseline model state

8. `baseline_room_hhi`
   - 2020: `0.2524742072`
   - 2021–25: `0.2834753601`
   - below the entire later-season range.

9. `baseline_top_room_share`
   - 2020: `0.3254747868`
   - 2021–25: `0.3550108871`
   - below the entire later-season range.

10. `incumbent_n`
    - same room-state distinction as continuing count: 2020 `2.25` vs 2021–25 `2.0026310471`, above the later-season range.

Interpretation: the production baseline entering 2020 Week 1 was **less concentrated** across a **larger incumbent set** than later Week-1 seasons. That is a plausible football mechanism for why a redistribution model may misallocate individual receiving work even when the RB-room total is reasonable.

## Important non-distinctions

The source audit also showed several things that do **not** explain 2020 by themselves:

- `r9_reliability` was exactly `1.0` in every season.
- R26 allocation-shift magnitude was not structurally unusual in 2020.
- balanced-turnover frequency was higher in 2020 (`0.50`) than the 2021–25 mean (`~0.407`) but did not satisfy the frozen structural-distinction rule.
- meaningful-exit count was not unusual.
- max prior exited-player RB-room share and max prior target rate were not unusual.
- entrant-state resolution and prior-depth coverage were not materially degraded.

This reinforces that the 2020 problem should not be reduced to one blanket balanced-turnover or source-quality guard.

## Scientific interpretation

R26J validates the hypothesis that 2020 Week 1 is not fully comparable to 2021–25 on the pregame football-state variables already available to R26/R26C.

The strongest coherent pattern is:

**larger Week-1 vacancy rooms + more continuing backs + more entrants, especially veteran entrants + more recently vacated receiving work + a flatter baseline allocation across incumbents.**

That combination is football-natural and can plausibly make within-room redistribution more difficult. It is consistent with the earlier R26F finding that 2020's problem was player-level allocation rather than room-total receiving volume.

However, R26J is deliberately source-only. It does not establish which of these dimensions caused the error and it cannot authorize a router merely from structural distinctness.

## Governance / what is and is not authorized

Authorized:
- a **frozen outcome/mechanism follow-up** that tests how the predeclared R26J structural dimensions relate to R26 allocation error, including whether the harmful 2020 pattern replicates in comparable states outside 2020.

Not authorized:
- excluding 2020 from historical qualification;
- changing the 2020 safety gate;
- prospective 2026 Week-1 shadow;
- production promotion;
- R9 refit or coefficient retuning;
- changing receiving-yard production means or R22;
- using sportsbook or target-game information upstream.

## Recommended next frontier

Freeze an **R26K-style outcome/mechanism atlas** before inspecting error slices. It should use only the R26J-predeclared dimensions/states, then evaluate R26-vs-baseline player-allocation error and replication outside 2020.

The purpose should be mechanism identification, not direct rescue. Only a replicated, football-coherent state should be allowed to authorize a later child design.
