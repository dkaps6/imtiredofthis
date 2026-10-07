# PLAYER OPPORTUNITY ALLOCATION AUDIT V1 — RESULT

Date: 2026-10-07  
Branch: `research-player-opportunity-allocation-audit-v1`  
Successful head: `a8ccf5968ae621721f03d86c692defdcdf1064d1`  
GitHub Actions run: `37686839645` — **SUCCESS**  
Artifact: `11512405401`  
Artifact digest: `sha256:b7075aa5536e183fd54415fd4c5225a3b45e84615ca6c40e9075236373bd95b0`

## Disposition

`PLAYER_OPPORTUNITY_COMPRESSION_REPLAY_SIGNAL_CONFIRMED__HISTORICAL_AVAILABILITY_PARITY_BLOCKER`

The individual-player opportunity audit is mechanically valid and confirms that
opportunity error is strongly transmitted into downstream player projection
error. However, a historical reconstruction mismatch was identified before any
new player-opportunity science may be proposed:

- `scripts/backtest/historical_inputs.py` admits both roster statuses
  `ACT` and `INA` into the historical pregame universe;
- the historical universe then drops roster status before modeling;
- canonical live production already has the separately promoted,
  clean-main-verified current player availability layer and excludes definitive
  unavailable players **before opportunity allocation**.

Therefore the zero-opportunity population from this historical replay cannot be
interpreted as evidence that current production presently prices definitively
inactive players. A bounded availability-parity diagnostic is required first.

No production change is authorized by this result.

## Mechanical validity

Run `37686839645` passed:

- exact simulation-no-op trace test;
- Weeks 1-4 leakage-safe football input build;
- exact frozen parent replay recovery;
- individual-player opportunity audit;
- audit-boundary certification;
- strict repository audit.

Boundary:
- paid OddsAPI used: **false**
- sportsbook inputs upstream: **false**
- parameters fit: **0**
- automatic promotion: **false**
- Week-5 RB room shadow rows in W1-4: **0**
- max MC allocation sampling deviation: **0.1072** (integrity gate <=0.25)
- rows: **2,308**
- unique player-weeks: **1,837**

## Player opportunity diagnostics

| Position | Opportunity | Rows | MAE | Bias | Pred/Actual Pearson | Actual-zero rate | Mean prediction at actual zero | Opportunity error -> yards error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QB | pass attempts | 128 | 9.069 | -0.880 | 0.332 | 7.81% | 26.14 | 0.814 |
| RB | carries | 471 | 4.343 | -0.547 | 0.694 | 35.88% | 3.93 | 0.827 |
| RB | targets | 471 | 1.366 | -0.159 | 0.515 | 45.86% | 1.16 | 0.787 |
| TE | targets | 490 | 1.801 | +0.383 | 0.636 | 45.31% | 1.91 | 0.788 |
| WR | targets | 748 | 2.113 | -0.050 | 0.617 | 32.49% | 2.15 | 0.743 |

For target-driven count errors:
- RB opportunity error -> receptions error Pearson: **0.923**
- TE: **0.907**
- WR: **0.872**

These are strong transmission relationships: when the model misses the
individual player's opportunities, the downstream yard/reception projection
usually misses in the same direction.

## Active/nonzero descriptive subset

Even when restricted to player-games with realized opportunity >0, the
reconstructed model is under the realized workload on average:

| Position | Opportunity | Rows | Pred mean | Actual mean | Bias | Opportunity error -> yards error |
|---|---|---:|---:|---:|---:|---:|
| QB | pass attempts | 118 | 28.71 | 31.88 | -3.17 | 0.696 |
| RB | carries | 302 | 6.34 | 9.39 | -3.05 | 0.805 |
| RB | targets | 255 | 1.57 | 2.85 | -1.28 | 0.719 |
| TE | targets | 268 | 2.65 | 3.53 | -0.88 | 0.722 |
| WR | targets | 505 | 3.50 | 4.61 | -1.11 | 0.667 |

But this is **not yet clean evidence of a current production allocation defect**.
Because the historical universe allocates team opportunity to `INA` rows too,
active players can be mechanically diluted by the historical reconstruction.
The availability-parity diagnostic must quantify that first.

## Workload-compression shape

The replay exhibits a consistent middle-compression pattern:

### QB pass attempts
- actual 1-20: bias **+12.17**
- actual 21-30: **+1.57**
- actual 31-40: **-5.91**
- actual 41+: **-17.62**
- actual zero: **+26.14**

### RB carries
- actual 1-3: **+2.12**
- actual 4-8: **-0.19**
- actual 9-14: **-4.04**
- actual 15+: **-10.29**
- actual zero: **+3.93**

### WR targets
- actual 1-2: **+0.89**
- actual 3-5: **-0.38**
- actual 6-8: **-2.52**
- actual 9+: **-5.32**
- actual zero: **+2.15**

### TE targets
- actual 1-2: **+0.68**
- actual 3-5: **-0.93**
- actual 6-8: **-3.60**
- actual 9+: **-7.02**
- actual zero: **+1.91**

### RB targets
- actual 1-2: **-0.10**
- actual 3-5: **-2.11**
- actual 6-8: **-4.48**
- actual 9+: **-8.25**
- actual zero: **+1.16**

This is diagnostically consistent with opportunity mass being spread too broadly
in the historical reconstruction. It does not by itself identify whether the
remaining production issue is participation, room allocation, or share
calibration.

## Critical reconciliation: current availability is already promoted

Canonical live production availability was merged at PR `#513`, merge commit
`f813f85ed814cc7c231a459e2301170171b8ed10`.

Clean-main no-odds Full Slate:
- run `34498365769`
- artifact `10160866044`
- conclusion **SUCCESS**
- availability before opportunity PASS
- PlayerForm from production-eligible current roles PASS
- promoted RB P3 current-role step PASS
- QB C2 state context PASS
- availability-aware current-output seams PASS
- strict audits PASS.

The locked production semantics explicitly require:
- availability before current player opportunity;
- definitive unavailable players excluded from active roles;
- sportsbook cannot resurrect unavailable players.

Therefore do **not** reopen current availability science and do **not** propose a
new inactive-player feature from this replay.

## Historical reconstruction mismatch

Historical replay builder:
`scripts/backtest/historical_inputs.py`

Currently:
`ALLOWED_ROSTER_STATUS = {"ACT", "INA"}`

Canonical grader independently defines:
- active statuses: `ACT`, `ACTIVE`
- inactive statuses: `INA`, `INACTIVE`, `DNP`

Thus `INA` has explicit inactive meaning in the same repository.

This mismatch is sufficient to block scientific interpretation of the
historical zero-state and may also dilute reconstructed active-player
opportunity.

## Next authorized step

Run a bounded, no-fit:

`HISTORICAL_AVAILABILITY_PARITY_DIAGNOSTIC_V1`

It must compare the exact same Weeks 1-4 opportunity audit under:

1. **baseline historical universe**: current replay semantics `ACT + INA`;
2. **explicit-status parity universe**: exact same nflverse weekly roster source,
   same schedule, same football stack, but explicit `INA` rows removed before
   context/opportunity allocation.

This is diagnostic reconstruction only. It is **not** new availability science
and cannot change production.

Required questions:
- how much modeled pass/carry/target opportunity was assigned to `INA` rows?
- how much do active-player opportunity MAE/bias/correlation change when explicit
  inactive rows are removed before allocation?
- how much of the middle-compression pattern survives?
- how much of the linked point-error relationship survives?
- which positions still have material active-player allocation error after this
  minimal availability-parity correction?

No threshold fitting, no market inputs, no science promotion, and no reopening
of the already-promoted availability hierarchy.
