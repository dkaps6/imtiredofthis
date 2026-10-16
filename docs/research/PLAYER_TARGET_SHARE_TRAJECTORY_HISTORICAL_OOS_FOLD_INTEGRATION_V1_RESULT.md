# Player Target Share Trajectory Historical OOS-Fold Integration V1 — RESULT

Date: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

## Frozen disposition

`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED`

Research-only. **No automatic production promotion.**  
The prospective 2026 Week-5+ confirmation contract is unchanged and remains
outcome-blind at this checkpoint.

## Exact successful authority

Final successful workflow:
- run: `37783801061`
- head: `4566fb9d1c6f2523487873d16b9ff76470b07b50`
- workflow conclusion: **SUCCESS**
- focused mechanics tests: SUCCESS
- immutable artifact verification: SUCCESS
- integration: SUCCESS
- result certification: SUCCESS
- strict repository audit: SUCCESS

Artifact:
- id: `11552284671`
- name: `player-trajectory-oos-fold-integration-v1`
- digest:
  `sha256:6d4e9a2a87fc2b364a847e6328d201f42fa4bf63b10f87416001c4fec8de03d2`
- retention expiry: 2027-01-06

Recovered OOS parent authority:
- run `37782189538` — SUCCESS
- artifact `11552447166`
- digest
  `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- status `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`

Trajectory feature authority:
- run `37638269235`
- artifact `11490707699`
- digest
  `sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`

No fresh historical provider rebuild was used.
No sportsbook inputs were used.
No paid OddsAPI call was made.
No 2026 outcomes were read.
No parameter was fitted.
No threshold/formula/window search was performed.
Production was not changed.

## Frozen primary population

Primary qualification population:
- season: **2024**
- weeks: **5+**
- WR parent: exact WR-R15 OOS fold, train 2023 -> test 2024
- TE parent: exact TE-R5P OOS test-2024 fold
- rows: **2,463**
  - WR: **1,619**
  - TE: **844**
- distinct players: **328**
  - WR: **213**
  - TE: **115**
- rooms:
  - WR: **416**
  - TE: **412**

All parent rows were retained. Missing exact trajectory state meant zero delta /
no change; rows were not outcome-filtered.

Secondary disclosure population:
- TE 2025 Weeks 5+
- 865 rows / 123 players / 412 rooms
- secondary only; it cannot rescue the joint 2024 primary result.

## Exact mechanism

The already-frozen no-fit rule was applied:

`weight_i = baseline_i * exp(trajectory_delta_i)`

then renormalized inside the protected room.

WR:
- exact original WR-R15 anchor semantics were preserved:
  max baseline entitlement row per event/team remains fixed;
- only the remaining WR pool is redistributed;
- total WR target mass is unchanged.

TE:
- the complete TE room is redistributed;
- total TE target mass is unchanged.

Receiving efficiency was held fixed:
shadow receiving yards = shadow targets × baseline yards per target.

Maximum conservation gaps:
- frozen WR anchor: **0.0**
- protected allocation pool:
  **1.7763568394002505e-15**
- room pool:
  **3.552713678800501e-15**

All are inside the frozen `1e-12` gate.

## Primary 2024 result

### Pooled WR + TE

Target MAE:
- baseline: **1.95101097**
- trajectory: **1.94604884**
- improvement: **0.00496214 targets**
- relative improvement: **0.254%**

Receiving-yards MAE:
- baseline: **20.56999067**
- trajectory: **20.51221057**
- improvement: **0.05778009 yards**
- relative improvement: **0.281%**

Within-room target-share MAE:
- baseline: **0.06479171**
- trajectory: **0.06446356**
- improvement: **0.00032815**
- relative improvement: **0.506%**

Target closer counts:
- trajectory closer: **983**
- baseline closer: **951**
- ties: **529**

### WR

Target MAE:
- **2.05188019 -> 2.04513661**
- improvement: **0.00674358**
- relative improvement: **0.329%**

Receiving-yards MAE:
- **22.57853699 -> 22.49978762**
- improvement: **0.07874937 yards**
- relative improvement: **0.349%**

Within-WR-room share MAE:
- **0.06879709 -> 0.06826454**
- relative improvement: **0.774%**

Target closer counts:
- trajectory 618 / baseline 569 / ties 432.

### TE

Target MAE:
- **1.75751896 -> 1.75597407**
- improvement: **0.00154489**
- relative improvement: **0.088%**

Receiving-yards MAE:
- **16.71710382 -> 16.69954796**
- improvement: **0.01755586 yards**
- relative improvement: **0.105%**

Within-TE-room share MAE:
- **0.05710839 -> 0.05717234**
- slightly worse by **0.112%**.

This TE room-share miss does **not** violate the frozen V1 disposition gate,
which was pooled room-share non-worsening plus separate TE target and receiving
yard improvement. Do not rewrite that gate after seeing results.

## Frozen bootstrap

Player-cluster bootstrap on primary target absolute-error improvement:
- clusters: **328**
- reps: **5,000**
- seed: `20261008`
- mean target-AE improvement: **0.00496214**
- `P(improvement > 0) = 0.9466`
- 95% percentile interval:
  **[-0.00108892, +0.01095625]**

The frozen acceptance gate was >=0.80 and passed.

The interval crosses zero; therefore this should be described as a **small
supported effect**, not a large or definitive effect.

## Secondary TE 2025 replication

Target MAE:
- **1.58314912 -> 1.57697072**
- improvement: **0.00617840**
- relative improvement: **0.390%**

Receiving-yards MAE:
- **15.86293984 -> 15.86065617**
- improvement: **0.00228368 yards**
- relative improvement: **0.014%**

Within-TE-room share MAE:
- **0.11051559 -> 0.10924267**
- relative improvement: **1.152%**

This is supportive secondary evidence only.

## Frozen gate outcomes

All predeclared primary disposition gates passed:

- 2024 pooled target MAE improves: PASS
- 2024 WR target MAE improves: PASS
- 2024 TE target MAE improves: PASS
- pooled receiving-yards MAE non-worse: PASS
- WR receiving-yards MAE non-worse: PASS
- TE receiving-yards MAE non-worse: PASS
- bootstrap probability >=0.80: PASS
- pooled room-share MAE non-worse: PASS
- conservation: PASS

Therefore:

`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED`

## Mechanical failure lineage

No failed run changed the science.

- `37783235618`: stopped before scoring because the initial historical wrapper
  incorrectly treated descriptive `wr_rank == 1` as WR-R15's anchor authority.
  Existing frozen WR-R15 code proved the true authority is maximum baseline
  entitlement per room. The plan was amended **before first scoring**.
- `37783523949`: first valid scoring run; scientific result/certification
  passed, but strict repo audit lacked repository `PYTHONPATH`.
- `37783697399`: reproduced the same scientific result; repo audit then
  correctly reported missing PyYAML from the focused environment.
- `37783801061`: exact same science with packaging dependencies repaired;
  every step including strict repo audit and artifact upload passed.

Do not interpret these mechanical failures as failed trajectory experiments.

## Interpretation

This is a real positive result, but the magnitude is modest.

The key scientific point is not that the trajectory transform suddenly solves
WR/TE opportunity allocation. It does not.

It shows that:
1. a strictly-prior individual role-change signal survives exact OOS specialist
   parents;
2. the no-fit frozen transformation improves target allocation and final
   receiving-yards MAE without creating target mass;
3. the gain is directionally consistent for WR and TE target MAE;
4. TE 2025 provides secondary replication;
5. the prospective confirmation lock remains necessary because the primary
   effect size is small and the bootstrap interval includes zero.

This supports keeping Target Share Trajectory V1 alive as a **qualified
individual-opportunity mechanism**, not promoting it automatically.

## Next-action boundary

Do not:
- relax the four-prior-same-season-game rule;
- retune the exponential coefficient;
- add a threshold/cap from these outcomes;
- rescue TE room-share behavior with a post-hoc subgroup;
- grade 2026 Week 5 prematurely;
- promote to production from this historical test alone.

The next cross-position research priority remains the unresolved opportunity
seams:
- QB team pass-attempt / starter / team-volume state;
- RB prospective carry/snap allocation remains sealed under its existing lock;
- WR/TE trajectory remains sealed prospectively while historical support is now
  established.

