# PLAYER SHARE INPUT COVERAGE AND RESIDUAL AUDIT V1 — RESULT

Date: 2026-10-07  
Branch: `research-player-share-input-coverage-residual-audit-v1`  
Successful head: `88f99fac0d2aafcd9d8ce5308fa187bf1fb707fd`  
GitHub Actions run: `37695773433` — **SUCCESS**  
Artifact: `11514753700`  
Artifact digest: `sha256:55cfe73281b54a0a776c8c8491b1bcde5ba52db36e683d78cb9b177a6d012db6`

## Disposition

`FOCAL_PLAYER_SHARE_COMPRESSION_CONFIRMED__BAYES_PRIMARY_COMPRESSION_STAGE__WR_TE_SPECIALISTS_PARTIAL_REPAIR__RB_TARGET_SHARE_UNCOVERED`

This audit is diagnostic only. Production changed: **false**.

The corrected ACT-only Weeks 1-4 reconstruction shows:

1. focal player misses generally **already have player-specific prior/current evidence**;
2. the largest systematic focal compression occurs when individual history is
   re-shrunk through the existing Bayesian share posterior;
3. football rules generally preserve that Bayesian share rather than creating
   the compression;
4. WR-R15 and TE-R5P **improve** target-share accuracy and partially repair the
   compression;
5. RB carries and RB targets have no analogous downstream specialist repair in
   this historical stack;
6. the already-frozen Week-5 RB rushing allocation shadow directly covers the
   carry-share seam;
7. general RB receiving target share remains the only uncovered skill-position
   share mechanism.

No threshold/carveout is authorized from the realized high-workload bins.

## Scientific boundary

Corrected parent authority:
- decomposition run `37694474836` — SUCCESS
- artifact `11515041385`
- corrected receiver denominator = actual team **dropback-side** opportunity
  volume, not official pass attempts.

Share audit:
- rows: **1,978**
- Weeks: 1-4
- no sportsbook input
- no paid OddsAPI
- parameters fit: **0**
- automatic promotion: **false**
- Target Share Trajectory rows applied in W1-4: **0**
- Week-5 RB room shadow rows applied in W1-4: **0**
- outcomes loaded only after pregame stage trace freeze: **true**
- final allocator vs corrected parent max share gap: <= `1e-10`
- strict repository audit: **PASS**

## RB carries — Bayes is the compression point

Paired rows with raw player history available: **396**.

### All paired rows
- raw-history share MAE: **0.12015**
- Bayesian share MAE: **0.16586**
- rules/final allocator MAE: **0.16586**

### Active paired rows
- raw MAE: **0.13992**
- Bayes/final MAE: **0.17523**
- raw bias: **-0.08941**
- Bayes/final bias: **-0.12364**

### Frozen 15+ carry diagnostic
Rows: **78**

Mean actual carry share: **0.69916**

- raw predicted share: **0.48195**
  - bias **-0.21721**
  - MAE **0.22961**
- Bayes/final predicted share: **0.36460**
  - bias **-0.33456**
  - MAE **0.33667**

Bayesian shrinkage therefore removes roughly **11.7 percentage points** of
mean predicted share from these focal backs versus raw player history.

Evidence is not missing:
- 60 / 78 high-workload rows are `prior+current`;
- the remaining 18 are `prior_only`.

Rules and the final top-five allocator do not materially change the Bayesian
share on these rows.

### Interpretation

The carry-share mechanism is real, but do **not** create a new retrospective
RB rushing repair:
- M96E retrospective rushing stop remains binding;
- the separately frozen Week-5 RB player-state allocation shadow already
  targets this exact carry-share mechanism prospectively with carry + snap state.

Disposition:

`RB_CARRY_SHARE_COMPRESSION_CONFIRMED__COVERED_BY_EXISTING_PROSPECTIVE_SHADOW`

## RB targets — uncovered receiving-share seam

Paired rows with raw history available: **396**.

### All paired rows
- raw target-share MAE: **0.03745**
- Bayes: **0.03629**
- rules/final: **0.03637**

So a wholesale raw-history replacement is **not** justified; Bayes slightly
improves average paired-row MAE.

### Active paired rows
- raw MAE: **0.04372**
- Bayes: **0.03656**
- final: **0.03647**

Again, no authorization exists to simply bypass Bayes globally.

### Frozen 9+ target diagnostic
Only **6 rows**; descriptive only.

Mean actual target share: **0.23059**

- raw predicted: **0.13154**
  - bias/MAE **-0.09905 / 0.09905**
- Bayes/final predicted: **0.07207**
  - bias/MAE **-0.15852 / 0.15852**

Four of six rows already have `prior+current` evidence and two have
`prior_only`.

The same focal compression direction is present, but the high-target sample is
small and outcome-defined. It cannot support a threshold or a focal-only
exception.

### Interpretation

This is the important uncovered seam:

- prior work already established RB targets as a major receiving-yard
  bottleneck;
- Current-Season State Persistence V1 independently showed RB target-share state
  persistence;
- R23-R27D generic RB receiving mean/efficiency lanes are closed;
- R26 is vacancy-gated receiving opportunity authority, not a general
  current-role target-share mechanism;
- the Week-5 RB player-state allocation shadow explicitly does **not** alter
  receiving.

Disposition:

`RB_TARGET_SHARE_RESIDUAL_CONFIRMED__NO_GLOBAL_RAW_REPLACEMENT__PROSPECTIVE_STATE_MECHANISM_MAY_BE_JUSTIFIED`

Any next RB receiving-share work must remain opportunity-only and prospective;
it may not reopen receiving-yard YPT/YPR/YAC mean science.

## WR targets — specialist repairs, trajectory shadow remains the next authority

Paired raw-history rows: **598**.

### All paired
- raw MAE: **0.05755**
- Bayes: **0.06185**
- rules: **0.06183**
- final WR stack: **0.05404**

Thus:
- Bayes worsens the paired raw-history baseline;
- M38 + WR-R15 more than recover the average loss and finish better than raw.

Full 669-row specialist comparison:
- rules MAE: **0.06372**
- post-M38: **0.05784**
- final: **0.05423**
- rules -> final improvement: **0.00949**

WR1 anchor:
- rules MAE **0.07836**
- final **0.06890**

WR2+:
- rules **0.06026**
- post-M38 **0.05522**
- final **0.05076**

### Frozen 9+ target diagnostic
Rows: **71**
Mean actual share: **0.26644**

- raw: predicted **0.19429**, MAE **0.08896**
- Bayes: **0.14109**, MAE **0.12535**
- rules: **0.14082**, MAE **0.12562**
- final specialist: **0.17540**, MAE **0.09314**

The specialist repairs most of the Bayes-induced focal compression but does not
fully return to the raw-history focal MAE.

Evidence is not absent:
- 57 / 71 are `prior+current`;
- among specialist-traced high rows, strong same-team history still retains
  meaningful negative bias.

### Interpretation

Do not open another WR target-share model.

The historically CONFIRMED Target Share Trajectory V1 and its immutable Week-5
prospective shadow directly target evolving individual target-share state.

Disposition:

`WR_SHARE_COMPRESSION_CONFIRMED__SPECIALIST_HELPFUL__TRAJECTORY_SHADOW_ALREADY_COVERS_NEXT_MECHANISM`

## TE targets — specialist helps, but focal compression remains

Paired raw-history rows: **362**.

### All paired
- raw MAE: **0.04589**
- Bayes: **0.04937**
- rules: **0.04934**
- final TE stack: **0.04571**

Full 437-row specialist comparison:
- rules / pre-TE-R5P MAE: **0.05333**
- final TE-R5P: **0.04843**
- improvement: **0.00490**

### Frozen 9+ target diagnostic
Rows: **19**
Mean actual share: **0.26288**

- raw: predicted **0.17687**, MAE **0.10288**
- Bayes/rules: **0.11893**, MAE **0.14395**
- final TE-R5P: **0.13084**, MAE **0.13204**

17 / 19 high-workload rows have `prior+current` Bayesian evidence.

Same-team specialist evidence is also usually present:
- 17 / 19 have prior same-team evidence;
- substantial negative focal bias remains.

### Interpretation

TE-R5P is beneficial and is not the source of compression.
Do not refit or replace it.

Target Share Trajectory V1 remains the authorized prospective next-state layer.

Disposition:

`TE_SHARE_COMPRESSION_CONFIRMED__TE_R5P_HELPFUL__TRAJECTORY_SHADOW_ALREADY_COVERS_NEXT_MECHANISM`

## Cross-position interpretation

The dominant pattern is:

`INDIVIDUAL_HISTORY -> BAYESIAN_RESHRINKAGE -> FOCAL_SHARE_COMPRESSION`

but the correct action differs by position.

- RB carry share: existing Week-5 RB allocation shadow covers it prospectively.
- WR target share: existing Week-5 trajectory shadow covers evolving share.
- TE target share: existing Week-5 trajectory shadow covers evolving share.
- RB target share: no equivalent general prospective state shadow exists.

The audit does **not** authorize:
- generic Bayes strength retuning;
- a high-workload threshold;
- raw-history replacement;
- position-level mean correction;
- a WR/TE specialist refit;
- another retrospective RB rushing search.

## Next authorized lane

Before creating any RB target-share implementation, freeze a narrow prospective
contract that uses **no 2026 W1-4 outcome-selected threshold** and does not
backtest a post-hoc focal subgroup.

A scientifically defensible candidate class is a **Week-5 prospective RB
target-share state shadow** that:

- uses only strict-prior receiving opportunity information;
- preserves the RB/FB room target mass exactly;
- changes no YPT/YPR/catch-rate/receiving-yard efficiency;
- changes no team pass/dropback volume;
- leaves R22 tail authority unchanged;
- leaves R26 vacancy behavior intact or composes with it under an explicit
  ordering contract;
- fits no coefficient from W1-4 outcomes;
- is graded only prospectively after lock.

Any formula must be frozen before Week-5 outcomes and must be justified by prior
independent RB target-share persistence evidence, not by selecting the six
realized 9+ target rows.

QB remains outside this lane; generic QB team-volume retuning remains closed
absent genuinely new pregame intent information.
