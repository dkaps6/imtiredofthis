# WR Post-R19 Read-Priority Source Semantics Audit V1 — Result

**STATUS: `READ_PRIORITY_SEMANTIC_CONTRACT_BLOCKED`. NO WR OUTCOME SCORING. NO R20 AUTHORIZATION.**

## Why this result exists

The source/redundancy workflow `34911705733` mechanically passed its frozen audit and, under the then-frozen current nflreadr dictionary mapping (`0` = primary read, `1` = second read), returned `READ_PRIORITY_R20_PLAN_ELIGIBLE`.

That eligibility is **not operative** because a source-semantic contradiction was discovered immediately afterward, before any WR outcome was loaded or any R20 football plan was frozen.

## Canonical source-only audit lineage

- branch: `research-wr-read-priority-source-audit-v1`
- audit plan commit: `5c8094893c304833ca27cb620d883550ad33ddca`
- implementation commit: `a0abbbd87e5757f5a107b1aed50f67bf69c82798`
- workflow head: `22c0a3b52bdb98b101ebe9820266c3e03f4829d5`
- run: `34911705733`
- artifact: `10375035263`
- digest: `sha256:8b05f8677237fe8dce806d74bbb853d852c827a1732e735c88cdd0330b03046f`

The audit proved:
- exact WR-R15 authority lineage passed;
- no WR receiving-yard outcome/residual field was loaded;
- 2024 WR-R15 projection/outcome fields were not parsed/materialized;
- no sportsbook inputs;
- exact FTN -> PBP play joins were 100%;
- supported 2023 coverage was 1667/2076 = 80.30%.

Under the provisional `0 = PRIMARY_READ` normalization, redundancy was low:
- Spearman(primary-share, entitlement_tgt_share) = +0.147744;
- Spearman(primary-share, pred_targets) = +0.146480;
- R^2 from entitlement + pred_targets + WR1 = 0.007700.

Those redundancy numbers are source-descriptive only and cannot authorize R20 while category semantics are disputed.

## Semantic contradiction discovered after the run

### 1. nflverse FTN ingestion does not recode `read`

`nflverse/nflverse-ftn` `R/nflverse.R` sets:

`read_thrown = read`

with no value transformation. Therefore any 0/1/2 meaning must come from FTN's raw coding or downstream documentation, not an nflverse recode.

### 2. Original nflreadr issue #216 points to a different convention

Issue `nflverse/nflreadr#216`, opened in 2023 by an nflverse maintainer, recorded:

`read_thrown definitions: 1, 2, CHK check down, SD scramble drill, DES designed play (read 0, eg screens)`

This source predates the expanded 2026 dictionary text and does not describe `0` as primary / `1` as second.

### 3. The current `0 = first, 1 = second, 2 = third+` wording was added only in 2026 PR #319

Commit `b32d4340123ead6b67cf7508e5e04495f9bdd882` / PR `nflverse/nflreadr#319` expanded the data dictionary to state:
- `0` first/primary;
- `1` second;
- `2` third+;
- 2022 primary reads appear as NA.

The PR contains no supporting derivation or raw-source citation for the value mapping. It was accepted as a documentation improvement.

### 4. Receiver-target frequencies strongly conflict with the new wording

On exact official receiver-target events from the source-only artifact, the provisional normalization produced:
- 2022: `PRIMARY_READ` (2022 NA) 201 / 17,306 = 1.16%; `1`-coded events 9,233 = 53.35%;
- 2023: `0` 233 / 17,483 = 1.33%; `1` 9,172 = 52.46%;
- 2024: `0` 274 / 17,013 = 1.61%; `1` 9,639 = 56.66%.

A true first-read rate of only ~1-2% of targeted passes, with a second-read rate above 50%, is football-wise implausible and is much more consistent with an off-by-one/documentation mismatch than with the proposed semantics.

## Disposition

`READ_PRIORITY_SEMANTIC_CONTRACT_BLOCKED`

The mechanical/source join is excellent and the feature appears nonredundant under either broad coding family, but the exact football meaning of the numeric values is not trustworthy enough to preregister a predictive hypothesis yet.

## What would clear the block

At least one authoritative resolution is required before R20 can be frozen:
1. direct FTN documentation/API schema confirming the raw `read` key meanings; or
2. an nflverse maintainer/source clarification reconciling Issue #216 with PR #319; or
3. another primary source tied to the FTN charting feed that unambiguously identifies the numeric progression coding.

If the correct convention is `1 = first`, the source-only audit must be rerun with that mapping before any WR outcome is exposed. If the current `0 = first` convention is affirmatively confirmed despite the observed distribution, that confirmation and rationale must be recorded before R20.

## Stop rules

- Do not run WR receiving-yard outcomes under either mapping until semantics are resolved.
- Do not choose the mapping based on whichever one predicts outcomes better.
- Do not inspect 2024 WR-R15 outcomes.
- Do not reopen contested-ball; that lane remains closed.
- No production change, no paid Full Slate, no RB work.
