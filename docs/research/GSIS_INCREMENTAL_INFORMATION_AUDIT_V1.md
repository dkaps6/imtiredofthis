# GSIS incremental-information audit v1

**Status:** COMPLETE FOR LINEUP DETAIL AND FORMATION USAGE

**Date:** 2026-09-28 America/Indiana/Indianapolis

**Acquisition gate:** `GSIS_NOVELTY_EXISTING_SOURCE_AUDIT_V1.md`

## Decision

The bounded source-information audit confirms that both retained A-class GSIS
reports contain current role/personnel state that the live stack does not encode:

- **Lineup Detail remains class A** for exact 11-player co-occurrence and
  pass/rush exposure by exact lineup.
- **Formation Usage remains class A** for live personnel grouping by situation
  and personnel-conditioned play choice.

The result is narrower than a model promotion. It establishes new information,
not predictive value. No coefficient was fit, no Week-3 performance outcome was
used, and no production, Full Slate, entitlement, or model file changed.

## Inputs and boundary

The comparison used:

- the already archived private 2026 REG GSIS point-in-time snapshot, gzip SHA-256
  `779b1bd6c5d4dbd390c992d1e02e007edb80dac7d5e19b8b9045b161000c8a23`;
- preserved Week-3 pregame Full Slate artifact `10923570170`, digest
  `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`;
- frozen WR/TE 2026 strict-prior snap-source artifact `10699873027`, digest
  `sha256:dbd15f7f9d2eb34d2c28aec07fb71dc0dccb80e8f6d40f0165b92c91eee62ddd`.

The pregame artifact fixes the comparator side before Week 3 and contains the
current Ourlads/availability state, PlayerForm, TeamForm, and WR/TE/RB target
entitlement state. The snap artifact supplies the exact Week-3 strict-prior
WR/TE snap features built from completed Weeks 1-2.

The audit script reads only these GSIS fields:

| Report | Fields read |
|---|---|
| Lineup Detail | Lineup, Plays, Passing Plays, Rushing Plays |
| Formation Usage | Down, Yards to Go, # TEs, # WRs, Play Count, Passing Plays, Rushing Plays |

It does not read average gain, first downs, touchdowns, turnovers, EPA, grading,
market results, projections, or any other performance/efficiency outcome. The
GSIS report is still a cumulative season/phase table with no week filter, so its
usage counts are allowed here only to test source dimensionality. They cannot be
used to diagnose Week 3, time a change, or support a model result.

## Lineup Detail result

### Exact co-occurrence is not recoverable from the live stack

| Check | Sanitized result | Interpretation |
|---|---:|---|
| Valid offensive exact-lineup rows | 2,118 / 32 teams | The report contains a large joint-state table, not one marginal player value. |
| Median distinct offensive lineups per team | 64.5 | Every team used many exact combinations. |
| Duplicate exact-player sets | 0 | Rows represent distinct 11-player sets rather than repeated display variants. |
| Median share of plays in the top lineup | 11.4% | A single depth-chart starting unit does not represent observed deployment. |
| Median share in the top three lineups | 25.6% | Even the three most-used combinations leave most plays outside those units. |
| Median effective number of lineups | 25.8 | Lineup state is highly distributed. |
| Median play-weighted substitutions from the top lineup | 2.08 players | The average observed unit differs materially from the most-used unit. |

The current Ourlads/availability table, PlayerForm, snap traces and entitlement
traces are all player-marginal tables. None contains a lineup identifier,
co-player set, exact 11-player unit, or lineup-conditioned pass/rush count.
Marginal snap shares cannot uniquely reconstruct a joint co-occurrence matrix.

The identity bridge reinforces that conclusion:

| Existing live source comparison | Aggregate coverage |
|---|---:|
| GSIS offensive player-team identities | 690 |
| Matched to current Ourlads skill-role identities | 395 |
| Matched WR/TE identities | 251 |
| Matched WR/TE identities present in the strict-prior snap trace | 241 / 251 (96.0%) |
| Matched skill identities present in PlayerForm | 345 / 395 (87.3%) |
| Matched skill identities present in the target-entitlement trace | 359 / 395 (90.9%) |

The lower whole-lineup Ourlads match rate is expected because the production
Ourlads table is a skill-position role source and does not represent the entire
offensive line or defensive unit. High WR/TE marginal coverage does not change
the central result: those sources still cannot identify which players shared a
specific snap.

### Exact-lineup pass/rush exposure is additional descriptive information

Using only lineups with at least five classified pass/rush plays:

- 281 offensive lineup rows qualified;
- those rows covered a median 49.6% of each team's classified plays;
- the median within-team, play-weighted absolute deviation from the team's
  lineup-table pass rate was **17.0 percentage points**.

This proves that lineup identity partitions play choice beyond one team-level
pass-tendency value. It does not prove persistence or forecast value. The
remaining small lineup cells are too sparse to treat as stable without future
snapshots and shrinkage research, which is outside this audit.

### Stability, churn and replacement timing

The current snapshot supports a **concentration baseline**, not temporal churn:

- all 32 teams have multiple exact offensive lineups;
- all 32 have observed variants at least three player substitutions away from
  their most-used lineup;
- the snapshot has no week column and cannot say when a unit appeared.

Therefore:

- lineup concentration/dispersion is measurable now;
- week-to-week lineup stability and churn require at least two immutable weekly
  snapshots;
- replacement-lineup emergence cannot be timed from the cumulative table;
- a lineup containing a current depth-chart replacement is not proof of when or
  why the replacement occurred.

No replacement-emergence claim is made in this audit.

## Formation Usage result

### Personnel state is materially richer than TeamForm

| Check | Sanitized result | Interpretation |
|---|---:|---|
| Nonempty situation/personnel rows | 2,036 |
| Teams with nonempty rows | 31 |
| Source-empty team at capture | WAS | Preserve as source-empty; do not impute. |
| Median situation/personnel cells per team | 66 |
| Median distinct personnel groups per team | 6 |
| Median share of plays in the most-used group | 57.8% |
| Median effective personnel groups | 2.39 |

The preserved pregame TeamForm has complete, nondegenerate `proe` and
`pass_rate_off` values for all 32 teams. Its `12p_rate` field, however, is
degenerate: all 32 rows are present and all 32 values are zero. It therefore
contains no usable live cross-team 12-personnel state in this artifact.

Formation Usage supplies live 11/12/13 and other grouping counts directly and
conditions them on down and yards to go. Aggregate snap share, current depth,
and a single team PROE/pass-rate value cannot reconstruct that matrix.

### Personnel-conditioned play choice adds descriptive information

Using situation/personnel cells with at least five classified pass/rush plays:

- 186 cells qualified;
- they covered a median 46.4% of each team's classified plays;
- the median within-team, play-weighted absolute deviation from the team table's
  pass rate was **13.7 percentage points**.

To separate personnel from generic down/distance state, the audit then held
team, down and yards-to-go bucket fixed and compared multiple personnel groups:

- 66 same-team/same-situation comparisons qualified;
- 193 personnel cells and 2,480 classified plays were represented;
- median within-situation personnel pass-rate deviation was **11.4 percentage
  points**.

That is genuine incremental descriptive state beyond team-level pass tendency
and beyond the shared down/distance bucket. It is not a predictive result and
does not authorize coefficients, thresholds, or production use.

## Information classification after comparison

| Information family | Result | Class/action |
|---|---|---|
| Exact offensive 11-player co-occurrence | Absent from snaps, depth, PlayerForm, TeamForm and entitlement traces | **A confirmed — preserve prospectively** |
| Exact defensive 11-player co-occurrence | Absent from the current live stack; no assignment semantics | **A confirmed, lower immediate entitlement relevance** |
| Lineup concentration | Measurable from one snapshot | **A descriptive baseline** |
| Week-to-week lineup churn | Not identifiable from one cumulative snapshot | **Unresolved until repeated snapshots** |
| Replacement-lineup emergence timing | Not identifiable without consecutive point-in-time captures | **Unresolved; no current claim** |
| Pass/rush exposure by exact lineup | Not present in current marginal sources | **A confirmed descriptive state** |
| Live personnel grouping by down/distance | Not reconstructable from individual snaps/depth | **A confirmed — preserve prospectively** |
| Personnel-conditioned pass/rush choice | Adds conditional state beyond scalar PROE/pass tendency | **A confirmed descriptive state** |
| Lineup/personnel gain and scoring outcomes | Excluded from this audit | **B; no use** |

## Lineup Combinations schema-only gate

The prior authenticated cloud session was no longer active. A normal NFL login
attempt reached the NFL access-support page before password/MFA, and the already
provided screenshots do not show the Lineup Combinations table. No account,
credential, session, or access-control workaround was attempted, and no Lineup
Combinations values were acquired.

Because its columns were not observed, this audit does **not** claim that the
report is derivable from Lineup Detail. Operationally it remains excluded from
acquisition and cannot enter A-class scope unless a later authenticated,
schema-only check proves a field that Lineup Detail cannot derive. If the future
schema contains only combination identity plus counts already present in Lineup
Detail, classify it **C** and drop it permanently.

## Source-quality findings and next boundary

1. Lineup Detail and Formation Usage pass the incremental-information gate as
   live source families.
2. Their current value is joint role/personnel state, not efficiency outcomes.
3. Actual churn and replacement emergence require consecutive immutable weekly
   snapshots; the first capture is only the baseline.
4. WAS Formation Usage was source-empty at capture and must remain explicitly
   missing.
5. Eight defensive lineup rows had an ambiguous 10/12 parsed-player count; they
   were excluded. All 2,118 offensive rows parsed to exactly 11 players.
6. No model experiment is proposed or authorized by this document.
7. PR #662 remains draft. Production and Full Slate remain unchanged.

The reproducible aggregate audit is implemented in
`scripts/research/audit_gsis_incremental_information_v1.py`. Its output is
sanitized by construction and never emits player names, exact lineups,
team-level GSIS values, URLs, cookies, headers, tokens, or session material.
