# Football Context Signal Qualification V1

**Status:** FROZEN PRE-EXPERIMENT QUALIFICATION GATE  
**Scope:** engineering/source/QA only; no predictive-lift claim; no production integration authorization.

## Purpose

Turn the newly defined football-context layers into a disciplined shortlist of genuinely useful information before any predictive experiment is authorized.

A signal is not experiment-ready merely because it is intuitive, novel, correlated with an outcome, or available in a source. It must first prove that it is pregame-knowable, historically materializable, sufficiently covered, stable enough to estimate, identity-safe, and mechanistically connected to a specific projection component.

## Candidate families

The initial qualification inventory covers:

1. player role/environment regime;
2. personnel continuity and vacated opportunity;
3. historical matchup analog descriptors;
4. WR/TE defender-proximity exposure;
5. OL/DL personnel and protection context;
6. certified BDB geometry/history summaries.

## Required qualification dimensions

Every candidate feature must receive explicit evidence for all of the following before a predictive plan may be frozen.

### 1. Pregame availability

Report the fraction of target player-games/team-games for which the value can be constructed using only information timestamped before kickoff.

Postgame reconstruction does not count as pregame availability.

### 2. Historical coverage

Report coverage by season, week, position, team and relevant role tier. A high global coverage rate cannot hide a severe position or era hole.

### 3. Identity and join integrity

Report stable-ID coverage, unmatched rate, ambiguous-match rate, duplicate published keys and canonical-base fanout. Duplicate published keys and silent fanout must be zero.

### 4. Missingness semantics

Distinguish `UNKNOWN`, true zero, not-applicable and source-unavailable. Missing values must not silently become zeros.

### 5. Strict-prior stability

Where a signal represents player/team tendency rather than a one-time event, quantify strict-prior repeatability across adjacent historical windows. Recommended descriptive statistics include Spearman rank correlation, median absolute change, sample-size distribution and shrinkage sensitivity.

No target-game outcome may be used to construct the feature being assessed.

### 6. Mechanism mapping

Each candidate must name the projection component it could plausibly inform. Examples:

- role change / vacated opportunity -> carries, targets, route/participation entitlement;
- OL continuity / edge mismatch -> pressure, sacks, QB efficiency and scramble environment;
- receiver spacing/proximity -> target quality, catch conversion, YAC/explosive environment;
- historical analog descriptors -> uncertainty/regime identification, not direct outcome lookup;
- blocker geometry stability -> protection continuity or matchup uncertainty, not universal assignment.

Candidates without a specific football mechanism remain descriptive only.

### 7. Redundancy audit

Before experimentation, compare the candidate against already-available canonical inputs. A candidate that is almost entirely reconstructible from existing features should not receive priority solely because it has a new name.

This is a feature-information audit, not a predictive model fit.

### 8. Sample support

Report total eligible rows and the distribution of prior observations used to construct the feature. Thin-history cohorts must be explicitly tagged and may require shrinkage or an unknown state.

## Qualification dispositions

Each candidate receives exactly one disposition:

- `READY_FOR_FROZEN_EXPERIMENT` — all integrity gates pass and the signal has sufficient pregame coverage, support, stability/reliability and a clear mechanism;
- `ENGINEERING_READY_SOURCE_THIN` — construction is valid but coverage/sample support is too thin for a broad experiment;
- `DESCRIPTIVE_ONLY` — valid football information without a sufficiently specific predictive mechanism or incremental-information case;
- `SOURCE_BLOCKED` — required authoritative information is unavailable;
- `REJECTED_INTEGRITY` — leakage, identity, fanout or semantic defects invalidate the candidate.

## Priority rule

Qualification should favor signals that address known projection failure mechanisms rather than broad feature fishing.

Initial high-value diagnostic questions are:

1. Can role/environment and personnel-continuity features identify games where recent historical usage is stale because the player's team/room/availability regime changed?
2. Can OL/DL continuity and certified blocker-history features describe protection environments not captured by aggregate team pressure rates?
3. Can receiver proximity/spacing histories describe stable receiver-route environments not already represented by target share and ordinary receiving efficiency?
4. Can analog descriptors identify regime similarity without using neighbor outcomes during neighbor selection?

## Required output table

Publish one row per candidate feature with at least:

- feature name;
- family;
- grain;
- seasons available;
- eligible rows;
- pregame coverage;
- stable-ID coverage;
- unknown rate;
- duplicate/fanout status;
- strict-prior support distribution;
- stability statistic where applicable;
- intended projection component;
- redundancy notes;
- provenance/source class;
- qualification disposition;
- blocker/reason.

## Scientific firewall

This qualification pass MUST NOT:

- fit a projection model;
- score candidate features against sportsbook lines;
- select thresholds based on target-game error;
- use target-game outcomes to choose analog neighbors;
- retune frozen production components;
- reopen failed closed research families;
- touch Issue #535.

A predictive experiment may begin only after a candidate is qualified and a separate frozen scientific plan defines its hypothesis, cohorts, metrics and pass/fail gates before outcomes are inspected.

## Immediate implementation order

1. build a qualification-inventory materializer that consumes the existing context manifests/QA outputs;
2. inventory existing canonical fields to establish redundancy baselines;
3. materialize the highest-coverage strict-prior role/personnel features first;
4. attach BDB-derived stability evidence already certified by the data-frontier work;
5. rank candidates by qualification disposition and mechanism relevance only;
6. freeze a separate predictive plan only for candidates marked `READY_FOR_FROZEN_EXPERIMENT`.

## Disposition

`FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1_FROZEN`
