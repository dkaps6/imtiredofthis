# RB Vacancy Opportunity V1 — 2026 Week 3 Prospective Pregame Lock

Date: 2026-09-25 local / 2026-09-26 UTC  
Status: **LOCKED BEFORE TARGET-GAME OUTCOMES**

## Disposition

**`RB_VACANCY_OPPORTUNITY_V1_WEEK3_PREGAME_BASELINE_AND_CANDIDATE_LOCKED`**

The first legitimate prospective V1 vacancy cohort now has:
- immutable pregame availability;
- immutable no-outcome vacancy/transfer state;
- immutable exact-production baseline football means;
- immutable exact-production V1 candidate football means;
- a separately frozen grading contract;
- **zero target-game outcomes attached**;
- **zero sportsbook inputs used**.

No Week-3 result is promotion-eligible by itself.

## Authority chain

### Canonical pregame Full Slate

- run: `36204768034`
- job: `108298835652`
- source main SHA: `f7d2011b73950488ea209124ba895b92c401b2b1`
- artifact: `10892728623`
- digest: `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- live odds acquisition: disabled
- sportsbook inputs upstream: 0

### No-outcome vacancy state

- run: `36205201005`
- job: `108300143323`
- research head: `c34ba8b761f042aca66bcd028435b2ca9b270900`
- artifact: `10893677385`
- digest: `sha256:69b55421be779cb474eea0735bfe98490d8c7d5a8e0847e96f1f65d1c5e9e732`
- candidate rows: 4
- vacancy-event teams: 2
- exclusions: 0

### Exact production baseline/candidate projection lock

- run: `36205758768`
- job: `108301827908`
- workflow head: `5eca354b462554aff842f5bb144fac52dfca3fab`
- exact production source checkout: `f7d2011b73950488ea209124ba895b92c401b2b1`
- artifact: `10893588588`
- digest: `sha256:54437c69f0a66c2c9bb97c520f3e8d9c933e0203dac39fe1fde5352dafa3eeea`
- simulation: 25,000 iterations, seed 42
- active RB/FB cohort rows: 5
- projection rows: 10 (5 players × rush attempts/rush yards)
- target-game outcomes attached: 0
- sportsbook inputs used: 0

### Grading contract

`docs/research/RB_VACANCY_OPPORTUNITY_V1_WEEK3_EVALUATION_CONTRACT.md`

Frozen at commit:
`75cdff648db62239262cb67a18491436926c0db5`

The grading contract was committed before any Week-3 target-game outcome attachment.

## Frozen vacancy events

### DEN — Jonah Coleman unavailable

Most-recent same-team strict-prior rush share:
**0.357142857143**

Transfer:
- JK Dobbins: weight 0.40 -> **+0.142857142857**
- RJ Harvey: weight 0.60 -> **+0.214285714286**

### PIT — Rico Dowdle unavailable

Most-recent same-team strict-prior rush share:
**0.304347826087**

Transfer:
- Riley Nowakowski: weight 0.040540540541 -> **+0.012338425382**
- Jaylen Warren: weight 0.959459459459 -> **+0.292009400705**

No transfer was assigned to Eli Heidenreich because V1 requires usable strict-prior same-team offense snap participation for successor weighting.

## Exact locked production projections

The candidate changes **only `rules_rush_share`** for the four frozen direct recipients.

Unchanged:
- YPC / rushing efficiency;
- ML component;
- State component;
- ensemble weights;
- game script;
- team rush-attempt state;
- every non-rushing-share football input.

### DEN

| Player | Market | Baseline | V1 | Delta |
|---|---|---:|---:|---:|
| JK Dobbins | rush attempts | 12.293331 | 12.274076 | -0.019255 |
| JK Dobbins | rush yards | 47.951505 | 47.789661 | -0.161844 |
| RJ Harvey | rush attempts | 7.672867 | 8.260694 | **+0.587827** |
| RJ Harvey | rush yards | 35.579024 | 39.787778 | **+4.208753** |

Active DEN RB/FB projected attempts:
- baseline: **19.966199**
- V1: **20.534770**
- delta: **+0.568571**

### PIT

| Player | Market | Baseline | V1 | Delta |
|---|---|---:|---:|---:|
| Jaylen Warren | rush attempts | 12.288710 | 13.116147 | **+0.827437** |
| Jaylen Warren | rush yards | 49.947595 | 56.820052 | **+6.872457** |
| Riley Nowakowski | rush attempts | 2.524458 | 2.233885 | -0.290573 |
| Riley Nowakowski | rush yards | 16.466599 | 14.208183 | -2.258416 |
| Eli Heidenreich | rush attempts | 2.524197 | 2.170025 | -0.354173 |
| Eli Heidenreich | rush yards | 16.441968 | 13.704573 | -2.737395 |

Active PIT RB/FB projected attempts:
- baseline: **17.337366**
- V1: **17.520057**
- delta: **+0.182691**

## Important pre-outcome structural finding: production attenuates raw vacancy transfer

The no-outcome V1 transfer table conserves the frozen vacated historical rush-share amounts exactly.

However, the canonical production simulator does not consume those values as unconstrained literal final probabilities. It:
1. builds current Bayesian/rule rushing-share weights;
2. keeps the top five team rushing weights;
3. caps/normalizes modeled rushing allocation to 95% of team attempts, leaving a 5% residual bucket.

For both vacancy-event teams, the **baseline active-player raw rushing-share sum was already above 0.95**:

- DEN baseline raw team sum: **1.192960**
- DEN V1 raw team sum: **1.550103**
- PIT baseline raw team sum: **1.387306**
- PIT V1 raw team sum: **1.691653**

The final residual probability is therefore already **0.05** in baseline and remains **0.05** in V1.

This means the current production stack is already implicitly redistributing nearly all finite team rushing opportunity among active modeled runners. The V1 vacancy input therefore acts primarily as a **relative room-allocation signal**, not as a literal restoration of otherwise missing team carry mass.

That is visible in the effective allocation changes:
- DEN Dobbins probability: 0.382072 -> 0.381595
- DEN Harvey: 0.251762 -> **0.325084**
- PIT Warren: 0.332691 -> **0.436823**
- PIT Nowakowski: 0.237491 -> 0.201692
- PIT Heidenreich: 0.237491 -> 0.194763

This is not an outcome-based rescue or formula change. It is a mechanical property discovered before grading.

The consequence is important:
- the hypothesis being prospectively tested is now concretely whether the frozen vacancy signal improves **who receives the already-finite room opportunity** under the exact production architecture;
- the evaluation must include all active RB/FB teammates, including nonrecipient Heidenreich, because finite-volume normalization can move them;
- no post-outcome attempt may reinterpret only positive-recipient deltas as the candidate effect.

## Frozen Week-3 evaluation scope

Once target games are final, attach canonical actual:
- rush attempts;
- rush yards.

Grade:
- all 5 locked active RB/FBs;
- direct-recipient subset;
- each event team;
- predeclared M96A high-volume slices: actual carries 20+ and 25+.

Week-3 observational classification is frozen in the evaluation contract. It cannot promote V1 by itself.

## Anti-rescue rule

Do not change after outcomes:
- availability semantics;
- successor weights;
- transfer magnitudes;
- production seam;
- YPC;
- ensemble weights;
- cohort;
- 20+/25+ high-volume cutoffs;
- recipient-only selection;
- sportsbook upstream use.

If the Week-3 result fails, record it. Do not tune V1 against the exposed games.
