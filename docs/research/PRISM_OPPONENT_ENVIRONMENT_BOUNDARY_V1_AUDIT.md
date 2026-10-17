# PRISM Opponent / Environment Boundary V1 — Anti-Retest and Candidate-Eligibility Audit

Date: 2026-10-16

Status: **AUDIT COMPLETE / NO OPPONENT CANDIDATE SCORED / NO CLEAN V1 CANDIDATE AUTHORIZED YET**

Model:
**PRISM — Player Role & Individual Situation Model**

Repository:
`dkaps6/imtiredofthis`

Active research branch:
`research-individual-opportunity-roadmap-2026-10-08`

## 1. Purpose

PRISM's canonical architecture places opponent/environment after opportunity
quantity and opportunity quality:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

The two newly completed team-side opportunity-quality mechanisms both closed:

1. direct same-team recent high-value event-rate persistence:
   `PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`;
2. same-team scoring-territory drive-entry persistence:
   `PRISM_TEAM_DRIVE_CREATION_CLOSED`.

The governing user instruction still says **do not add opponent/environment
transmission yet**.

This audit therefore does not score an opponent candidate.

Its job is to determine whether a future opponent layer can be opened cleanly
without:
- reopening generic defense-vs-position shortcuts;
- repackaging already-failed Football Matchup Transmission formulas;
- hiding player-specific logic behind a team/position average;
- using unsupported or non-parity sources.

## 2. Player-first standard

The opponent is an obstacle the named player must move through. It is not the
player's identity and it is not a position-wide multiplier.

A future PRISM opponent mechanism must answer a player-conditioned question such
as:

> Given this named player's current role, opportunity type and teammate
> competition, what property of this opponent changes the path or value of
> those specific opportunities?

It is not sufficient to say:
- opponent is bad vs RB;
- opponent allows many yards to TE;
- opponent recently allowed many red-zone touches;
- defense is bottom-N versus a position.

Downstream betting/ticket construction must remain separate:
approved PRISM player projections first, then market comparison / alt-line /
correlation / ticket construction.

## 3. Protected new parents

### Canonical parent opportunity quantity

`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`

- run `38991915503`;
- artifact `12085194208`;
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`;
- field `authority_denominator_mean`.

### Qualified player allocation

- Stage 1 active-role: QUALIFIED ALL;
- Stage 2 conditional allocation:
  - RB rushing QUALIFIED;
  - RB receiving QUALIFIED;
  - TE receiving QUALIFIED.

QB Stage-2 remains CLOSED.
WR Stage-2 remains source-blocked/not scored.

### Opportunity-quality closures

Direct team high-value creation:
- freeze `b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`;
- run `38999998874`;
- artifact `12087739890`;
- disposition `PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`.

Drive territory creation:
- freeze `82f74487e68729efdbcc0868968fd6ddccd4eef1`;
- run `39000352640`;
- artifact `12088459232`;
- disposition `PRISM_TEAM_DRIVE_CREATION_CLOSED`.

Neither may be rescued by adding an opponent term to the same failed formula.

## 4. Opponent source readiness already exists — but source readiness is not candidate permission

The raw opportunity-quality source already proved strict-prior opponent-state
attachment is source-ready.

Available reconstructable opponent-side objects include:
- high-value rushing opportunities allowed;
- red-zone targets allowed;
- parent rushing opportunities faced;
- parent targets faced;
- drives allowed to reach opponent 20 / 10 from the offense perspective;
- strict-prior last-up-to-4 opponent history;
- strict-prior league environment.

The source audit reported 100% strict-prior opponent-history attachment for the
2024/2025 target population.

This means lack of data is not the blocker for a basic opponent-allowance rate.

The blocker is **scientific novelty / anti-retest**.

## 5. Historical Football Matchup Transmission V1 — required anti-retest parent

Branch:
`research-football-matchup-transmission-v1`

### Phase B/C diagnostic result

Canonical:
- run `37514137803`;
- artifact `11436786668`;
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`.

Four Phase-B specs replicated descriptively.

Three exact-live-semantic signals advanced to separately frozen integration
candidates:
1. RB rushing — opponent `def_pass_rate_faced`;
2. WR receiving — offense `true_proe`;
3. TE receiving — opponent `def_pass_success_allowed`.

TE position YPT allowed replicated only diagnostically and remained source-parity
blocked.

Critically:
**no predeclared Phase-C usage × matchup interaction replicated across both
2024 and 2025.**

That means the already-tested simple idea

`player opportunity share × generic matchup weakness`

is not an open family.

### Frozen integration result

Canonical:
- run `37519246784`;
- artifact `11440195795`;
- digest `sha256:4e2fac59ad262de477c3cd9acbaeae623db619aba2b54e5a88606177a8f01c38`.

All three frozen V1 integrations failed closed:

RB opponent pass-rate-faced:
- primary RB rush-yards change trivial;
- failed supported improvement;
- caused supported collateral harm in RB rush+receiving and RB receiving.

WR true-PROE:
- small favorable direction;
- failed the frozen replicated cluster/RMSE gate.

TE opponent pass-success allowed:
- 2025 improved;
- 2024 did not replicate;
- failed the required two-season gate.

Therefore CLOSED:
- generic opponent pass-rate-faced pass/rush replacement;
- simple TE pass-success multiplier;
- simple production pass-share replacement from true PROE;
- post-hoc coefficient or role carveout rescue.

## 6. Older opponent / matchup families that remain closed or blocked

### Generic RB role × run-defense

M95A/M95B:
- descriptive football truth existed;
- generic prospective role+defense / offense-defense integration was mixed or
  worse out of year;
- no stable production promotion.

Do not reopen:
`weak run defense -> boost RB`.

### QB generic defensive adaptation

- M56 richer static defensive matchup: failed signal screen;
- M83 conditional defensive adaptive gameplan:
  `NO_DEFENSIVE_ADAPTATION_MECHANISM`.

QB remains a control / separate promoted M89/M90 path.

### Opponent defender injury

Canonical source audit:
- run `37494717496`;
- artifact `11426662979`;
- disposition `SOURCE_PARITY_NOT_CLEARED`.

Report-status completeness was far below the frozen >=95% requirement.

Do not substitute practice status or infer blanks as active.

### WR-CB assignment

Historical direct WR-CB assignment remains source-blocked without a qualifying
free source.

Do not invent assignment.

### Position-YPT / box / coverage proxies

Historical public proxies for:
- position YPT allowed;
- box rates;
- man/zone;
- outside/slot;
- middle-open;

either lack exact live semantic parity or are source-blocked.

A descriptive signal is not permission to promote a mismatched proxy.

## 7. Candidate-family audit

### A. Opponent recent high-value event rate allowed

Example:
`prior opponent inside-10 rushes allowed / prior opponent rush attempts faced`.

Source status:
**READY**.

Candidate status:
**NOT AUTHORIZED AS NEXT V1**.

Reason:
- mechanically mirrors the just-closed direct same-team recent high-value event
  persistence family on the other side of the matchup;
- functions as a generic defense rate before any player-conditioned mechanism;
- risks becoming the exact forbidden “bad defense vs position => boost player”
  shortcut.

It may be descriptive context later, but should not be the first opponent
integration candidate.

### B. Opponent recent scoring-territory drives allowed

Source status:
**READY**.

Candidate status:
**NOT AUTHORIZED AS NEXT V1**.

Reason:
- symmetric defensive version of the newly closed team drive-persistence
  mechanism;
- still a generic team tendency;
- not sufficiently player-conditioned to justify another transmission trial.

### C. Generic opportunity-share × opponent weakness

Source status:
varies by feature.

Candidate status:
**CLOSED**.

Reason:
Football Matchup Transmission Phase C already tested the predeclared usage ×
matchup interaction family and none replicated across both seasons.

Do not retry with new thresholds, lead-player subsets or renamed weakness
scores.

### D. TE opponent pass-success multiplier

Source status:
exact semantics available.

Candidate status:
**CLOSED IN V1 FORM**.

Reason:
replicated diagnostically, but the separately frozen integration failed
two-season confirmation.

### E. Opponent pass-rate-faced for RB volume

Source status:
exact semantics available.

Candidate status:
**CLOSED IN V1 FORM**.

Reason:
the separately frozen integration failed supported primary improvement and
caused collateral RB harm.

### F. Position-specific YPT / coverage / WR-CB

Candidate status:
**SOURCE-PARITY BLOCKED / SOURCE BLOCKED**.

No scoring is authorized.

## 8. What would count as genuinely new opponent state

A future opponent candidate must introduce a football object that is not just a
new transform of the closed generic matchup families.

Examples of scientifically distinct source classes, if exact historical/live
parity can first be proven:
- opponent personnel/package response to the named player's actual role;
- route/alignment responsibility matched to the named player's current route
  identity;
- defensive front / box / run-fit state matched to the named player's actual
  carry type / role rather than position alone;
- scoring-area defensive topology matched to the named player's current
  high-value role;
- opponent state conditional on the specific opportunity subtype that PRISM
  already projects for the named player.

These are **source/audit directions only**.

They are not authorized candidates today.

## 9. Why the current boundary should stay closed

At this checkpoint, the obvious opponent variables fall into one of four
categories:

1. generic source-ready team allowance rates too close to newly closed
   opportunity-quality persistence;
2. previously tested simple matchup-transmission formulas that failed;
3. player-usage × matchup interactions that failed replication;
4. richer player-conditioned matchup data that lacks exact source parity or is
   source-blocked.

Therefore the correct scientific disposition is:

`PRISM_OPPONENT_ENVIRONMENT_BOUNDARY_AUDIT_COMPLETE__NO_CLEAN_CANDIDATE_YET`

This is not a claim that matchup does not matter.

It is a claim that PRISM should not manufacture a generic matchup multiplier
just to fill the architecture box.

## 10. What can continue without crossing the opponent boundary

While opponent/environment remains unopened, PRISM may continue only along
already-authorized, non-opponent layers that do not reopen closed science, for
example:
- player-specific efficiency under already-qualified football conditions;
- player projection reliability / trust / selectivity;
- market-specific mapping of qualified player state;
- source-parity work for richer player-conditioned opponent objects.

Any such lane still requires its own anti-retest review and freeze before
scoring.

The locked Week-6 Stage-2 shadow remains untouched and ungraded until all 13
included games are final.

## 11. Integrity

This audit:
- scores predictive opponent candidates: **0**;
- reads 2026 outcomes: **0**;
- uses sportsbook inputs: **0**;
- calls OddsAPI: **0**;
- fits coefficients: **0**;
- searches thresholds: **0**;
- changes production: **false**;
- changes canonical parent opportunity: **false**;
- changes qualified player allocation: **false**.

No opponent/environment boundary has been crossed.
