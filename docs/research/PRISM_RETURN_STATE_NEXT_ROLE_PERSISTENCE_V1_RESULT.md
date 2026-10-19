# PRISM Return-State Next-Role Persistence V1 — Result

Date: 2026-10-18

Status: **QUALIFIED STATE ONLY / NAMED-PLAYER CURRENT-ROLE PERSISTENCE / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_FREEZE.md`

Freeze:
`c25d5f3d19cab3f68a0c0ad0482c79bf53a6730e`

Canonical execution:
- run `39179843306` — **SUCCESS**;
- exact head `57c962a935f25a6fbad521c3aad43dc2db3e4bb9`;
- job `117830060246` — **SUCCESS**;
- artifact `12156195287`;
- artifact name `prism-return-state-next-role-persistence-v1-39179843306`;
- digest `sha256:8a6ac90e3209f1ec8f53d859a2f7b2a64b84b682f033667512d99b16ddee3988`;
- strict repository audit: **PASS**.

Disposition:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`

## Player-first question

Once a named player has source-certified
`RETURN_PARTICIPATION_RESTORED`, does his immediately following clean
same-team participation state behave more like his own pre-injury role than
like the interruption / continued-decline participation state?

The answer under the frozen V1 contract is **yes**.

This is participation-state evidence only.

## Frozen comparison

For every eligible restored episode:

- pre-injury anchor:
  `pre_episode_snap_pct`;
- continued-decline participation anchor:
  - episode-start snap percentage for an injury-corroborated positive-participation collapse;
  - `0.0` only for the source-certified no-offensive-snap participation state;
- target:
  the immediately next scheduled same-team observation after the restored
  return;
- target must be exactly `NO_REPORT_PARTICIPATED`;
- no skipping forward past a reported or unresolved next observation.

Primary gain:

`abs(decline_anchor - actual_next_offense_pct)
 - abs(pre_role_anchor - actual_next_offense_pct)`

Positive means the named player's own pre-injury participation role was closer.

## 2024 development

- scored episodes: **84**;
- named players: **79**;
- equal-player average gain: **+0.336361**;
- median named-player gain: **+0.300000**;
- improved / tied / worsened:
  **86.08% / 0.00% / 13.92%**;
- p10 / p25 / p75 / p90:
  **-0.026000 / +0.100000 / +0.550000 / +0.818000**.

Family diagnostics only:
- QB: 7 episodes / 6 players, equal-player **+0.802500**;
- RB: 26 / 24, **+0.219375**;
- TE: 20 / 20, **+0.361250**;
- WR: 31 / 29, **+0.319569**.

All frozen development gates passed.

## 2025 independent confirmation

- scored episodes: **105**;
- named players: **95**;
- equal-player average gain: **+0.299474**;
- median named-player gain: **+0.255000**;
- improved / tied / worsened:
  **85.26% / 0.00% / 14.74%**;
- p10 / p25 / p75 / p90:
  **-0.058000 / +0.068750 / +0.451250 / +0.809000**;
- named-player bootstrap 95% interval:
  **[+0.234841, +0.365316]**.

Family diagnostics only:
- QB: 8 episodes / 8 players, equal-player **+0.855000**;
- RB: 23 / 18, **+0.161389**;
- TE: 30 / 28, **+0.239018**;
- WR: 44 / 41, **+0.292988**.

All frozen confirmation gates passed.

No family result was needed to rescue the contract.

## Fail-closed immediate-next observations

The scorer did not skip forward when the immediate next observation was not a
clean ordinary participation state.

Excluded:
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **13**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **20**;
- `REPORTED_INJURY_PARTICIPATED`: **93**;
- no immediate next observation: **34**.

That exclusion is causal hygiene, not a favorable-outcome search.

## What this means for an individual player

Once the player's return participation is source-certified as restored, PRISM
should not automatically keep carrying the injury interruption forward as if
the player's participation role were still in decline.

For the named player, the restored state has predictive persistence into the
next clean participation observation.

This is exactly the state distinction the Coker failure showed was missing:
an injury interruption is not ordinary trajectory evidence, and a restored
return changes the player's current participation state.

## What this does NOT mean

This result does not establish:
- restored target share;
- restored carry share;
- restored receptions;
- restored route share;
- restored efficiency;
- restored yards;
- restored touchdown access;
- medical full health.

Snaps are snaps, never routes.

Participation restoration remains separate from target earning.

Target earning remains separate from efficiency.

No automatic yards multiplier is authorized.

## Teammate competition / finite opportunity

The V1 mechanism changes no team target total, carry total, target share, carry
share, or teammate allocation.

Existing finite-opportunity / teammate-competition architecture remains
untouched.

Any later opportunity integration must be separately frozen and must conserve
team opportunity.

## Anti-retest

Do not rescue or retune this qualified V1 with:
- another snap window;
- another return threshold;
- second-game / third-game lookahead;
- family-specific thresholds;
- position-specific coefficients;
- blends of pre-role and return snaps;
- target-share deletion/reweighting;
- the closed exposure x earning formula.

The exact V1 is qualified and frozen.

## Exact next scientific boundary

The next step stays in the same return-state causal lane and moves one layer
downstream:

> After participation is source-certified restored for this named player, can
> PRISM source-validly identify the player's market-specific opportunity state
> without assuming that restored snaps automatically restore target/carry
> earning?

The first legal step is source/lineage only.

For receiving opportunity, source-map exact stable player + team + week target
evidence and finite team target volume onto restored-return episodes before any
predictive target-share relationship is scored.

Do not:
- infer target restoration from snap restoration;
- reuse the closed exposure x earning formula;
- delete/reweight injury rows as a rescue;
- use 2026 outcomes;
- use sportsbook/OddsAPI upstream;
- mutate production.

## Integrity

- workflow: **SUCCESS**;
- frozen-before-implementation verification: **PASS**;
- immutable source verification: **PASS**;
- focused tests: **PASS**;
- strict repository audit: **PASS**;
- targets used: **0**;
- carries used: **0**;
- receptions used: **0**;
- yards used: **0**;
- touchdowns used: **0**;
- routes inferred from snaps: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.
