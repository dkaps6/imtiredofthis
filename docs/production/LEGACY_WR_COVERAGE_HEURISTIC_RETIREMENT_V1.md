# Legacy WR Coverage Heuristic Retirement V1

Date: 2026-09-29  
Status: **PRODUCTION RETIREMENT CANDIDATE — USER AUTHORIZED, VALIDATION REQUIRED BEFORE MERGE**

## Decision

Retire the legacy `coverage_penalty()` adjustment from the production receiving
path.

The retired rule applied static target-share / YPT multipliers from:
- a `primary_cb` / shadow flag;
- team man-coverage rate;
- team zone-coverage rate.

It is not replaced by another coefficient or by a paid data source.

## Why

### 1. No reliable free production-grade WR<->CB assignment feed was established

A current public-source audit found useful free editorial/current matchup
information, including:
- RotoBaller weekly WR/CB matchup articles/charts;
- FantasyPros weekly WR/CB matchup articles;
- the VSiN WR/CB tool powered by Fantasy Points.

Those sources are useful for research, but this audit did not establish a
free, full-slate, machine-reproducible WR<->CB assignment feed with stable
weekly retrieval and historical parity suitable for production ingestion.

The full VSiN/Fantasy Points ecosystem is presented as a premium/pro tool, and
the public page shell does not provide a stable machine-readable full table
contract in this audit.

RotoBaller publishes free weekly analysis and chart visuals, but the chart is
editorial/visual, updated during the week, and not a stable machine-readable
full-board feed.

### 2. Current production did not actually have Week-3 player-level matchup authority

The exact preserved Week-3 production artifact showed:
- 161 WR football rows;
- 131 rows affected by the heuristic;
- 27/30 teams affected;
- **0 WR rows with a frozen nonmissing primary-CB assignment**;
- all active Week-3 adjustments came from the generic heavy-zone branch.

Therefore the production rule was functioning mostly as a broad team-zone WR
tilt, not as a genuine WR-vs-CB matchup model.

### 3. The heuristic is mechanically material

Exact outcome-free replay showed removal would change:
- affected WR final target entitlement: about -2.6%;
- affected WR YPT: -3.846%;
- affected WR entitlement x YPT mean kernel: about -6.4% median;
- affected-team TE and RB/FB target pools: about +2.9% through conservation.

This is too large to leave as an unvalidated grandfathered assumption.

### 4. Existing historical evidence does not justify keeping it

The closest historical no-coverage ablation was near-neutral. No exact modern
full-stack qualification of the static 0.94/0.92/1.04/1.06 coefficients was
found.

## Implementation

Production behavior change:
- remove `coverage_penalty()`;
- remove its call from `simulation_rules.apply_rules_to_metrics()`;
- WR `base_ypt` is no longer multiplied by static shadow/man/zone coverage
  coefficients;
- WR `base_tgt` receives only the independently retained target multiplier
  path already present in `matchup_multipliers()`.

This retirement does **not** remove all team-level defensive scheme context from
the model. Other team-level matchup multipliers are a separate scientific lane
and are unchanged here.

## Future WR-CB science

WR-CB matchup research remains open if a source later clears:
- full-slate coverage;
- pregame timing;
- stable player/defender identity;
- reproducible weekly extraction;
- historical or prospective temporal validation;
- no paid dependency unless explicitly authorized by the user.

Any future WR-CB signal must enter under a new frozen contract. It may not
silently restore these retired coefficients.

## No rescue

Do not replace the retired rule with:
- editorial matchup picks;
- manually transcribed weekly matchups;
- RotoBaller/FantasyPros article labels;
- team zone/man thresholds retuned from 2026 outcomes;
- sportsbook movement;
- another unvalidated static multiplier.

## Merge gate

Merge only after:
- repository CI passes;
- focused simulation-rules tests pass;
- static search confirms no production call to `coverage_penalty()` remains.

No OddsAPI spend is required.
