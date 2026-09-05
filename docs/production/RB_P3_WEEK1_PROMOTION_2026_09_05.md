# RB P3 Week 1 Production Promotion — 2026-09-05

## Scope

This promotion is deliberately limited to **RB rushing yards for 2026 Week 1**.
It does not authorize the unresolved Weeks 2–18 enriched-allocation source path.
The production implementation fails closed outside Week 1 until that source
contract is separately resolved.

## Research lineage

- RB STACK1 canonical full-stack reconstruction: run `33535308110`.
- RB STACK2 enriched opportunity/allocation: run `33538770934`.
- RB STACK3 deterministic P3 composition/champion: run `33539468967`.
- P3 contract:
  - Week 1 = STACK1 full-stack rushing-yard projection unchanged.
  - Weeks 2–18 = enriched RB carries × STACK1 implied YPC.
- Final RB qualification preserved P3 as the best overall football research
  model while retaining the documented high-end ceiling-compression blocker;
  no tail-waiver or retrospective retuning was authorized.

## Frozen Week 1 production parameters

Promoted STACK1 ensemble weights:

| Market | MC | ML | State |
|---|---:|---:|---:|
| `rush_att` | 0.3164919683016017 | 0.6528957474344519 | 0.030612284263946517 |
| `rush_yards` | 0.5569542426070742 | 0.4430457573929258 | 0.0 |

P3 production version: `RB_P3_SYNTHESIS_V1`

Week 1 route: `WEEK1_STACK_OVERRIDE`

Sportsbook contract: **football-only**. The P3 context is built from
Ourlads/schedule + football model inputs with `FETCH_LIVE_ODDS=false` semantics.
Sportsbook lines and odds enter only after the final football mean exists.

## Historical parity evidence

The production reconstruction reproduced the authoritative historical P3
composition row-by-row to floating-point tolerance and reproduced the frozen
2025 P3 aggregate metrics used for final qualification. The production branch
also restores the exact validated STACK1 RB ensemble weights rather than the
previous fresh-checkout MC-only fallback.

## 2026 Week 1 dry-run evidence

Canonical all-green no-odds Week 1 gate before pricing integration:

- run `33832680431`
- job `100898785942`
- SHA `3922a6b09205fb28b061fecad948cf91b74e8536`
- artifact `9922264863`
- artifact digest `sha256:e23a314810bb106173a057be9a5212f41570608c1316d5bd18b3d1b50fc40cd8`

Result:

- 107 RB/FB players;
- 32 teams;
- every row routed through `WEEK1_STACK_OVERRIDE`;
- every Week 1 P3 projection equaled its full-stack parent within tolerance;
- calibrated STACK1 `rush_att` and `rush_yards` ensemble provenance preserved;
- sportsbook inputs used = 0;
- strict repository audits passed.

Expanded pricing-adapter gate:

- run `33993422377`
- job `101379581768`
- SHA `bc00094c340fd8e061c5f76b10eb5f1974a745e7`
- conclusion: **success**

This run additionally proved that the production pricing adapter resolves every
live Week 1 RB identity back to the football-only P3 context with no projection
or route drift, and the full repository test suite passed.

## Production files

- `data/model_ensemble_weights.csv`
- `scripts/modeling/rb_rush_synthesis_v1.py`
- `scripts/modeling/rb_pricing_adapter_v1.py`
- `scripts/run_rb_week1_no_odds.py`
- `scripts/run_pricing_v2.py`
- `.github/workflows/full-slate.yml`
- `.github/workflows/rb-week1-promotion-gate.yml`
- `tests/test_rb_rush_synthesis_v1.py`
- `tests/test_rb_pricing_adapter_v1.py`

The Full Slate path now builds the promoted RB football context before
odds-dependent metrics/pricing. `run_pricing_v2.py` still calculates and emits
MC/ML/State/ensemble diagnostics, but for Week 1 `rush_yards` the authoritative
final football mean is P3. The simulator distribution is rescaled around that
mean before downstream sportsbook probability comparison, matching the existing
position-specific synthesis pattern used for promoted QB passing yards.

## Fail-closed gates

Production must fail rather than silently fall back if:

- the promoted RB context is missing/empty;
- a Week 1 RB identity is absent or duplicated;
- team/opponent identity disagrees;
- the P3 version or route drifts;
- sportsbook-leakage flags are nonzero;
- the generic `rush_yards` ensemble is not calibrated;
- final `model_proj` differs from `rb_synthesis_proj` beyond numerical tolerance;
- a `rush_yards` pricing request targets a week other than Week 1 under this
  promotion version.

## Final integration gate

A stricter no-Odds-API end-to-end test was added after the successful adapter
run. It builds synthetic downstream line/odds fields only after the football
context exists, invokes the real `run_pricing_v2.price()` path for the live
Week 1 RB universe, and requires every final rushing-yard mean to equal P3.

Candidate run: `33993595929`, SHA
`1daba60e3a6cd53319ba3223a1fc5f7e7184222e`.

**Status at document creation: in progress.** This document must be updated with
that run's final conclusion before merge to `main`.

## Disposition

- Week 1 RB research architecture: **FROZEN**.
- Week 1 no-odds production context: **QUALIFIED**.
- Week 1 pricing adapter: **QUALIFIED** by run `33993422377`.
- Final end-to-end synthetic pricing gate: **PENDING** at document creation.
- Weeks 2–18 enriched allocation: **NOT PROMOTED by this ledger entry**.
- No sportsbook variable is authorized upstream of the football projection.
