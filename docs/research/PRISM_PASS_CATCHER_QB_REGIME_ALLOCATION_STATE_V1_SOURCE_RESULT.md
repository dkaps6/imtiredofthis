# PRISM Pass-Catcher × QB-Regime Allocation State V1 — Source Result

Date: 2026-10-17

Status: **SOURCE READY ALL / PREDICTIVE SCORING NOT YET AUTHORIZED BY THIS RESULT**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_AUDIT.md`

Canonical execution:
- run: `39067415356` — **SUCCESS**
- job: `117497842701`
- exact head: `d189d2305b745d0c7f104e6f4fd8d0d305d1da16`
- artifact: `12113054267`
- digest: `sha256:61b701091bdfcdad32e86d0c6ba2e6a471b773656c95b4fa6179adbbb54214a2`

Disposition:

`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_READY_ALL`

## Scientific question

Can PRISM identify a leakage-safe target-week QB regime before kickoff and recover enough strictly-prior same-QB history to describe how the named WR/TE was allocated targets under that QB identity, separately from generic recent target-share state?

The answer at the **source/readiness layer** is yes for WR and TE in both 2024 and 2025.

This is not a predictive qualification and changes no projection.

## Exact support

Across the frozen 2024/2025 active-role pregame universe:
- pregame universe rows: **15,011**
- QB-regime team-weeks: **1,088**
- WR/TE target player-state rows: **9,096**
- source-ready family-season cells: **4 / 4**

### WR 2024

- target rows: **2,834**
- QB-regime source-valid team-week coverage: **98.8971%**
- target stable identity: **100%**
- prior completed-game primary-QB identity: **100%**
- player target/team-target identity: **100%**
- >=2 same-QB prior games: **76.3552%**
- >=3 same-QB prior games: **65.9772%**
- >=2 same-QB support: **2,141 rows / 219 players**
- source ready: **true**

### TE 2024

- target rows: **1,730**
- QB-regime source-valid team-week coverage: **98.8971%**
- target stable identity: **100%**
- prior completed-game primary-QB identity: **100%**
- player target/team-target identity: **100%**
- >=2 same-QB prior games: **78.1744%**
- >=3 same-QB prior games: **67.9930%**
- >=2 same-QB support: **1,336 rows / 127 players**
- source ready: **true**

### WR 2025

- target rows: **2,817**
- QB-regime source-valid team-week coverage: **98.3456%**
- target stable identity: **100%**
- prior completed-game primary-QB identity: **100%**
- player target/team-target identity: **100%**
- >=2 same-QB prior games: **75.5957%**
- >=3 same-QB prior games: **65.0181%**
- >=2 same-QB support: **2,094 rows / 221 players**
- source ready: **true**

### TE 2025

- target rows: **1,715**
- QB-regime source-valid team-week coverage: **98.3456%**
- target stable identity: **100%**
- prior completed-game primary-QB identity: **100%**
- player target/team-target identity: **100%**
- >=2 same-QB prior games: **76.9139%**
- >=3 same-QB prior games: **66.6469%**
- >=2 same-QB support: **1,296 rows / 129 players**
- source ready: **true**

All frozen gates passed in all four family-season cells.

## Integrity

- chronology violations: **0**
- target-game outcomes used in source state: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- predictive candidates scored: **0**
- coefficients fit: **0**
- thresholds searched: **0**
- production changed: **false**

## Interpretation

This result establishes that the repository can legally ask a genuinely player-first question that was previously unavailable:

> Does the target share of this named WR/TE change in a repeatable way when the pregame QB regime is the same named QB, after accounting for the receiver's ordinary strictly-prior role state?

The result does **not** establish that:
- same-QB target share should replace current entitlement;
- a QB change should automatically boost or reduce any receiver;
- same-QB history predicts YPT or receiving yards;
- WR-R15 or TE-R5P should be overwritten;
- finite allocation may be bypassed.

The next test must remain at the target-allocation/share layer. Efficiency and market translation stay downstream.

## Forward boundary

Freeze a separate predictive allocation-state test **before scoring**.

The first predictive test must:
- use 2024 as development only and 2025 as untouched confirmation;
- target next-game player target share;
- keep WR and TE separate;
- use same-QB-regime allocation state only as an incremental signal;
- control for ordinary strictly-prior target-share state / recent trajectory;
- avoid a raw same-QB-share replacement;
- preserve the finite-opportunity principle;
- make no claim of WR-R15/TE-R5P production integration unless an exact OOS authority exists for the scored season;
- use no sportsbook, OddsAPI or 2026 outcomes.

A pass at this stage would authorize a separately frozen production-authority integration test; it would not promote a projection directly.
