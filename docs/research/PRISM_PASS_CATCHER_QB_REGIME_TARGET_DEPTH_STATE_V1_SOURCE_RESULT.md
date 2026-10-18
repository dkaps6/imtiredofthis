# PRISM Pass-Catcher × QB-Regime Target-Depth State V1 — Source Result

Date: 2026-10-17

Status: **SOURCE READY ALL / SCORING NOT YET PERFORMED**

Frozen source plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_PLAN.md`

## Canonical source authority

- workflow: `Research PRISM Pass Catcher QB Regime Target Depth Source V1`
- run: `39087349029` — **SUCCESS**
- exact head: `57e14d7675ad57f394f76db55fb1b978e9cee4a1`
- artifact: `12120839869`
- artifact name: `prism-pass-catcher-qb-regime-target-depth-source-v1-39087349029`
- digest: `sha256:d4b6af7295f39b359e8ff9e872b6b4a0364c547e4beb0d512fb68200d2ebff51`
- disposition:
  `PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_ALL`

All workflow steps passed:
- frozen safeguards;
- source execution;
- source-boundary certification;
- strict repository audit;
- result record;
- artifact upload.

## Frozen family-season gates

| Family | Season | Exact-state rows | Players | QB regime | Receiver ID | Passer ID | Air yards | Ready |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WR | 2024 | 1,727 | 163 | 98.897% | 100% | 100% | 100% | YES |
| TE | 2024 | 896 | 85 | 98.897% | 100% | 100% | 100% | YES |
| WR | 2025 | 1,657 | 174 | 98.346% | 100% | 100% | 100% | YES |
| TE | 2025 | 898 | 85 | 98.346% | 100% | 100% | 100% | YES |

Chronology violations: **0**.

The exact-state rows already satisfy the frozen V1 history semantics:
- latest up to 8 prior same-team receiver games;
- >=4 prior receiver-games;
- >=10 finite-air-yard targets;
- >=2 same-QB prior receiver-games;
- >=4 finite-air-yard same-QB targets;
- current QB selected only from the pregame QB-regime authority.

## What this proves

The repository has enough strict-prior, player/QB-identified target-depth history to test the genuinely distinct hypothesis:

> Does the named pass catcher's target depth under the currently selected QB predict his next-game target depth better than his ordinary player target-depth history?

This source pass does **not** say the signal works.

## What remains closed

This result does not reopen:
- direct same-QB target-share adjustment (run `39067914161`, CLOSED);
- raw QB-receiver pair YPT;
- C1/C3 shared QB/receiver families;
- R7 signed air-yard/YAC/explosive level-trait work;
- universal symmetric target-depth distribution transform;
- direct WR/TE depth-YPT matchup transform;
- generic DvP / generic opponent multipliers.

## Integrity

- predictive candidates scored: 0;
- target-game outcomes used in predictor: 0;
- receiving yards scored: false;
- receptions scored: false;
- touchdowns scored: false;
- market translation scored: false;
- coefficients fit: 0;
- thresholds searched: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.

## Authorized next step

Freeze one exact target-depth scoring mechanism before outcome exposure.

Target entitlement must remain unchanged. A pass may authorize only a separately frozen market-specific translation test later.
