# PRISM Pass-Catcher × QB-Regime Target-CPOE State V1 — Source Result

Date: 2026-10-17

Status: **SOURCE READY ALL / SCORING NOT YET PERFORMED**

Frozen source plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_STATE_V1_SOURCE_PLAN.md`

## Canonical source authority

- workflow: `Research PRISM Pass Catcher QB Regime Target CPOE Source V1`
- run: `39087841064` — **SUCCESS**
- exact head: `d30c310b99377874695343f20d565be676244a61`
- artifact: `12121277666`
- artifact name: `prism-pass-catcher-qb-regime-target-cpoe-source-v1-39087841064`
- digest: `sha256:d5ad7404f2f1f99890b82b2cabc9dfa6af8c54bd8712d4150dd479e2fea6ddc3`
- disposition:
  `PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_STATE_V1_SOURCE_READY_ALL`

## Frozen family-season gates

| Family | Season | Exact-state rows | Players | QB regime | Receiver ID | Passer ID | Native CPOE | Ready |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WR | 2024 | 1,451 | 146 | 98.897% | 100% | 100% | 99.994% | YES |
| TE | 2024 | 641 | 67 | 98.897% | 100% | 100% | 99.994% | YES |
| WR | 2025 | 1,409 | 151 | 98.346% | 100% | 100% | 99.994% | YES |
| TE | 2025 | 676 | 71 | 98.346% | 100% | 100% | 99.994% | YES |

Chronology violations: **0**.

Exact-state rows satisfy:
- latest up to 8 prior target-bearing receiver games selected first;
- >=4 prior target-bearing games;
- >=16 finite-CPOE ordinary targets;
- >=2 same-QB prior games;
- >=8 same-QB finite-CPOE targets.

## What this proves

The repository can test the pair-conditioned conversion-quality hypothesis without changing the source contract:

> Does the named receiver's target CPOE with the currently selected QB predict next-game target CPOE better than the receiver's ordinary recent CPOE state?

The source pass does not say the candidate works.

## Anti-retest boundary preserved

This result does not reopen:
- same-QB target share;
- same-QB target depth;
- raw QB-receiver pair YPT;
- WR-R18 through R20;
- QB M70/M81;
- generic catch-rate/YPT/YPR correction families.

## Integrity

- predictive candidates scored: 0;
- chronology violations: 0;
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

Freeze exactly one direct target-CPOE prediction test before exposing target-game CPOE outcomes.

Target entitlement and target depth must remain unchanged.
