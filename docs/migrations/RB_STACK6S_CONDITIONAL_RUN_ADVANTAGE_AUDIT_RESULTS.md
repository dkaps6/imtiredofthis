# RB STACK6S — Conditional Run-vs-Pass Advantage Audit Results

## Status

Authoritative no-fit audit completed successfully. Neither predeclared conditional efficiency signal qualified. No predictive model, P3 recomposition, or production change is authorized.

## Authoritative run

- Workflow: `RB STACK6S Conditional Run Advantage Audit`
- Run: `33654839590`
- Job: `100330692497`
- Branch: `research-rb-stack6s-conditional-run-advantage-audit`
- Tested SHA: `fb0208b44588e8ad5dfc8ce748d7f560bebd1fe2`
- Artifact: `rb-stack6s-conditional-run-advantage-audit`
- Artifact ID: `9856350334`
- Artifact SHA256: `5fdd8b469edcb0aae6f1f321cc8c5a27c0b960b09c26d7dc0aef8ee775fa3b54`
- Disposition: **`CONDITIONAL_ADVANTAGE_SIGNAL_NOT_QUALIFIED`**

## Integrity

- 2023-2025 offensive PBP rows: `101,636`
- 2025 team-games joined: `544`
- W6-18 team-games: `388`
- W6-18 decision plays: `23,820`
- unique target decision cells: `3,946`
- exhaustive context identity: PASS
- nflverse success source: present
- finite signal coverage: >=99% PASS
- strict-prior construction: PASS
- fitted models / feature search / model-family search / hyperparameter search / threshold search / coefficient search: `0`
- sportsbook inputs: `0`

## Frozen signal qualification

| Scheme | Signal | Full corr vs call residual | Top-bottom residual spread | W6-12 corr | W13-18 corr | Qualifies? |
|---|---|---:|---:|---:|---:|---:|
| TEAM5 | EPA run advantage | 0.021890 | 0.029480 | 0.017117 | 0.024627 | No |
| TEAM5 | Success run advantage | 0.017095 | 0.021143 | 0.016026 | 0.017168 | No |
| TEAM8 | EPA run advantage | 0.017883 | 0.018003 | 0.009698 | 0.024216 | No |
| TEAM8 | Success run advantage | 0.017508 | 0.017360 | 0.017221 | 0.016668 | No |

Frozen qualification required under both history schemes:

- full correlation >= +0.03;
- top-minus-bottom residual spread >= +0.03;
- positive W6-12 correlation;
- positive W13-18 correlation.

Neither signal passes. No threshold is waived.

## Durable conclusion

The conditional designed-run decision identified by STACK6R is not sufficiently explained by a simple strict-prior offense-plus-opponent **run-versus-pass EPA advantage** or **run-versus-pass success-rate advantage** in the same score-state × down/distance cell.

The EPA signal is directionally positive and nearly reaches the three-percentage-point spread gate under TEAM5, but it is too weak overall and degrades under TEAM8. That is evidence against fitting/tuning a conditional efficiency-edge model from this family.

Do not respond by adding generic run/pass EPA, PROE, pressure, or matchup features to STACK6Q; those broad concepts were already tested in earlier migrations and the exact conditional efficiency formulation here did not qualify.

The next high-value structural question is whether the remaining conditional-call error is actually missing **richer target-game football state**—especially field position and game phase—rather than pregame matchup preference. A no-fit hierarchical context oracle should answer that before another fitted model is attempted.