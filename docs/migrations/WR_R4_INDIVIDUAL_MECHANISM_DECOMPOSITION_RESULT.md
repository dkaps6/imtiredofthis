# WR-R4 Individual Mechanism Decomposition — Result

## Canonical run
- Run: `34065389048`
- Tested SHA: `5150d74000b8df02e9a18545f065fadfbbafc7ad`
- Artifact: `9998771784`
- Artifact SHA256: `99a59a5821632af870d5690ee45f2312315a9bc65dd3e297176dda2bebe69536`
- Exact M38 WR receiving-yard rows: 12,396
- Scoreable paired receiving-yards/receptions rows: 12,393
- Players: 478; qualifying individual profiles (>=8 games): 337
- Decomposition max identity error: `4.263256414560601e-14`
- Sportsbook/model fitting/production change: none

## Dominant individual mechanism
Among 337 qualifying WRs:
- **RECEPTIONS: 226**
- **MIXED: 83**
- **YPR: 28**

Thus recurring individual receiving-yard error is much more commonly driven by reception-volume error than by pure yards-per-reception error under the frozen 1.25x dominance rule.

Examples among high-MAE WRs:
- Puka Nacua: yard MAE 49.56; RECEPTIONS; reception-component share 0.671.
- Ja'Marr Chase: 45.95; RECEPTIONS; share 0.636.
- Tyreek Hill: 43.69; RECEPTIONS; share 0.612.
- Justin Jefferson: 43.69; RECEPTIONS; share 0.656.
- A.J. Brown: 39.11; RECEPTIONS; share 0.594.
- Jaxon Smith-Njigba: 36.90; RECEPTIONS; share 0.657.

## Official disposition
**`WR_INDIVIDUAL_MECHANISMS_MAPPED`**

This is diagnostic, not a production win. M38 remains the protected baseline. The result directs subsequent WR research toward the mechanics underneath receptions (target opportunity versus catch conversion) rather than another blanket receiving-yard correction or another target-share multiplier hunt.
