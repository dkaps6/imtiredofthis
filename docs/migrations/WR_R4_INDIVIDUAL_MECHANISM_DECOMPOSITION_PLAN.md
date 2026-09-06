# WR-R4 Individual Mechanism Decomposition — Frozen Plan

## Purpose
Use the six-season M38 casebook to determine whether each WR's receiving-yard misses are primarily reception-volume misses or yards-per-reception misses. This follows the individual-MAE strategy without applying player-specific offsets.

## Canonical evidence
- WR-R1 run `34058453941` exact paired casebook.
- Exact M38 WR rows, seasons 2020-2025.
- receiving-yards and receptions markets only.
- Sportsbook data prohibited.

## Exact decomposition
For each season/week/team/player, pair M38 receiving yards with M38 receptions.
Define actual/projected YPR as yards divided by receptions when receptions >0; define YPR=0 only when both yards and receptions are zero. Any inconsistent zero-reception/nonzero-yard row is excluded and counted.

Use two-factor Shapley decomposition of `actual_rec_yards - projected_rec_yards`:
- reception contribution = `(actual_receptions - projected_receptions) * (actual_ypr + projected_ypr) / 2`
- YPR contribution = `(actual_ypr - projected_ypr) * (actual_receptions + projected_receptions) / 2`

The two contributions must reconstruct the receiving-yard residual within 1e-6.

## Player profiles
For each WR with >=8 scoreable games report:
- receiving-yard MAE/bias and 20+/30+/40+ miss rates
- reception MAE/bias
- mean and mean-absolute reception contribution
- mean and mean-absolute YPR contribution
- contribution shares
- dominant mechanism: `RECEPTIONS` if mean-absolute reception contribution >=1.25x YPR; `YPR` if reverse; otherwise `MIXED`.

Report aggregate/season summaries, high-MAE players, and counts by dominant mechanism.

## Integrity
- exact M38 WR receiving-yard source count 12,396 before pairing.
- seasons 2020-2025.
- decomposition identity <=1e-6 on scoreable rows.
- no fitting, sportsbook, or production change.

Disposition: `WR_INDIVIDUAL_MECHANISMS_MAPPED`. The result guides the next genuinely mechanistic WR hypothesis; it does not itself change M38.
