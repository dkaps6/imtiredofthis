# RB Role-Order Remap V2 — Frozen Individual-Accuracy Plan

## Purpose
Test the specific production concern proven by the Week-1 audit: timestamp-safe current depth order is preserved in data but does not directly control rushing allocation. This is independent of sportsbook lines and independent of generic player-bias calibration.

## Canonical evidence
- STACK1 production-equivalent 2025 rushing baseline, run `33535308110`.
- STACK2 timestamp-safe depth metadata, run `33538770934`.
- Expected player-game rows: 1,393.
- Expected depth coverage: 0.949748743718593.
- Sportsbook data prohibited.

## Frozen candidate: ROLE_ORDER_REMAP_V2
For each team-week:
1. keep all baseline projected carry amounts unchanged as a team-level multiset;
2. among true RB/HB rows with usable pregame depth rank, sort current depth rank ascending;
3. sort those same players' baseline projected carries descending;
4. assign the larger existing carry amounts to the higher current depth slots;
5. players without usable depth rank remain unchanged;
6. exact team carry mass must be preserved;
7. candidate rushing yards = remapped carries x that player's unchanged baseline implied YPC when defined; zero-carry edge cases retain baseline yards.

No role multipliers, no new carry mass, no market matching, no tuning after results.

## Scoreboard
- carries: MAE, RMSE, bias, median AE, p90 AE, 5+/10+ miss rates
- rushing yards: MAE, RMSE, bias, median AE, p90 AE, 20+/40+ miss rates
- W1, W2-18, W13-18
- current depth RB1/RB2/RB3
- rookies, depth/order mismatches, mismatch+rookie, limited-prior-history
- player-level MAE deltas for players with >=8 games

## Frozen production-evidence gate
ALL must hold:
1. exact 1,393-row and depth-coverage parity.
2. max absolute team carry-mass difference <=1e-9.
3. overall carry MAE improves >=1.0%.
4. overall rushing-yard MAE improves >=0.5%.
5. carry RMSE does not worsen.
6. rushing-yard RMSE does not worsen.
7. absolute carry bias does not worsen by >0.10 carries.
8. carry MAE improves in W1, W2-18, and W13-18.
9. current-depth RB1 carry MAE improves.
10. mismatch+rookie carry MAE improves.
11. median qualifying-player carry-MAE delta <0.

Pass disposition: `ROLE_ORDER_REMAP_V2_FULL_STACK_AUTHORIZED`. Fail: `ROLE_ORDER_REMAP_V2_REJECTED`. No threshold/ordering variant search after results.
