# RB Individual Mechanism Decomposition — Result

## Canonical run
- Run: `34065409969`
- Tested SHA: `c3695e1f9f3218a67816eea8d84b8cb293ceeb66`
- Artifact: `9998778424`
- Artifact SHA256: `664cd4c5f8842120236ba80ebc854c92561141641f34f0d8293917e1411a8646`
- Rows: 1,393; players: 148; qualifying profiles (>=8 games): 86
- Decomposition max identity error: `3.552713678800501e-14`
- Sportsbook/model fitting/production change: none

## Aggregate mechanism
- Rushing-yard MAE: 20.424
- Mean absolute carry contribution: 14.215 yards
- Mean absolute YPC contribution: 12.784 yards

Among 86 qualifying players:
- **CARRIES: 35**
- **MIXED: 26**
- **YPC: 25**

The RB problem is therefore heterogeneous; no single carry-allocation correction can reasonably solve every player.

Important examples:
- James Cook: 46.53 yard MAE, MIXED.
- Derrick Henry: 44.46, MIXED.
- Bijan Robinson: 42.81, YPC.
- Jahmyr Gibbs: 41.40, YPC.
- Kimani Vidal: 38.68, CARRIES.
- Rico Dowdle: 36.84, CARRIES.
- Jonathan Taylor: 35.84, YPC.
- De'Von Achane: 34.35, YPC.
- Christian McCaffrey: 33.69, CARRIES.

State slices:
- rookie: carry component 14.96 vs YPC 10.93
- mismatch+rookie: carry component 16.86 vs YPC 11.29
- current depth RB1: carry component 16.00 vs YPC 16.44 (balanced)

## Official disposition
**`RB_INDIVIDUAL_MECHANISMS_MAPPED`**

This explains why the blunt current-depth role remap was rejected: current depth rank cannot simply replace usage history across all RBs. Future work should target carry-dominant and YPC-dominant mechanisms separately, with special attention to new/limited-history role situations for carries.
