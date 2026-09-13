# RB-PD2 Multi-Season Current-Production-Route Replication V1 — Result

## Canonical run

- workflow: `RB-PD2 Multi-Season Current Route Replication V1`
- successful run: `34733536116`
- head SHA: `7c28754ab2cbb971c2de60db54d7bdb5e9409aba`
- artifact: `rb-pd2-multiseason-current-route-v1`
- artifact ID: `10310481521`
- digest: `sha256:a005eb6609e3429dc7a9a89bfd94e88f5382280377ac52ebe5d5472190630ded`
- source M95Q run: `33450395426`

The first canonical attempt (`34733432877`) failed before any scientific output because the market-pair integrity check used Pandas tuple broadcasting. The only repair replaced that comparison with an elementwise tuple predicate. Cohort, inputs, gates, history window, and hypotheses were unchanged.

## Integrity

PASS:

- M95Q disposition = `M95Q_EXPANDED_PANEL_READY`;
- 2024 M91 universe parity = PASS;
- 2024 downstream parity = PASS;
- target seasons exactly 2021-2024;
- 2025 rows = 0;
- canonical ensemble weights fit on S-1 only and applied to S;
- paired `rush_att`/`rush_yards` RB identity unique;
- walk-forward leakage violations = 0;
- sportsbook inputs used = false;
- production changed = false.

Cohort:

- parent/source rows = **5,607**;
- scoreable last8/min4 rows = **4,652**;
- players = **265**.

The GitHub artifact independently reproduced the pre-CI dry reconstruction to floating-point noise (pooled/season metrics <= ~`6.4e-14`, parent/walk-forward numeric columns <= ~`1.8e-13`).

## Frozen-gate results

| diagnostic | n | Spearman | Q4-Q1 gap | sign agreement | W5-12 gap | W13-18 gap | positive seasons | replicated |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Carry directional persistence | 4,652 | **0.182693** | **+2.487 carries** | **60.51%** | +2.774 | +2.548 | 4/4 | **PASS** |
| Carry difficulty persistence | 4,652 | **0.320695** | **+2.616 carries** | — | +2.630 | +2.805 | 4/4 | **PASS** |
| Yard directional persistence | 4,652 | **0.136874** | **+13.277 yd** | **58.08%** | +14.981 | +13.829 | 4/4 | **PASS** |
| Yard difficulty persistence | 4,652 | **0.377996** | **+18.143 yd** | — | +16.469 | +19.681 | 4/4 | **PASS** |

All four pass the original PD2 pooled gates **and** the preregistered multi-season guard.

## Season consistency

Every diagnostic has positive Spearman and positive quartile-gap direction independently in **all four** target seasons.

### Carry directional

- 2021: rho 0.164755, gap +2.421
- 2022: rho 0.186028, gap +2.665
- 2023: rho 0.185295, gap +2.464
- 2024: rho 0.182912, gap +2.360

### Carry difficulty

- 2021: rho 0.271783, gap +2.350
- 2022: rho 0.362029, gap +3.116
- 2023: rho 0.339917, gap +2.667
- 2024: rho 0.286518, gap +2.246

### Yard directional

- 2021: rho 0.113340, gap +9.933 yd
- 2022: rho 0.157706, gap +15.643 yd
- 2023: rho 0.131991, gap +12.876 yd
- 2024: rho 0.128503, gap +13.111 yd

### Yard difficulty

- 2021: rho 0.260752, gap +12.730 yd
- 2022: rho 0.416610, gap +19.982 yd
- 2023: rho 0.385503, gap +17.263 yd
- 2024: rho 0.400532, gap +20.395 yd

## Disposition

`MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED`

The original 2025 PD2 result was not a one-season artifact. Under a genuine non-2025 temporal reconstruction of the currently authorized RB rushing mean route, both signed bias persistence and player-specific difficulty persistence reproduce across 2021-2024.

Per the frozen authorization boundary:

- **carry uncertainty-width lane: UNLOCKED for a new prospectively frozen research experiment** because carry difficulty (B) replicated;
- **yard uncertainty-width lane: UNLOCKED for a new prospectively frozen research experiment** because yard difficulty (D) replicated.

This result does **not** implement any width multiplier and does **not** authorize a production change.

## Important scope boundary

`literal_pd6_p3_equivalent_blocker_resolved = false`.

This experiment intentionally uses the current production-equivalent season-long mean route. Production P3 is promoted only for Week 1; the enriched Weeks2-18 P3 path remains unpromoted. Therefore this result does **not** pretend to satisfy the separate PD5/PD6 memo's literal request for a multi-season P3-equivalent panel.

No production/model/weight/threshold change.
