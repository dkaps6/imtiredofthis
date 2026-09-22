# Current-Season State Persistence V1 — Result

**Status:** COMPLETE — diagnostic research only  
**Branch:** `research-current-season-state-persistence-v1`  
**Frozen plan:** `docs/research/CURRENT_SEASON_STATE_PERSISTENCE_V1_PLAN.md`  
**Canonical run:** `35741758765`  
**Run head:** `28d79153f723765b01a447c2023b47108b14138c`  
**Artifact:** `10699781744` / `current-season-state-persistence-v1`  
**Artifact digest:** `sha256:2d81bcb5222136ae812575b15a0d79952d27a03e68034cd312aa0f927503868a`  
**Sportsbook inputs:** 0  
**2026 outcomes used:** 0  
**Production changed:** 0

## Integrity

- frozen synthetic tests: PASS, 4/4;
- strict-prior target-week boundary: PASS;
- exact PlayerForm four-game pseudo-prior formula reproduced;
- stable player identity used;
- evaluation seasons: 2022-2025;
- 2022-2024 development/descriptive;
- 2025 replication;
- row-level persistence panel: 41,745 rows;
- strict repository audit: PASS;
- artifact upload: PASS.

## Core result

The user's live-season premise is supported.

Completed current-season player state contains substantial next-game information,
and the existing historical + current-season PlayerForm shrinkage generally
improves next-game prediction over a prior-season-only baseline.

Under the frozen replication rule, **9 of 10** production-aligned metrics receive
`CURRENT_SEASON_SIGNAL_REPLICATES` on 2025.

The one exception is **RB YPC**, where early/current-season efficiency does not
improve the historical baseline.

### 2025 replication

| Position | Metric | n | Prior MAE | Current-only MAE | Blend-4 MAE | Blend gain vs prior | Relative blend gain | State-delta Spearman | Frozen disposition |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| QB | YPA | 464 | 1.7386 | 1.7993 | **1.6548** | +0.0838 | **4.82%** | +0.1913 | CURRENT_SEASON_SIGNAL_REPLICATES |
| RB | rush share | 929 | 0.1486 | **0.1096** | 0.1166 | +0.0320 | **21.53%** | **+0.6256** | CURRENT_SEASON_SIGNAL_REPLICATES |
| RB | target share | 929 | 0.04611 | 0.04418 | **0.04268** | +0.00344 | **7.45%** | +0.3610 | CURRENT_SEASON_SIGNAL_REPLICATES |
| RB | YPC | 867 | **1.9135** | 2.1243 | 1.9160 | -0.00257 | -0.13% | +0.1478 | **NO_REPLICATED_CURRENT_SIGNAL** |
| TE | catch rate | 831 | 0.2598 | 0.2683 | **0.2547** | +0.00511 | 1.97% | +0.2007 | CURRENT_SEASON_SIGNAL_REPLICATES |
| TE | target share | 839 | 0.05153 | 0.04643 | **0.04553** | +0.00601 | **11.66%** | +0.4485 | CURRENT_SEASON_SIGNAL_REPLICATES |
| TE | YPT | 831 | 3.9903 | 4.1334 | **3.8234** | +0.1669 | **4.18%** | +0.2447 | CURRENT_SEASON_SIGNAL_REPLICATES |
| WR | catch rate | 1,531 | 0.2340 | 0.2501 | **0.2332** | +0.00081 | 0.35% | +0.1861 | CURRENT_SEASON_SIGNAL_REPLICATES |
| WR | target share | 1,554 | 0.06773 | 0.06352 | **0.06144** | +0.00630 | **9.30%** | +0.4083 | CURRENT_SEASON_SIGNAL_REPLICATES |
| WR | YPT | 1,531 | 5.0035 | 5.0457 | **4.7787** | +0.2248 | **4.49%** | +0.2017 | CURRENT_SEASON_SIGNAL_REPLICATES |

The strongest state persistence is therefore **opportunity / role state**, not
raw RB rushing efficiency.

## Exact two-completed-game read — the Week-3 analogue

The frozen game-count analysis isolates rows with exactly two completed
current-season games before the target game.

### 2025 two-game bucket

| Position | Metric | n | Prior MAE | Current-only MAE | Blend-4 MAE | Read |
|---|---|---:|---:|---:|---:|---|
| QB | YPA | 47 | **1.7383** | 2.1903 | 1.7983 | Two-game individual YPA is too noisy; do not overreact |
| RB | rush share | 84 | 0.1288 | 0.1158 | **0.1113** | Strong live role signal; shrinkage still helps |
| RB | target share | 84 | 0.04462 | **0.04106** | 0.04170 | Current receiving role already informative |
| RB | YPC | 77 | **2.0697** | 2.6333 | 2.1239 | Keep efficiency anchored to history |
| TE | target share | 88 | 0.03941 | 0.04000 | **0.03428** | Strong early entitlement update |
| TE | YPT | 87 | 4.1543 | 4.7033 | **4.0615** | Small benefit only when shrunk |
| WR | target share | 151 | 0.05933 | 0.05847 | **0.05451** | Strong early entitlement update |
| WR | YPT | 147 | 4.3073 | 4.9101 | **4.2505** | Small benefit only when shrunk |
| TE | catch rate | 87 | **0.2982** | 0.3498 | 0.3062 | No two-game update benefit |
| WR | catch rate | 147 | **0.2365** | 0.2746 | 0.2375 | No meaningful two-game update benefit |

The same high-level pattern appears when all 2022-2025 rows with exactly two
completed games are pooled:

- RB rush-share blend improves MAE `0.1393 -> 0.1187`;
- WR target-share blend improves `0.06047 -> 0.05496`;
- TE target-share blend improves `0.04353 -> 0.03971`;
- RB YPC does not improve;
- raw current-only YPT/catch-rate is generally too volatile this early.

## Multi-season robustness

The state signal is not a one-season artifact.

RB rush-share blend improvement versus prior-only is positive in every measured
season:

- 2022: +0.02984 MAE;
- 2023: +0.03605;
- 2024: +0.02752;
- 2025: +0.03199.

WR target-share blend improvement is also positive in all four seasons:

- 2022: +0.00753;
- 2023: +0.00713;
- 2024: +0.00958;
- 2025: +0.00630.

TE target-share blend improvement is positive in all four seasons:

- 2022: +0.00510;
- 2023: +0.00684;
- 2024: +0.00362;
- 2025: +0.00601.

QB YPA blend improvement is positive in all four full-season summaries, but the
2025 one- and two-current-game buckets are negative. This supports gradual
shrinkage rather than aggressive early-season individual-QB efficiency updates.

RB YPC is the notable non-replicating 2025 exception and should not be treated
as a reliable early-season state signal.

## 2026 snap-source audit — actionable source gap

Disposition:

`CURRENT_2026_SNAP_SOURCE_AVAILABLE_NOT_CONSUMED`

The maintained nflverse snap source already contains:

### Week 1
- rows: 1,492;
- teams: 32;
- offense-snaps non-null: 1,492;
- offense-pct non-null: 1,492.

### Week 2
- rows: 1,502;
- teams: 32;
- offense-snaps non-null: 1,502;
- offense-pct non-null: 1,502.

Total: 2,994 rows, all 32 teams, Weeks 1-2.

But the shared production snap loader used by both TE-R5P and WR-R15 is
hardcoded to:

`[2020, 2021, 2022, 2023, 2024, 2025]`

and therefore does not load 2026.

This is not a claim that the learned R5P/R15 coefficients should change.
Those models already use strict-prior snap features and their historical
semantics support same-season prior-game participation.

The next authorized step is a **frozen source-continuation/parity test** that
adds 2026 to the snap-source availability without refitting any learned model,
proves Week 1 is invariant, proves Week 2+ uses only completed prior-week snaps,
and quantifies the football-projection impact before any production change.

## Scientific interpretation

The correct live-season architecture is not "throw away history after two
weeks" and not "ignore 2026 until the sample is large."

The evidence supports:

> historical prior + aggressively updated opportunity/role state + more heavily
> shrunk efficiency state.

In particular:

- **RB rushing opportunity is highly stateful.**
- **WR and TE target entitlement are materially stateful.**
- **RB YPC is not sufficiently stateful to chase early-season noise.**
- individual QB YPA should remain heavily shrunk in the first few weeks even
  though full-season current-state blending adds value.
- WR/TE current participation should be investigated immediately because the
  production entitlement adapters currently omit available 2026 snap evidence.

## Production boundary

No production change is authorized by this result.

Any change to:
- the four-game PlayerForm pseudo-prior,
- WR-R15 / TE-R5P source seasons,
- RB allocation,
- or any position authority

requires its own frozen follow-up test.

## Disposition

`CURRENT_SEASON_STATE_PERSISTENCE_V1_COMPLETE`

Primary next action:

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1`

Secondary action after Claude's Week-2 performance grade lands:

choose the highest-error position/market at the intersection of:
1. demonstrated current-state persistence;
2. a missing live state seam;
3. deployable pregame data;
4. no closed-family retest.
