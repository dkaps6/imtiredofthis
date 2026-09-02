# RB STACK6L — Results

## Canonical corrected run

- branch: `research-rb-stack6l-state-tendency-shapley`
- run: `33639241740`
- job: `100277828082`
- SHA: `3adc8b0c616d2a56b5005f2421431c84254ddcb1`
- artifact ID: `9850103613`
- artifact digest: `4e0eecd596fe150876f8b7780298fca3e33fbe7eb04e5ca94bfec200cef1c570`

First execution failed before scientific outputs because `pyarrow` was omitted from the `nflreadpy` Polars-to-pandas dependency path. The correction only added `pyarrow==17.0.0`; frozen subsets, Shapley math and gates were unchanged.

## Integrity

PASS.

Fresh 2025 PBP reconstruction reproduced the M94C artifact exactly:
- offensive plays max abs difference `0`
- PBP rush attempts max abs difference `0`
- state-share max differences ~`1.11e-16`

Further identities:
- W6-18 n `388`
- occupancy MAE reproduced: `5.5183819623`
- all-state-tendency MAE reproduced: `3.4503279032`
- total tendency recovery reproduced: `2.0680540592`
- Shapley values sum exactly to `2.0680540592`
- no fit/search/sportsbook inputs

## Direct single-state corrections

Starting from perfect occupancy with M94C's pregame state tendencies:

- lead-only correction: MAE `5.049003`, recovery `+0.469378`
- neutral-only correction: MAE `4.922640`, recovery `+0.595741`
- trail-only correction: MAE `4.503012`, recovery **`+1.015369`**
- all three: MAE `3.450328`, recovery `+2.068054`

## Exact overall Shapley attribution

| State | Shapley recovery | Share of total tendency recovery |
|---|---:|---:|
| Lead | 0.448592 | 21.69% |
| Neutral | 0.594556 | 28.75% |
| Trail | **1.024907** | **49.56%** |

## P3 pool overprojection >=5

| State | Shapley recovery | Share |
|---|---:|---:|
| Lead | 0.487213 | 13.87% |
| Neutral | 0.740498 | 21.08% |
| Trail | **2.285773** | **65.06%** |

## P3 pool underprojection >=5

Attribution is more balanced:
- lead `1.016447` (30.54%)
- neutral `1.150897` (34.58%)
- trail `1.161277` (34.89%)

## Absolute P3 pool miss >=5

- lead `0.753215` (22.02%)
- neutral `0.946772` (27.68%)
- trail **`1.720581` (50.30%)**

## Frozen disposition

`TRAIL_TENDENCY_DOMINANT`

Trail passed the predeclared dominance gate:
- Shapley recovery > `0.75`
- fraction > `45%`
- positive in both over- and underprojection extremes.

No production change and no predictive model authorized.

## Research meaning

The largest identified weakness in M94C's run-rate architecture is its single coarse rushing tendency for all plays where a team trails by more than three points. The effect is especially large when P3 badly overprojects the RB carry pool.

The next diagnostic should split trail plays into deficit severity and game phase before fitting: one-possession vs multi-score, and early vs late. This tests whether the coarse trail state is hiding materially different football behaviors.

P3 remains the RB point champion.
