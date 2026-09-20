# BDB2026 Receiver Release Geometry Qualification V1 — Result

**Status:** CLOSED — ENGINEERING READY / SOURCE THIN  
**Frozen plan:** `docs/research/BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1.md`  
**Implementation commit:** `c91c0f78c2b1dc1fb5ba2c5af60172b629dde2a6`  
**Canonical run:** `35513387991`  
**Job:** `106085119188`  
**Artifact:** `10606360377`  
**Artifact digest:** `sha256:4a34f88c204334d7f255cbea1d0fe59ab1854ef2e0fd7429d6404c2f5c89c88a`  
**Certified materializer SHA:** `47bcd58aecf453f54b3f5db06a9dbdc94b000ad2`  
**BDB2026 source hash:** `228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`

## Final qualification verdict

Both frozen candidates are valid, stable, identity-safe and clearly incremental versus
canonical PlayerForm target-opportunity state, but **neither clears the frozen 80%
broad pregame-coverage gate**.

Final disposition for both:

`ENGINEERING_READY_SOURCE_THIN`

No predictive receiving-yards/receptions experiment is authorized from V1.

## Candidate results

| Candidate | Broad 2023 rows | Pregame coverage | Stability Spearman | Stability profiles | Holdout reconstructibility R2 | Redundancy | Disposition |
|---|---:|---:|---:|---:|---:|---|---|
| nearest defender release history | 4,631 | **59.7063%** | **0.704979** | 188 | **0.241973** | incremental information survives | `ENGINEERING_READY_SOURCE_THIN` |
| second defender release history | 4,631 | **59.7063%** | **0.614016** | 188 | **0.109448** | incremental information survives | `ENGINEERING_READY_SOURCE_THIN` |

Both candidates had median strict-prior support of **23** observations when present.

Outcome-free reconstruction used 2023 Weeks 1-9 as train and Weeks 10-18 as holdout:
- train rows: **977**
- holdout rows: **1,788**

Production-blend Spearman correlations were:
- nearest release spacing: **-0.435364**
- second-defender spacing: **-0.229011**

These are descriptive novelty diagnostics only.

## Coverage pattern

The broad denominator retained Week 1 and all canonical 2023 WR/TE/RB player-games.

Examples:
- Week 1: **0%** for every position
- Week 8 WR: **82.54%**
- Week 14 WR: **90.83%**
- Week 16 WR: **85.82%**
- Week 14 TE: **80.36%**
- Week 16 TE: **84.38%**
- RB remained below 80% even late in the season (Week 16: **73.68%**)

This late-season behavior is **not** a rescue. V1 froze broad-season coverage >=80%.

## Identity / integrity

Direct stable identity:
- BDB targeted-receiver IDs: **464**
- mapped directly to canonical GSIS identity: **464**
- direct stable-ID bridge coverage: **1.0000**
- ambiguous BDB nfl_id mappings: **0**
- ambiguous canonical GSIS mappings: **0**
- name fallback used: **false**

Canonical broad universe:
- eligible players: **467**
- eligible player-games: **4,631**
- stable identity coverage: **1.0000**
- duplicate canonical keys: **0**
- join fanout: **0**
- Week-1 rows retained: **266**

Temporal/materializer integrity:
- receiver-history rows checked: **7,161**
- chronology violations: **0**
- materializer support-threshold violations: **0**
- mapped snapshot duplicate rows: **0**
- target-game rows used in pregame history: **0**
- landing/post-release fields used pregame: **0**

Semantic firewall:
- nearest defender was **not** relabeled as coverage responsibility.

## Artifact/input lineage

- canonical 2022-2023 history SHA-256:
  `924d6cd399e05cb9261bb47ebea83fc35ae089411381e79f755512d9ec029042`
- nflverse player crosswalk SHA-256:
  `35a60c7d63dee4e78c8085ebb933b0b344f5d67b726f05defb1179c1bd0b9be5`
- BDB2026 raw geometry ephemeral SHA-256:
  `dbdfc1cebb95d8c0cd31b2be71311fb7289c5d3db643c86bab3ed43e342340fd`
- BDB2026 receiver-history snapshot SHA-256:
  `e60882727a03ba14ff9e43d5229d6d491d8833a76f91d9f6412f313c89993efe`

## Interpretation

This is useful positive engineering evidence and a negative qualification result.

The receiver release-spacing histories are:
- real;
- persistent;
- identity-safe;
- strict-prior;
- not highly reconstructible from existing target-opportunity inputs.

But the public 2023 BDB slice does not provide enough broad-season historical support
for the frozen V1 deployment gate.

Do not rescue with:
- late-season-only cohorts;
- WR-only qualification after seeing coverage;
- a lower support threshold than 8;
- a lower broad coverage floor;
- route-conditioned target-game route truth;
- crowding fields added after this result.

Final disposition:

`BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1_SOURCE_THIN_CLOSED`

## Next distinct mechanism

Move to the separately motivated BDB2023 blocker/protection geometry family under the
existing OL/DL protection-context contract.

That next lane remains outcome-free qualification first and must preserve:
- interaction != universal assignment;
- strict-prior blocker histories only;
- no PFF target-game outcome labels as predictors;
- no sportsbook input;
- no production change;
- no Issue #535 interference.
