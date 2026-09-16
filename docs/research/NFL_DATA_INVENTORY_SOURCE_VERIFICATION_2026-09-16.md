# NFL Data Inventory — Source Verification and QA Addendum

**Status:** documentation-only QA companion to `docs/research/NFL_DATA_INVENTORY_AND_NEW_INFORMATION_FRONTIER.md`.

**Audit base:** `91afb3a51cda316467cd8431fa5243a28195c1b0`.

**Scope boundary:** no predictive experiment, no model retuning, no production-science change, no purchase, no film charting, no CV implementation, and no Issue #535 scientific direction.

This addendum records the final external-source verification pass, resolves ambiguous Big Data Bowl season wording, and provides a compact reproducibility/accounting layer for the canonical inventory.

---

## 1. Inventory accounting

Sections 3.1-3.5 of the canonical inventory contain **49 family-level inventory rows**:

- 16 core game/player/environment families;
- 9 team-context/defense/participation families;
- 10 receiver/coverage families;
- 4 FTN tactical/decision/error families;
- 10 opportunity/entitlement/research-state families.

This is intentionally a **family count**, not a raw-column count. The M82 ledger's 27 research families overlap substantially with these inventory families and therefore should not be added to 49 as if they were separate raw datasets.

The repo is already broad enough that a new-data project should be presumed duplicative until it demonstrates richer grain: route, frame, assignment, responsibility, contact, or another relationship-level concept not already represented.

---

## 2. Big Data Bowl source contract — verified competition-by-competition

### 2020 — rushing / handoff geometry

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2020`

- Theme: predict rushing-yard distribution after a handoff.
- Public training lineage centered on 2017-2018 rushing tracking examples; the code competition later reran against future 2019 games.
- The public competition data are best thought of as a **handoff-state/snapshot geometry** resource, not a complete route/contact tracking archive.
- Use here: development reference for spatial run-state features.
- Do not claim: full multi-season live NGS tracking contract.

### 2021 — pass coverage

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2021`

- Data: 2018 regular-season passing plays.
- Focus: pass coverage.
- Public tracking includes player/ball trajectories and event timing for eligible passing plays; linemen were excluded from this competition slice.
- Use here: receiver/defender trajectory, separation, target-agnostic coverage and route-geometry development/validation.
- Do not claim: complete historical defender responsibility or a current weekly feed.

### 2022 — special teams

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2022`

- Theme: special-teams performance.
- Public competition package contains NGS tracking, play, game, player and PFF scouting data for **2018-2020 special-teams plays**.
- Competition hosts explicitly noted that plays with tracking-data issues were removed, so this is a cleaned competition corpus rather than an exhaustive play ledger.
- Use here: tracking/CV validation, pursuit/space-control methods, special-teams interaction labels.
- Direct relevance to offensive prop modeling is secondary.

### 2023 — pass protection / line play

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2023`

- Theme: evaluate linemen on pass plays.
- Public slice: **Weeks 1-8 of the 2021 season**, dropback pass plays.
- Tracking: player location/speed/acceleration/direction from snap until pass release.
- PFF scouting data supply responsibility/context unavailable in ordinary nflverse PBP.
- Use here: primary public ground truth for pass-rush/protection geometry and responsibility concepts.
- Competition-host guidance explicitly states that broader player/ball tracking is not publicly available as an extensible feed; previous BDB competitions are the public slices.

### 2024 — tackling

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2024`

- Theme: tackling.
- Public slice: **Weeks 1-9 of the 2022 NFL season**.
- Tracking includes all 22 players plus football location, speed and acceleration.
- `tackles.csv` provides tackle, assist and forced-fumble indicators plus PFF's missed-tackle label.
- Tracking rows include `x`, `y`, `s`, `a`, `dis`, orientation, direction and event tags.
- The competition data page notes event-window filtering by play type rather than unrestricted full-game frame history.
- Use here: best public truth slice for first-contact/tackle detection, closing-angle, pursuit and yards-after-contact extraction QA.

### 2025 — pre-snap tendencies plus unusually rich player-play semantics

Official competition: `https://www.kaggle.com/competitions/nfl-big-data-bowl-2025`

- Theme: use pre-snap behavior to understand team/player tendencies.
- The package contains NGS frame tracking plus a `player_play` table with unusually valuable PFF/NGS semantic labels.
- Verified examples include:
  - `inMotionAtBallSnap`;
  - `shiftSinceLineset`;
  - `motionSinceLineset`;
  - `wasRunningRoute`;
  - `routeRan`;
  - `blockedPlayerNFLId1/2/3`;
  - `pressureAllowedAsBlocker`;
  - `timeToPressureAllowedAsBlocker`;
  - `pff_defensiveCoverageAssignment`;
  - `pff_primaryDefensiveCoverageMatchupNflId`;
  - `pff_secondaryDefensiveCoverageMatchupNflId`.
- Tracking rows also expose player/ball IDs, frame/time, club, play direction, x/y, speed, acceleration, distance, orientation, movement direction and event tags.
- The public corpus is a **competition slice from the 2022-season tracking family**, not evidence of a complete free historical/current feed.
- Use here: strongest public semantic-label truth set for route classification, motion, coverage responsibility and blocker-rusher relationships.

### 2026 — ball-in-air movement, prediction + analytics

Official competitions:

- `https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction`
- `https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics`

**Correction/clarification to the canonical inventory wording:**

- The **published prediction training files are `input_2023_w[01-18].csv` / corresponding 2023 output files**.
- The prediction competition stops the observed input at pass release and asks models to forecast player x/y movement while the ball is in the air.
- Inputs include player identity/physical attributes, side/role, x/y, speed, acceleration, orientation/direction, number of output frames, and ball landing x/y.
- The competition's live forecasting leaderboard was based on previously unseen games from the **last five weeks of the 2025 regular season**.
- Therefore the safe description is: **published 2023 historical training + later unseen/live forecasting games**, not "2023-2024 training data" as a single homogeneous public training contract.

Use here: trajectory-model/CV validation and a useful demonstration that target/defender movement after release can be represented as structured frame-level learning targets.

---

## 3. Big Data Bowl classification for this project

| Use | Classification | Why |
|---|---|---|
| Direct historical pregame feature source | **Limited / slice-only** | Competition datasets cover selected seasons/themes and are not an extensible weekly feed. |
| Live 2026 production source | **No** | No public complete current tracking/responsibility contract. |
| Algorithm development | **Excellent** | True x/y trajectories, events and player identities exist in multiple competitions. |
| CV calibration / homography validation | **Excellent** | Tracking coordinates can serve as coordinate truth on matching video where lawful access exists. |
| Route-classification ground truth | **Strong** | 2025 exposes `routeRan` / `wasRunningRoute`. |
| Coverage-responsibility ground truth | **Strong but slice-limited** | 2025 exposes PFF coverage assignment and primary/secondary matchup IDs. |
| Blocker-rusher ground truth | **Strong but slice-limited** | 2023/2025 contain pass-protection responsibility concepts and exact blocked-player IDs. |
| Tackle/contact ground truth | **Strong but slice-limited** | 2024 includes NGS trajectories plus PFF missed-tackle labels. |
| Production-history substitute | **No** | Theme/slice filtering, removed bad tracking plays, licensing and lack of live update contract prevent that claim. |

---

## 4. Licensing/availability caution

Do not assume that "public on Kaggle" means unrestricted commercial reuse.

- 2024 competition rules/data are published under **CC BY-NC 4.0**.
- 2026 prediction data are published under **CC BY-NC 4.0**.
- Other competition pages may carry competition-specific access/rule language.
- Any internal proprietary-data program should separately verify rights for storage, model training, derivative labels, video use and commercial deployment before scaling.

This addendum makes no legal conclusion; it only prevents the engineering plan from treating competition access as equivalent to unrestricted production licensing.

---

## 5. Economic decision framework

Public enterprise pricing for the exact NFL assignment/tracking products relevant here is generally not sufficient to make a responsible fixed-dollar comparison from public pages alone. The repo should therefore compare vendor quotes to the **full internal cost**, not merely annotation wages.

For a self-built historical charting system, annualized internal cost should include:

`video/licensing + engineering + compute/storage + detector/tracker maintenance + identity resolution + semantic human review + QA/rework + versioning/ops`

A useful workload equation is:

`review_hours = eligible_plays × average_human_review_minutes_per_play / 60`

The decisive distinction is semantic burden:

- field calibration / x-y reconstruction: relatively automatable;
- route shape from clean tracks: relatively automatable;
- first-contact/tackle candidate detection: moderately automatable;
- exact player identity through occlusion: meaningful human QA burden;
- man/match-zone coverage responsibility, bracket/help/exchange semantics: high human QA burden;
- run concept, intended gap and blocking responsibility: very high semantic burden.

Therefore:

- **BUY** exact charted semantics when a vendor offers broad history + current update contract + usable IDs + acceptable rights at a cost below internal semantic-review operations.
- **BUILD** deterministic derivatives from trustworthy raw tracks/labels.
- **HYBRID** when raw/partial automation is cheap but semantic assignment needs selective human review.
- **SKIP** anything that merely reproduces conventional aggregates already in the repo.

---

## 6. QA verdict

The external-source verification does **not** change the frontier ranking in the canonical document.

The most defensible proprietary-data targets remain:

1. route-level receiver/defender geometry + true coverage responsibility;
2. true blocker-rusher assignment + protection geometry;
3. RB first-contact/tackle/run geometry;
4. run concept / point-of-attack / run-block responsibility;
5. 2D ball/throw-window geometry as a secondary shared layer.

The main strategic conclusion remains unchanged:

> The repository does not need more conventional football statistics. It needs a small number of high-fidelity relationship-level variables that existing public/aggregate sources cannot tell us.

This addendum is documentation only and does not authorize acquisition, implementation, charting, predictive testing, production integration or any change to active research governance.
