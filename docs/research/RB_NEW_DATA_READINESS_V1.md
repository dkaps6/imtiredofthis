# RB New-Data Readiness Map V1 — opponent/teammate injuries + route volume

**STATUS: INVENTORY AND SOURCE-READINESS AUDIT ONLY.**
No candidate, no model fit, no threshold search, no evaluation against exposed
2025 outcomes, no production change, no mutation of the frozen PD2
forward-confirmation study. No performance claim is made anywhere in this
document.

Authority: `CURRENT_NFL_RESEARCH_HANDOFF.md` → "STANDING PROHIBITION —
RETROSPECTIVE RB RUSHING RESEARCH IS CLOSED". M96E closed the retrospective
line; the two sanctioned continuations are prospective 2026 evidence and a
separately justified new-data source. This audit scopes the second, and is
deliberately confined to *what exists and what is wired*, not to whether any of
it predicts anything.

Base: `main@76e5e0452a88018b95ebc212fb0d1b21a2ca90a3`.

---

## 0. Headline

The most consequential finding is not a missing data source. It is that **the
production model has no mechanism by which any teammate's absence can raise
another back's carries.** The data to drive one is already present, live, and
keyed — it is unused for rushing, not unavailable.

Separately, the obvious route-reconstruction path is a trap: nflverse
participation data **stops at 2025** and cannot be loaded for 2026, so any route
feature built on it would be unusable in live production. Snap counts are the
live-viable proxy.

---

## 1. Opponent / teammate injury propagation

### 1.1 What the repo already has

| Surface | Path | Live 2026? | Notes |
|---|---|---|---|
| nflverse injuries | `nfl.load_injuries` | **yes** | 2026 wk1–2: 433 rows, 34 RB/FB |
| weekly injury artifact | `data/injuries_weekly.csv` (`scripts/build/build_injuries_weekly.py`) | yes | production build |
| slate injury artifact | `data/injuries.csv` | yes | consumed by context bridges |
| current availability | `scripts/build/build_current_player_availability_v1.py` | yes | certification path |
| preserved-scope audit | `data/preserved_replay_injury_scope_audit.json` | yes | replay provenance |

nflverse injury schema carries `gsis_id`, `position`, `report_status`
(`Out` / `Doubtful` / `Questionable`) **and** `practice_status` /
`practice_primary_injury`. 2026 week 1–2 `report_status` distribution:
`Out` 75, `Questionable` 79, `Doubtful` 12.

Two properties matter for later use and are worth stating explicitly:

- **`gsis_id` is present**, which is the same identity the GSIS grader resolves
  to. Join quality is therefore high and does not depend on name canonicalization.
- **`practice_status` is distinct from `report_status`.** Practice participation
  is observable mid-week, well before the final game-status designation. That
  matters for any pregame-safe contract.

### 1.2 Where the pipeline drops it — the actual defect

Injury information reaches only `scripts/modeling/context_bridge.py` /
`context_bridge_v3.py`, and is applied in
`scripts/modeling/simulation_rules.py`. Its entire effect is two rules:

**Rule 1 — `_injury_target_overrides` (lines 70–104).** Fires *only* when a
team's **WR1** is injury-limited, and redistributes that WR1's **target share**
to WR1_5 (60%), SLOT/TE (30%), RB/FB (10%).

**Rule 2 — self-haircut (lines 200–203).** If the player himself is
injury-limited, `base_tgt *= 0.50` and `base_rush *= 0.50`.

Every write to `base_rush` in that module:

```
160:  base_rush = _num(row.get("bayes_rush_share", row.get("rush_share", ...)))
202:      if np.isfinite(base_rush):
203:          base_rush *= 0.50
208:  out.at[idx, "rules_rush_share"] = base_rush
```

A repo-wide search for any injury path touching rushing
(`grep -rn "injur" scripts/modeling/*.py | grep -iE "rush|carr|attempt"`)
returns **nothing**.

Consequences, both in the same direction:

1. **An `OUT` RB1 is still projected to take 50% of his normal carries.**
   `_injury_limited()` matches `OUT`/`DOUBTFUL`/`IR`/`PUP` and halves his share
   rather than zeroing it. A player who will take zero carries retains half.
2. **No other back gains anything.** There is no rush-share redistribution of
   any kind. RB2's projected carries are identical whether RB1 is healthy or
   ruled out.

The team's carries are then allocated by `_top_n_shares(raw_rush_shares, 5)` in
`scripts/simulation_v2.py` using those unadjusted shares.

### 1.3 Why this is the relevant gap

Per the now-canonical M96A regime evidence (cited as motivation only, not
re-derived or re-tested): the 20+ and 25+ actual-carry slices are the largest
**opportunity** recovery bands, at `11.369` and `11.762` yards respectively,
while the 11–19 middle is efficiency-dominated.

A back reaching 20+ carries is the workload-spike case, and the single most
legible pregame driver of a spike is the other back being unavailable. The
model currently cannot represent that, in either direction.

### 1.4 Missing vs merely unused

**Merely unused** (data present, live, keyed):
- teammate `report_status` / `practice_status` at RB, QB, FB/TE;
- backfield-room composition implied by those statuses;
- the distinction between `Out` and `Questionable`.

**Genuinely missing**:
- **OL availability as a unit.** Injuries carry individual OL players, but the
  repo has no line-continuity construct. Note the OL/front-cohesion lane is in
  the forbidden family and this is *not* a proposal to reopen it — recorded only
  so the inventory is complete.
- **Official inactives at T-90.** `report_status` is the Friday designation;
  the inactive list is the ground truth and lands ~90 minutes pregame. Existing
  surfaces: `espn-official-inactives-v1` branch,
  `scripts/backtest/audit_qb_official_inactive_availability_v3.py`.

### 1.5 Leakage-safe timestamp contract (design note only)

Any future use must record, per row: the source observation timestamp, the
designation as of that timestamp, and the kickoff from
`scripts/build/_schedule_utils.get_nfl_schedule(season)` → `kickoff_utc`
(**not** `data/team_week_map.csv`, whose `gameday` is date-only midnight UTC —
see issue #535 comment 5754153538). Admissible only when
`observation_timestamp < kickoff_utc`, fail closed otherwise.

`practice_status` and `report_status` are mid-week/Friday observations and are
naturally pregame. An inactive list is pregame but late, and would need its own
T-90 cutoff contract.

---

## 2. Routes run / route volume

### 2.1 The trap

Empirically tested against nflreadpy on this branch:

| Source | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---|
| `load_participation` | 46,168 | 45,919 | 45,184 | **UNAVAILABLE** |
| `load_snap_counts` | 26,540 | 26,615 | 26,612 | **2,901** |

`load_participation(seasons=[2026])` raises
`ValueError: Season must be between 2016 and 2025`.

**Participation is historically rich and prospectively dead.** Any route or
route-share feature reconstructed from participation could be built and
validated on 2016–2025 and then could not be computed for a single live 2026
slate. Recording this prominently because it is exactly the kind of lane that
looks fundable on historical coverage and fails only at deployment.

### 2.2 What is live-viable

`load_snap_counts` **does** cover 2026 (weeks 1–2 present; 211 RB/FB rows, 117
players, all 32 teams). Schema: `game_id, pfr_game_id, season, game_type, week,
player, pfr_player_id, position, team, opponent, offense_snaps, offense_pct,
defense_snaps, defense_pct, st_snaps, st_pct`.

Two caveats stated plainly:

- Snap counts are a **postgame** artifact. Pregame use means prior-week snaps,
  which is leakage-safe but lagged.
- Identity is `pfr_player_id` / `player` name — **not `gsis_id`**. Unlike the
  injury surface, this needs the canonical resolver, and the `LA`/`LAR`-class
  team-namespace issue applies.

Snap share is a coarser construct than routes run: it does not separate pass
snaps from run snaps, so it is an opportunity proxy, not a route measure.

`load_pbp(seasons=[2026])` returns 5,334 rows for weeks 1–2 but has **no
personnel columns**, so per-play personnel cannot be recovered from PBP alone in
2026.

### 2.3 Does live PlayerForm lack routes?

Yes. No routes-run or route-participation field exists in the live build path.
`load_participation` and `load_snap_counts` appear **only** under
`scripts/backtest/` (research), never in `scripts/build/` or the `run_*`
production entry points.

So the answer to "is it missing or merely unwired" differs by field: routes are
**missing prospectively** (no 2026 source at any price from nflverse), whereas
snap counts are **present but unwired**.

---

## 3. Readiness matrix

| Field | Historical | Live 2026 | Pregame-safe | Join key | Cost | Current repo path | Missing work |
|---|---|---|---|---|---|---|---|
| teammate `report_status` | 2024–26 | **yes** | yes (Fri) | `gsis_id` | free | `build_injuries_weekly.py` | rush-share redistribution does not exist |
| teammate `practice_status` | 2024–26 | **yes** | yes (mid-week) | `gsis_id` | free | same artifact, unused | not surfaced to modeling |
| backfield-room vacancy state | derivable | **yes** | yes | `gsis_id` | free | none | construct does not exist |
| official inactives (T-90) | partial | needs check | yes, late | name/team | free | `espn-official-inactives-v1` | not in live path; own cutoff contract |
| prior-week snap share | 2016–26 | **yes** | yes (lagged) | `pfr_player_id`/name | free | backtest only | not wired; identity resolver needed |
| routes run | 2016–**2025** | **NO** | n/a | n/a | n/a | backtest only | **no 2026 source — external purchase only** |
| per-play personnel | 2016–2025 | **NO** | n/a | n/a | n/a | backtest only | same limitation |
| OL continuity | individual only | individual only | yes | `gsis_id` | free | none | no unit construct (forbidden family — noted, not proposed) |

---

## 4. Minimal integration plan (no fitting, no evaluation)

Ordered by ratio of enabling value to work, smallest first. Each is plumbing
only; none is a candidate, and none may be evaluated before the first valid
prospective lock exists.

1. **Surface teammate availability into the modeling frame.** Injury rows
   already load; `PlayerContext.features` already carries `injury_status`.
   The missing piece is a per-team backfield-room view so a rule *could*
   reference "RB1 is Out" when computing RB2's share. No rule is proposed here.
2. **Correct the `OUT` self-haircut.** A player with `report_status == Out` who
   is confirmed inactive retains 50% of his rush share today. This is arguably a
   production correctness bug independent of any new-data lane, and should be
   raised on its own merits rather than bundled into research.
3. **Wire prior-week snap share into PlayerForm** as a descriptive field, via
   `canonicalize_player_name_safe` plus `canon_team`, with a strictly-prior week
   filter. Descriptive only.
4. **Record an availability observation timestamp** on injury rows so the
   §1.5 contract is satisfiable later.

Explicitly out of scope here: any weighting, any threshold, any
redistribution coefficient, any backtest.

---

## 5. What may need to be obtained externally

One item, and only one:

**Routes run / route participation for 2026+.** nflverse participation ends at
2025 and there is no free substitute that provides per-play personnel for the
current season. If route volume is wanted as a live feature, it requires a
commercial source (e.g. a charting provider offering routes-run by player-week
with a current-season feed). Everything else in this audit is already owned and
free.

Before any purchase, note that §1 is the larger and cheaper gap: the injury
mechanism is fully owned, live, and keyed, and is simply not connected to
rushing.

---

## 6. Compliance

- No RB candidate, threshold search, router, or outcome-tuned evaluation.
- No reuse of exposed 2025 outcomes for optimization; the only historical
  numbers cited are M96A's canonical figures, used as motivation for *where*
  information could matter, not re-derived or re-tested.
- No production science, code, workflow or data change; this commit adds one
  document.
- Frozen PD2 forward-confirmation study untouched.
- CLV lane not opened.
