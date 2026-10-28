# PRISM — the roster existence ceiling is system-wide, but its materiality is RB-concentrated

Approved as diagnostic-only by GPT audit
[#535 comment 6479393261](https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6479393261)
item C. Companion to `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`, which established that
**Ourlads is the sole authority on player existence** (W7 chain: `roles_ourlads` 458 →
`roles_ourlads_status_v1` 458 → `current_player_availability` 458 →
`roles_current_production_eligible_v1` 396 → `football_simulation_universe` 396 — the
availability layer never adds a row).

**Question:** is that ceiling an RB problem or a system-wide one?
**Answer:** system-wide, and it costs ~4% of targets versus ~7% of carries — but the
*shape* of the loss differs in a way that matters more than the magnitude.

## 1. Receiving, 2026 week 7, the exact Thursday universe

| class | players | targets | receptions | yards |
|---|---|---|---|---|
| modeled | 196 | 775 | 529 | 6,031 |
| **absent** | **17** | **34** | 23 | 209 |
| suffix-only join gap | 4 | 8 | 5 | — |

- **absent share of all targets: 4.16%**
- team-level outside-universe target fraction: mean **4.41%**, median **0.0**,
  p90 9.4%, p95 19.5%, max **30.3%**, with **15/27 teams at zero**

Absent by position: **RB 6 players / 13 targets**, WR 5 / 11, TE 6 / 10.

## 2. The comparison that answers the question

| market | absent share | teams at zero | worst single absent player |
|---|---|---|---|
| rushing | **7.41%** of carries | 18/27 | **Jordan Mason, 15 carries** (a lead back) |
| receiving | **4.16%** of targets | 15/27 | Marquise Brown, 5 targets (rotational) |

The ceiling is **real in both markets** — so it is a structural property of the
architecture, not an RB quirk, exactly as the audit predicted. But rushing is ~1.8× worse
by volume **and categorically worse by role**: the rushing absences included a back who
took 15 carries, while **no absent receiver held a feature role.** Receiving absences are
peripheral; one rushing absence was a team's primary ball-carrier.

So: **fix the architecture once (it is one bug), but expect the measurable projection
benefit to land mostly on RB.** This does not license deprioritizing the receiving side —
a max of 30.3% on one team is not peripheral for that team.

## 3. State labels — the dominant class is the fixable one

| state | players | targets |
|---|---|---|
| **`PROVIDER_OMISSION_was_active_earlier`** | **14** | **28** |
| `LATE_ADDITION_no_prior_week` | 1 | 4 |
| `PRACTICE_SQUAD_ELEVATION` | 1 | 1 |
| `RETURN_FROM_RESERVE` | 1 | 1 |

**14 of 17 absent receivers were already `ACT` in an earlier week.** As with rushing, the
dominant class is deterministic provider omission, not genuine pregame roster uncertainty.
Top cases: Marquise Brown (PHI, 5 targets), Will Mallory (IND, 3), Nick Muse (ATL, 2),
Brandon Smith (PIT, 2), Andrew Ogletree (IND, 2), Andrew Armstrong (KC, 2),
Justin Watson (KC, 1).

Six players appear on **both** the rushing and receiving absent lists — Mason, Coleman,
Sampson, Carter, Herbert, Mitchell — consistent across the two measurements.

**Timing caveat, per the audit:** state labels come from `load_rosters_weekly`, and the
current-season roster asset is **mutable**. Today's file is strong evidence of what was
knowable on Thursday; it is **not** a contemporaneous snapshot. Labels are evidence, not
proof. This caveat also applies retroactively to the nine rushing labels in the companion
doc.

## 4. Correction to the audit: the Mike Washington case is NOT a production existence defect

The audit called it "a pure identity bug" in the repo and recommended fixing it
independently. The second half is right; the attribution is not.

**The player was modeled.** The universe carries him as `Mike Washington` →
`mikewashington`. nflverse calls him `Mike Washington Jr.` → `mikewashingtonjr`. The defect
is that **`canonicalize_player_name_safe` does not strip name suffixes**, so the two never
join. It is a *canonicalizer* gap exposed by a measurement join, not a roster gap.

Sized: of 489 players with carries or targets in 2026 weeks 1–7, **29 (5.9%) carry a
suffix, and all 29 retain it in the clean key** (342 carries, 362 targets at stake *if*
the other side of a join spells them without it). At week 7 the realised mismatch is
**4 players** — `Kevin Coleman Jr.`, `Mike Washington Jr.`, `Omar Cooper Jr.`,
`Ted Hurst III` — all present in both the universe and `roles_ourlads` under non-suffixed
spellings.

**This matters beyond the diagnostic.** Any nflverse→production join on `player_clean_key`
silently drops these players, which includes **grading joins**. The 0.98% of targets lost
to the suffix gap is a measurement artifact, not a football fact, and it inflated the
rushing out-of-universe figure by one player (Washington, 6 carries) in the companion doc.

## 5. Corrections accepted from the audit

**Stacked-box terminology.** My framing — that
`percent_attempts_gte_eight_defenders` "partially restores a documented dead end" — is
**too strong and is withdrawn**. With strictly-prior values it is a **historical
player-exposure/response signal** (how often *this player* previously ran into 8+ boxes).
It is **not** a current-week opponent box tendency and does **not** replace the unavailable
2026 participation field `defenders_in_box`. A valid hypothesis would be *"prior player
stacked-box exposure plus prior RYOE response carries stable player-specific information
beyond current PRISM YPC/role inputs"* — which must earn persistence and walk-forward value
first.

**M75 does not close the RB NGS question.** The audit read the code: M75 tested NGS
*receiving* tracking aggregated to team-week, plus PFR secondary coverage, against **QB
YPA residual**. It did not test NGS rushing, expected rush yards, RYOE, time-to-LOS,
player-level stacked-box exposure, or any named-player projection. Its null is **narrow**.
NGS RB stays **QUEUED / UNTESTED**, not closed — and still behind roster integrity and
budget reconciliation.

**Existence source.** `load_rosters_weekly` is approved as a production existence-union
candidate (NFL Shield-derived, nflverse refreshes daily 07:00 UTC) — **prospectively safe
when fetched at run time with bytes/timestamp/digest archived**, and explicitly *not* a
post-hoc historical oracle. A 07:00 UTC refresh is not real-time, so the contract also
needs a source-freshness timestamp, a game-day refresh, official inactives, and an explicit
unresolved-late-transaction state whose mass is **not** silently redistributed.

## 6. Scope

One week, 27 teams, one preserved pregame universe. Nothing retuned; no target share, no
coefficient, no cap, no weight changed. Per the audit, this result must **not** be used to
retune target shares. No sportsbook variable in any measurement. Betting remains
**CLOSED**. No odds fetch.
