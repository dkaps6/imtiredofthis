# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-26 — V2 — PLAYER IDENTITY CARD V1 BUILT, GATED, COMMITTED / ANOMALY GATE ENFORCED / TRANSMISSION STILL OPEN

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

`docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`

Restart prompt:

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`

The checkpoint this one supersedes is archived verbatim at:

`docs/handoffs/NFL_HANDOFF_2026-10-26_V2_ARCHIVED_TOP_WEEK7_FORENSIC_PLAYER_FIRST_TRANSMISSION.md`

Do not recursively read older dated handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Do not scrap/rebuild PRISM; identity/state/opponent work is additive.

Always distinguish: `SOURCE READY` / `SOURCE BLOCKED` / `PREDICTIVE CLOSED` / `QUALIFIED`.

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work:
1. read `AGENTS.md`;
2. read only this TOP checkpoint;
3. read the deep handoff;
4. fetch newest #535;
5. reconcile live `main` / PRs / Actions.

### LIVE STATE

`main` at checkpoint: `c4de3d4044e828e39d80aa7db19af0de2caf652a` (plus this handoff commit).

Active research branch, **authorized by the user and a genuine descendant of main** (merge base = main's head):

`research-prism-player-first-identity-environment-2026-10-26` @ `d93e4cca`

Newest #535 comment at checkpoint: **6423234263**

This chat's #535 record, in order:
- `6422827149` — Identity Card V1 corrected build, source audit, two findings
- `6422904132` — correction: the card did not satisfy the anomaly gate
- `6422987723` — correction: my player-level-bridge prediction was wrong
- `6423234263` — anomaly gate enforced, incl. a 22% false positive in my own gate

**No additional paid OddsAPI call is authorized.** Betting remains CLOSED.

Canonical paid Week-7 board remains run `39849371821`, artifact `12456298985`, digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`, durable under `data/market_track_record/`.

Carried-forward repo debt:
- Repo CI on main still fails one known stale no-credit exact-string `FETCH_LIVE_ODDS` assertion. **Do not weaken the live-odds authorization guard to make it pass.**
- `claude/nice-cori-4qkwqs` is an **orphan branch sharing no merge base with main**. Tooling stranded there (Week-6 graders, `scripts/operations/salvage_production_decision_gates_v1.py`, Week-6 result docs) must be ported file-by-file; no merge is possible.
- `requirements.txt` leaves `pyarrow` unbounded; workflows pin `pyarrow==17.0.0` individually.

### PLAYER IDENTITY CARD V1 — BUILT, GATED, COMMITTED

`scripts/research/build_player_identity_card_v1.py`
`.github/workflows/research-player-identity-card-v1.yml`
`tests/test_build_player_identity_card_v1.py` — 40 tests

Committed artifact (git, not merely an artifact — AGENTS.md §5a):
`data/research/player_identity/player_identity_card_v1_2026.csv` + `_status.json`

Green run **`40371208533`**. 35,196 cards, 52 columns, 3,053 players, 32 teams, seasons 2025+2026, target weeks 1–7.

Boundary: `DESCRIPTIVE_STATE_ONLY_NO_FIT_NO_PROMOTION`. Nothing fit, scored or promoted. `sportsbook_inputs_used = 0` on every row, asserted in CI. Predictive use remains gated by `6420198257`.

Sources: `load_rosters_weekly` (64,455 rows — target-week team, player universe, position, and the gsis↔pfr bridge), `load_snap_counts` (36,501 — participation denominator), `load_pbp` (67,332 — opportunity rows + situational columns). `load_weekly_rosters` does not exist in this nflreadpy.

All five audit defects from `6420504608` are closed. Concrete evidence rather than assertion:
- **(C)** 180 skill-position players carry a 2026 target-week team appearing nowhere in their history (e.g. Kenneth Walker III, week 1: `team_target_week=KC`, `teams_in_history=SEA`). `mode(posteam)` would have mislabelled every one.
- **(B)** 7,666 active zero-opportunity player-weeks preserved as real zeros that the first version silently dropped.
- **(D)** universe is every rostered player; `participation_only_zero_opportunity` (1,587) is labelled distinctly from `no_history`.

**Composition matters when reading the counts.** 69% of cards are DB/OL/LB/DL/K/P/LS and 86% of the headline `no_history: 27,928` is those positions — a punter having no rushing history is not a finding. `can_receive_opportunity` flags QB/RB/FB/HB/WR/TE: **10,960 opportunity-eligible cards**, whose split is `no_history` 3,903, `prior_only` 2,593, `prior+current` 2,407, `participation_only_zero_opportunity` 1,519, `current_only` 538.

### PLAYER-STATE ANOMALY GATE — ENFORCED, WITH NAMED LIMITS

`anomaly_gate_version = PLAYER_STATE_ANOMALY_GATE_V1_PARTICIPATION_ONLY`. The suffix is deliberate: the gate currently reasons from participation alone.

Each week is judged **only against the player's own earlier weeks**, so a label does not shift with the target week and one bad week cannot excuse itself by dragging down its own baseline.

States supported today: `DNP_INACTIVE` (excluded by construction), `HEALTHY_FULL`, `UNRESOLVED_CONTEXT`, `NO_BASELINE`.
`INJURY_LIMITED` / `PARTIAL_GAME` / `RETURN_FROM_INJURY` need an injury / participation-detail source **not yet loaded**; those cases land in `UNRESOLVED_CONTEXT`, which is the prescribed fail-closed, not a silent `HEALTHY_FULL`.

Current distribution over 9,574 participation weeks: `HEALTHY_FULL` 6,786, `UNRESOLVED_CONTEXT` 1,358 (14.2%), `NO_BASELINE` 1,430. 1,976 week-slots withheld across 1,398 cards.

**Only `UNRESOLVED_CONTEXT` is withheld.** Withholding merely-zero-opportunity weeks would re-introduce the exact upward bias fix (B) removed; a test exists solely to stop a future change collapsing those two ideas back together.

Gate applies to trajectory, stability **and** efficiency. Canonical fields are the gated ones; ungated figures remain under `_all_weeks` suffixes so using contaminated evidence requires naming it. `gate_weeks_used` / `gate_weeks_withheld` / `gate_state_counts` / `efficiency_opp_used` / `efficiency_opp_withheld` make it auditable per card.

**A 22% false positive was found and fixed in the gate itself.** `returned_from_absence` fired on any participation gap — and a team bye is a gap for every player on that roster, so the week after a bye flagged whole teams. 385 player-weeks (1,743 → 1,358). A gap now counts as an absence only if the player's team actually played during it, derived from the snap-counts team column (no new source). `absence_bye_explained` records excused gaps. Without a team column the stricter reading is kept and CI fails the run.

Thresholds are **screening heuristics, not fitted parameters**: `PARTIAL_PARTICIPATION_RATIO = 0.5`, `MIN_BASELINE_WEEKS = 2`. They were deliberately not tuned, because tuning them would make them fitted quantities on data not authorized for fitting.

CI fails the run if the gate is unstamped, never fired, withheld nothing, changed no published number, lost bye discrimination, or published a rate not reconstructible from `opp_in_gate_weeks / gate_weeks_used`. A decorative gate fails.

### IDENTITY BRIDGE — SOURCE-CAPPED, QUANTIFIED

`pfr_id` ↔ `gsis_id` is built at **player level** (it is a player invariant, not a weekly fact). Ambiguity fails closed; currently 0 ambiguous ids, 2,520 players mapped.

The binding constraint is nflverse roster completeness, not the join: 48,111 of 64,455 roster rows (74.6%) carry a `pfr_id` while 64,429 carry a `gsis_id`. 381 of 384 never-bridged players are absent from the roster `pfr_id` index entirely, so **no player-level map can reach them** — the per-week→player-level change recovered only 6 player-weeks, not the "most" I predicted in `6422987723`.

Partial-bridge coverage is now **structurally impossible** (1 → 0 players; `untrusted_denominator_cards` 0), so the understated-denominator bias is eliminated by construction rather than mitigated.

**The statistic that matters:** coverage on opportunity-eligible cards with current-season opportunity is **2,939 / 2,945 = 99.80%**. `bridge_match_rate` and `offensive_player_weeks_unmatched` are the wrong statistics — the 4,900 unmatched weeks are overwhelmingly offensive linemen, who take snaps and never a carry or target.

Named exceptions: **Cody White** (WR, LV — 17 targets by wk 7, no bridged participation week, rates correctly NaN) and **Treyton Welch** (TE, NO — the sole former partial-bridge case). 24 `participation_denominator_contradicted` cards are Cody White plus linemen/punters with freak opportunity rows; all already NaN.

### WEEK-7 TICKET FORENSIC — RULING UNCHANGED

Construction failure, not model failure. `6419138959`, accepted by GPT in `6420198257`: "safe" tickets were ~41–47% naive hit probability; 25/57 unique yardage legs were bought-down OVERs against a canonical UNDER edge direction; SD / exact clearance probability not respected; portfolio concentrated in repeated players; ATD consumed share while uncertified.

**Construction failure does NOT close opponent/environment transmission.** Do not collapse these.

### OPPONENT / ENVIRONMENT TRANSMISSION FACTS — STILL OPEN

- `def_rush_epa` present but not consumed by generic RB matchup rules;
- explosive-play suppression available but not consumed generically;
- WR/TE/RB YPT and outside/slot YPT dropped before generic player projection;
- YBC/stuff fields dropped;
- generic game script hardcodes `pass_share = 0.57` (`scripts/modeling/rules_v2.py:98`, inside `project_game_script`);
- lead/trail does not alter generic pass/rush split;
- qualified Stage-2 conditional allocation does not reach production.

QB M89/M90 is richer; do not conflate it with generic RB/WR/TE.

**Kenneth Walker III remains the preferred first generic-skill transmission decomposition case.** GPT has a version at commit `cca416bd` / `docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`.

Environment is **not** game total / spread / implied points. That is one input. The environment is the football situation: opportunity topology, game script, teammate competition, defensive scheme and personnel.

### BETTING-LAYER CORRECTIONS — STILL IN FORCE

Until an evidence-backed construction V2 is frozen:
- alt direction conflicting with canonical main-line model direction ⇒ default **PASS**;
- raw yard cushion ≠ safety;
- use model mean + distribution/SD + exact threshold + `P(clear)` + exact price;
- do not label ~41% full-ticket probability "Safe";
- surface repeated-player and same-game concentration;
- no universal anchor across every tier;
- ATD remains uncertified/fun and separate.

Do not invent numeric exposure caps or threshold rules from Week 7.

### EXISTING SCIENCE BOUNDARIES — PRESERVE

Preserve all qualified/closed/source-blocked dispositions from the 2026-10-22 V2 handoff. Especially:
- QB M89/M90 + C2 active
- WR-R15 active
- TE-R5P active
- RB P3/R22 Week-1-only, **not** active Week 7
- Stage-2 conditional allocation qualified in research, not production-wired (`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`)
- direct WR/CB unavailable/gated off
- dedicated ATD model not certified
- RB rushing Stage-1 source reset remains `SOURCE BLOCKED`
- RB/TE team-context reset candidates remain `PREDICTIVE CLOSED`

### OPEN DECISIONS AWAITING THE AUDITOR

Claude has deliberately **not** decided these:

1. **Position universe** — restrict cards to opportunity-eligible positions, or keep the complete universe with `can_receive_opportunity` as the filter? GPT wrote fix (D), so the reach of (D) is GPT's call. Either way is a one-line filter.
2. **Threshold sensitivity** — a 0.3–0.7 sweep on `PARTIAL_PARTICIPATION_RATIO` is available as a descriptive diagnostic on request. Not run unasked, to avoid making a screening threshold look fitted.

### KNOWN GAPS — STATED, NOT HIDDEN

- **Injury source not loaded.** 14.2% of participation weeks are `UNRESOLVED_CONTEXT`, so trajectory evidence is genuinely thinner than the raw card suggests. This is the honest state.
- **Team shares are not yet gated.** `team_share_current`, `team_red_zone_share_current`, `team_inside_5_share_current` are role claims and therefore in the gate's scope, but gating a share correctly requires a **week-matched team denominator** — excluding the player's contaminated weeks while leaving the team total whole would deflate every share by construction.
- Raw counts, team totals and the opportunity-quality topology are intentionally ungated: per-play facts, outside the gate.
- Role elasticity and the full teammate-competition map remain deferred to V2.

### EXACT CURRENT NEXT ACTIONS

1. audit newest Claude work first; re-fetch #535 (newest `6423234263`) and live Actions;
2. resolve the two open decisions above;
3. load the injury / participation-detail source to convert `UNRESOLVED_CONTEXT` into `INJURY_LIMITED` / `PARTIAL_GAME` / `RETURN_FROM_INJURY` — highest leverage, shrinks every other gap;
4. week-matched team shares, per the known gap above;
5. Walker-first read-only opponent/environment transmission decomposition;
6. extend to representative WR/TE identities;
7. freeze any new predictive mechanism **before** scoring;
8. separately develop evidence-backed alt-threshold / portfolio construction V2;
9. port the stranded `claude/nice-cori-4qkwqs` tooling file-by-file;
10. grade Week 7 once the graders are ported to main.

No paid OddsAPI. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No reopening closed science without new causal evidence.

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
