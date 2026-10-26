# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-26 — WEEK-7 SUNDAY/SNF FORENSIC / CONSTRUCTION FAILURE ACCEPTED / PLAYER-FIRST ENVIRONMENT TRANSMISSION OPEN

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

`docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`

Reusable restart prompt:

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`

Historical continuity remains in prior dated deep handoffs. Do not recursively read them.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only.

Do not scrap/rebuild PRISM. New identity/state/opponent work is additive.

Always distinguish:
- `SOURCE READY`
- `SOURCE BLOCKED`
- `PREDICTIVE CLOSED`
- `QUALIFIED`

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work:
1. read `AGENTS.md`;
2. read only this TOP checkpoint;
3. read the deep handoff;
4. fetch newest #535;
5. reconcile live main / PRs / Actions.

### LIVE PRE-HANDOFF STATE

Pre-handoff `main`:
`97ecae88f391a692cf694971c2bf2f34c04cf593`

PR #695 is merged and preserves the full Week-7 Sunday/SNF ticket evidence under:

`docs/research/evidence/2026-10-25_week7_ticket_ledger/`

Canonical paid Week-7 board remains:
- run `39849371821` SUCCESS
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- durable under `data/market_track_record/`

**No additional paid OddsAPI call is authorized.**

After PR #695:
- archive workflow `40341879091` SUCCESS
- Repo CI `40341289972` failed only on the known stale no-credit exact-string assertion:
  - 660 passed
  - 1 skipped
  - 1 failed
- do not weaken the live-odds authorization guard to make that stale test pass

### WEEK-7 TICKET FORENSIC — CURRENT RULING

User reported only one Sunday ticket hit and no SNF regular tickets hit.

Claude #535 comment `6419138959` found a major **downstream construction failure**:
- “Safe” tickets were only ~41–47% naive hit probability;
- 25/57 unique yardage legs were bought-down OVERs while the canonical main-line edge direction was UNDER;
- model SD / exact clearance probability was not properly respected;
- portfolio was heavily concentrated in repeated players;
- ATD consumed meaningful portfolio share despite being uncertified.

GPT accepted that diagnosis in `6420198257`.

**BUT: construction failure does NOT close opponent/environment transmission.**

Correct disposition:

- Week-7 parlay failure = primarily construction / threshold / exposure problem.
- Opponent/environment transmission = independently **OPEN** architecture/research problem.

Do not collapse these into one conclusion.

### OPPONENT / ENVIRONMENT TRANSMISSION FACTS

Prior audits established:
- `def_rush_epa` present but not consumed by generic RB matchup rules;
- explosive-play suppression available but not consumed generically;
- WR/TE/RB YPT and outside/slot YPT dropped before generic player projection;
- YBC/stuff fields dropped;
- generic game script hardcodes pass_share=0.57;
- lead/trail does not alter generic pass/rush split;
- qualified Stage-2 conditional allocation research does not currently reach production.

QB M89/M90 is richer; do not conflate it with generic RB/WR/TE.

**Kenneth Walker III is the preferred first generic-skill transmission decomposition case.**

### PLAYER IDENTITY — NOT BLOCKED

Claude corrected an overstatement in `6420003475`.

Strictly-prior player identity can be reconstructed leakage-safe from published football history (PBP, player stats, snap counts, schedules, etc.).

Current gate-design direction:
- preregister football-semantic partitions before scoring;
- require independent replication;
- pooled population must not materially harm;
- declared partition must improve consistently;
- minimum support;
- no subgroup shopping / top-N rescue / post-hoc named-player carveout;
- report per-player heterogeneity.

### BETTING-LAYER CORRECTIONS

Until an evidence-backed construction V2 is frozen:

- alt direction conflicting with canonical main-line model direction => default PASS;
- raw yard cushion != safety;
- use model mean + distribution/SD + exact threshold + P(clear) + exact price;
- do not label ~41% full-ticket probability “Safe”;
- surface repeated-player and same-game concentration;
- no universal anchor across every tier;
- ATD remains uncertified/fun and separate.

Do not invent numeric exposure caps or threshold rules from Week 7.

### EXISTING SCIENCE BOUNDARIES

Preserve all qualified/closed/source-blocked dispositions from the 2026-10-22 V2 handoff.

Especially:
- QB M89/M90 + C2 active
- WR-R15 active
- TE-R5P active
- RB P3/R22 Week-1-only, not active Week 7
- Stage-2 conditional allocation qualified in research but not production-wired
- direct WR/CB unavailable/gated off
- dedicated ATD model not certified
- RB rushing Stage-1 source reset remains SOURCE BLOCKED
- RB/TE team-context reset candidates remain PREDICTIVE CLOSED

### EXACT CURRENT NEXT ACTIONS

After live reconciliation:

1. audit newest Claude work first;
2. continue descriptive Player Identity Card construction if not already advanced;
3. run/verify fresh Stage-2 source-readiness re-probe under the frozen contract if not already completed;
4. run Walker-first read-only opponent/environment projection decomposition;
5. extend to representative WR/TE player identities;
6. freeze any new predictive mechanism before scoring;
7. separately develop evidence-backed alt-threshold / portfolio construction V2;
8. preserve richer football-state/explanation evidence going forward.

No paid OddsAPI.
No sportsbook upstream.
No aggregate-MAE optimization.
No post-hoc subgroup rescue.

Newest visible #535 comment at final pre-handoff reconciliation:
**6420198257**

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
