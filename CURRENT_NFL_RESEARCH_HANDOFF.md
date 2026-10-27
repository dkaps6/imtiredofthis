# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-27 — V13 — RATE-SCALE PLAYER-FIRST ENTITLEMENT REPLAY IN PROGRESS / NO MERGE / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis. **Current state:** research-only; no production changes authorized by this handoff.

**Read in this exact order**:
1. AGENTS.md (canonical 2026 operating contract, player-first first principle).
2. **ONLY THIS NEWEST ACTIVE TOP V13**. Do not recursively read prior V2–V12 checkpoints.
3. **Full V13 continuity:** docs/handoffs/NFL_HANDOFF_2026-10-27_V13_RATE_GUARD_PLAYER_ALLOCATION_CONTINUITY.md. This is the detailed verified state, prior audits, exact SHAs, known artifact identifiers, source locations, unresolved issues, and acceptance criteria. Read it in full before independent action.
4. GitHub Issue https://github.com/dkaps6/imtiredofthis/issues/535 — fetch **comments after 6456181944** FIRST and latest research-branch SHA. This is the GPT ↔ Claude Code work channel.

**Most important immediate next task:** Audit Claude's next research-branch read-only replay target (if posted). The acceptance is SOURCE-PINNED team context rebuild → real rule target shares → M38 → finite team entitlement → TE-R5P by **named player**, with BASELINE PARITY first, no markets/outcomes, no production merge. Claude latest known comment 6456141256 explains why prior paid replay does NOT test the patch; GPT replied 6456181944 with precise acceptance design. At checkpoint writing, no newer Claude target observed. Refetch; do not assume one exists or claim work occurred in background.

### NON-NEGOTIABLE USER SCIENCE
**THE PLAYER SHOULD HAVE THE RULES, NOT THE POSITION HAVE THE PLAYER.** The user wants named-individual projections based on identity, injury/state, real opportunity, teammate competition, defense, role, script and market-specific football logic. Positional MAE/RMSE are QA only. **ANOMALIES ARE QUESTIONS, NOT TRENDS**: check injury/partial games before inferring role decay (Jalen Coker example). Maintain promoted Bayesian/ML/State/MC/QB/RB layers, do not scrap previous work. Avoid simplistic statistics, unsupported yes-saying, broad assertions, timeouts, and asking user to reteach already documented details.

### RESEARCH CURRENT STATUS
- **CLEARED** source-wide opponent transmission: research target 8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea. Generic RB rush has weak box-only direct opponent channel; def_rush_epa and explosive allowance transported-but-stranded in generic rules. QB promoted opponent pipeline distinct. Walker Week7 exact case cleared: NEVER redo.
- **CLEARED read-only** named JSN/Kupp/Barner SEA-vs-KC role trace target 4f95723154d5473b362f33c450b6f2d5fd8f6b60 and GPT ruling 6446595260. Source-conditioned TE zone×middle gate 1.265; Kupp structural WR-label exclusion caused **NO actual direct coverage penalty** in that matchup; Kupp -28% shift from M38 hierarchy NOT opponent, conditional no injury override; 0.95/0.05 finite target room. Final player yardage mean attribution unproven.
- **CLEARED descriptive-only** 2025 gate prevalence proxy target 91c3b81e2a201ca31b4632d980bf3deffa336fd9, GPT 6449328708. Across 16 arms×bases only heavy-box gate unconditionally near-dead; no production Sharp 2026 population inference, denominator unknown.
- **CORRECTED/no positive persistence proof** 2026 PBP stranded-signal dispersion revision target 210205071774a41348e92f1cb87204301ff0fe21, Claude 6450508461; earlier cumulative stability artifact misleading due overlapping weeks, disjoint split nearly all weak. 2026 PBP EXISTS; 2026 nflverse participation (defenders_in_box and defense_man_zone_type) unavailable. Signals and production provenance NOT equivalent.
- **LIVE Week8 input scale bug CONFIRMED** from actual Full Slate GitHub Actions run 40551229604 artifact 12755640785. Coverage-v2 28 active teams, Team Context 32. middle_open_rate supplied as percentage points 24.4..62.8 but compared to fraction >=.50. Middle-open false fires 15/28 active defenses (ATL BUF CAR CHI CIN CLE DAL DEN DET IND LAC LV NE PHI TB); 4 bye teams HOU NO NYG SF had unscaled man/zone fallback. Box rates already finite fractions; don't change box gates. Sharp coverage denominator UNESTABLISHED. Full artifact details/source hashes in deep V13.
- **RATE-SCALE FIX PROPOSED RESEARCH BRANCH ONLY** target ab29a0962bb77bc2a3430cd8200c4cbe327b6c88. docs/production/RATE_SCALE_GUARD_FIX_PROPOSAL_V1.md. Explicit rate-type allowlist, percent→fraction, out-of-range fail closed, provenance suffix; no threshold tune, no production merge. Middle_closed_rate is NOT a materialized TeamContext field; helper only future-protects it.
- **GPT Week8 multiplier SHADOW replay VERIFIED limited** #535 comment 6455253348: exact original diagnostics reproduced all eight rule multiplier fields for 404/404 rows; 40 source fields scale-corrected, 217/404 diagnostic rows have slot & TE multiplier changes, all other multipliers unchanged. Only 45 TE rows have current direct role-target pathway; team-room reallocation can indirectly change teammates. **NOT 217 impacted named projections**; final means not tested.
- **SLOT branch unreachable structural** Claude 6455350999; target 80b5745cb1b848cf6d4c089e18b15382f7254c5f; docs/research/PRISM_SLOT_ROLE_UNREACHABLE_V1.md. SWR alignment collapsed upstream to WR, no downstream role literal SLOT; zero SLOT rows W7/W8. This is a separate identity/role contract and **must NOT be silently changed** as part of rate-scale patch.
- **REPLAY BLOCKER**: .github/workflows/replay-paid-full-slate-artifact-once.yml pinned to EXPIRED Week5 artifact run 37855917750/id 11584521127; restores prebuilt team_context_v3 and skips corrected team_context_v3.py. Blindly repointing would misstate a green regression as rate-guard validation and break Week5-specific paid/quarantine fixture assertions. Keep its separate CI maintenance lane. Correct current acceptance is isolated canonical-context-builder shadow, not this workflow.

### MUST-PASS ACCEPTANCE FOR NEXT CLAUDE TARGET
1. Read frozen W7 source inputs (team_form, promoted QB context, coverage v2, player form/consensus, injuries, schedule, Bayesian baseline, full metrics universe) with exact source hashes. Fail closed if unavailable.
2. Same frozen inputs both arms, no sportsbook/odds and no target-week outcomes. Baseline arm MUST reproduce archived **named-player rules_tgt_share, post-M38, entitlement, identities/order and 0.95+0.05 room mass**; if not, STOP attribution.
3. Only percent-scaling differs; no YPT/YPC/pace/pass+rush attempts/other gate/role changes.
4. Use canonical production build_team_context_v3, apply_rules_to_metrics with context actually patched and not loaded from old disk, materialize_target_entitlement, apply_te_r5p_entitlement. Do not invent a replacement allocator.
5. Provide complete named-player before→after and team-room direct TE vs indirect teammate changes; unchanged games as controls, provenance and SHA. Final yards mean unclaimed unless actual mean builder run.
6. GPT independent audit source/tests/artifacts and post #535 ruling. **User must expressly authorize any merge**. This handoff does NOT authorize implementation, fitting, production changes, paid odds, replay scoring or threshold selection.

### REMAINING AUTHORIZED RESEARCH QUEUE
Preserve coverage/box/opponent source VALUES and rules multiplier snapshots (not only column names); separate player-first SLOT identity fix proposal and raw-vs-Bayesian rank mismatch; Claude generic game script lane: success_diff/lead-neutral-trail computed, but generic pass share remains hardcoded at 0.57; evaluate whether any opponent game-script output actually drives opportunities. Inside-five TD, TE-R5P, player-centric RB role and broader prop ticket lanes remain governed/queued, not blanket approved. Betting CLOSED, no paid OddsAPI, no sportsbook upstream, no Walker redo, no W7 outcome fitting.

### COMMUNICATION / DURABILITY
Claude communication: GitHub Issue #535. Don't confuse executor local pytest counts with independently CI-verified tests; audit-only GitHub workflow usually compile+strict audit, NOT pytest. Do not claim the user approved a merge. Update this ONE root TOP after material milestones and archive displaced checkpoints under docs/handoffs. Existing displaced full V12 checkpoint archived as:
docs/handoffs/NFL_HANDOFF_2026-10-27_V12_PRE_V13_ROOT_ARCHIVE.md

**Authoritative deep research continuity:** docs/handoffs/NFL_HANDOFF_2026-10-27_V13_RATE_GUARD_PLAYER_ALLOCATION_CONTINUITY.md
**Exact last GPT governance ruling:** https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6456181944
**Next step:** check #535 after 6456181944, audit new SHA if any; DO NOT ask user to repeat PRISM goals.
