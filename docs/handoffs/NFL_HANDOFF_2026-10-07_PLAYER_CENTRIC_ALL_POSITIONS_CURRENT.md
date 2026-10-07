# NFL HANDOFF — 2026-10-07 — PLAYER-CENTRIC ALL-POSITIONS CURRENT

**STATUS: CURRENT CROSS-CHAT AUTHORITY**

GitHub is canonical. This handoff supersedes the 2026-10-06 Week-4 matchup-transmission handoff for immediate execution.

## 0. Immediate user intent

The user explicitly does **not** want to throw away prior validated work.

The current direction is:

> Preserve the existing team/opponent/position/model science, but add a stronger **individual-player layer** so the prediction object is the actual player, not merely a position bucket with shared adjustments.

The user now wants to **button up every skill position before running an all-player replay/backtest**.

The immediate unanswered question when the prior chat maxed out was:

> Which positions are fully buttoned up, and which still need player-level work before we rerun all players / all positions?

Do not skip this question in the next chat.

---

## 1. Canonical production state

At handoff creation, live `main` before this docs-only continuity update was:

`e1e7adbaa27cd0b4ff1e2b04c467586a2e7c3e40`

Always query live main before editing.

Do not alter canonical production simply because a research signal is positive.

Protected production concepts remain intact:
- QB M89/M90 point mean;
- C2 distribution where applicable;
- M38 WR1 anchor;
- WR-R15 WR2+ entitlement;
- TE-R5P TE entitlement;
- Week-1 RB specialist authorities;
- RB Rush+Receiving Conservation V2;
- existing joint MC / team opportunity / matchup / injury / opponent structure.

No paid OddsAPI pull without explicit user authorization.

---

# 2. Core architectural finding — Player Individualization Audit V1

Branch:
`research-player-individualization-audit-v1`

Run:
`37558838148` SUCCESS

Artifact:
`11455757608`

Digest:
`sha256:0d60c03a6d3aebdaf1cce47472c82988defd6320e0106a07418b567ea8071a56`

Disposition:
`PLAYER_INDIVIDUALIZATION_PARTIAL`

Important correction/addendum:
`docs/research/PLAYER_INDIVIDUALIZATION_AUDIT_V1_FULL_STACK_ADDENDUM.md`

Addendum commit:
`5756b1aecaab62c17148b50e314d9d44d67faeed`

### What the audit proved

The stack is **not** simply “all WRs / RBs / TEs / QBs get the same projection.”

PlayerForm/Bayesian already retain player-specific:
- target share;
- rush share;
- receptions/target;
- YPT;
- YPC;
- YPA;
- YPRR/route-rate where a real source exists.

But player-specific history is still shrunk toward position priors, and many weekly environment transformations are shared by role/team/opponent buckets.

Typical Week-5 position-prior contribution remained material:
- WR target share ~23.1%
- RB rush share ~23.1%
- TE target share ~25%
- YPT/YPC ~29-31% position prior in representative established-player cases.

The main philosophical conclusion:

> Current architecture = player-level outputs + meaningful individual history + positional shrinkage + shared weekly transforms + some promoted player specialists.

Do **not** replace this whole stack. Add missing individual-player state where evidence supports it.

---

# 3. Full-stack position map — where we are NOW

This section directly answers the user's current “what positions are left?” question.

## QB — largely BUTTONED UP for this player-individualization pass

Current authority:
- QB pass yards mean already uses M89/M90 with individual QB attempts/YPA history plus team/opponent environment.
- C2 can own distribution shape for selected QBs.

Prior player-reliability work:
- QB-PD2 canonical run `34064528914`
- disposition `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE`
- QB-PD3 internal reliability also recovered null.

New QB-receiver pair work in this chat:
- pair source READY:
  - run `37626968980`
  - artifact `11483954236`
  - 2022-25 pair history clean
  - 2026 W1-4 = 329 live primary-QB/receiver pairs
  - 100% pair-ID coverage
- raw exact-pair YPT candidate CLOSED:
  - run `37628838585`
  - pooled WR+TE YPT MAE `4.434895 -> 4.550342` worse
  - bootstrap P(improve)=0.0

### QB conclusion

**Do not reopen generic QB mean before the all-position replay.**

QB is sufficiently individualized for this phase. If QB player-level work resumes later it needs genuinely new information, not player-error persistence, internal-disagreement calibration, or raw QB-receiver pair YPT.

QB is **not the blocker** to the requested all-position run.

---

## RB — NOT buttoned up; this is still the biggest player-state gap

Full-stack addendum conclusion:
- Week-1 RB has promoted specialists;
- non-Week-1 RB does **not** have the same promoted multiseason individual-room entitlement specialist as WR-R15 / TE-R5P.

Player-state source readiness:
- branch `research-player-state-live-coverage-v1`
- run `37560311001` SUCCESS
- artifact `11456556226`
- 30/30 scheduled Week-5 RB rooms have distinguishable strictly-prior individual state.

Prospective RB allocation shadow:
- branch `research-rb-player-state-allocation-shadow-v1`
- contract freeze `39074a743eff4e5cb47dcc58c24e9b68c5c48ef1`
- Week-5 lock run `37560824479` SUCCESS
- artifact `11456916566`
- row digest `sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a`

Locked:
- 30 teams
- 98 RB/HB/FB identities
- all 30 rooms receive a non-zero individual allocation change
- frozen shadow:
  `0.50 recent RB-room carry share + 0.50 recent RB-room snap fraction`
- median team max player-share movement = **9.46 percentage points**
- no sportsbook
- no Week-5 outcomes
- no fitted coefficient
- production unchanged

Important RB anti-retest:
- M96 retrospective RB stop remains binding.
- Do not reopen another retrospective RB rushing feature hunt merely because we want all positions.
- RB-PD2 historically found player difficulty persistence, but PD3/4/5 width-style follow-ups failed.
- RB receiving-yard mean lane is separately closed absent genuinely new pregame football information.

### RB conclusion

RB is the **main unfinished position**.

Before the requested “all players / all positions” replay, the next chat should decide exactly what “buttoned up” means for RB without violating M96:
1. preserve the existing prospective Week-5 room-allocation shadow;
2. do not pretend it has a result yet;
3. audit whether there is any still-valid **new player-level RB football state** needed for receiving / efficiency, or whether current closed science means RB should enter the global replay as:
   - existing production baseline +
   - prospective allocation shadow only where legally available.

Do not manufacture a retrospective promotion test just to make the matrix look complete.

---

## WR — opportunity science substantially buttoned up; distribution layer still unfinished

Exact promoted parent:
- M38 WR1
- WR-R15 WR2+ entitlement

WR mechanism persistence:
- branch `research-wr-player-mechanism-persistence-v1`
- run `37634044900` SUCCESS
- artifact `11487359185`
- disposition `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`

Pooled:
- signed opportunity persistence rho **+0.1699**
- opportunity-difficulty rho **+0.3183**
- signed efficiency persistence **FAIL**, rho **-0.0511**
- efficiency-difficulty rho **+0.2833**

This means:
- individual WR opportunity errors persist;
- individual efficiency **difficulty** persists;
- signed YPT-style efficiency bias does not persist.

### Confirmed WR/TE opportunity state: target-share trajectory

Branch:
`research-player-target-share-trajectory-v1`

Run:
`37638269235` SUCCESS

Artifact:
`11490707699`

Digest:
`sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`

Disposition:
`PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED`

Frozen feature:
- recent2 individual target share
- minus earlier same-season same-team target share.

WR replication:
- 2023 rho **-0.03999**
- 2024 rho **-0.10773**
- pooled **-0.07646**
- cluster P(negative)=**1.000**

Interpretation:
rising individual role tends to expose underprediction by the existing entitlement stack.

### Week-5 WR/TE trajectory shadow already locked

Branch:
`research-player-target-share-trajectory-shadow-v1`

Contract freeze:
`c0a071ce5324d0d3329f846a52a96a5c415a0797`

Corrected lock run:
`37654382316` SUCCESS

Artifact:
`11497153776`

Artifact digest:
`sha256:c12878ed97a4543a907ac54f4cbadadda7178db068491373091d2c048f176e88`

Canonical row digest:
`sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

Locked:
- 277 WR/TE rows
- 167 WR
- 110 TE
- trajectory available 230/277 = 83.03%
- 247 players changed
- 60 protected rooms changed
- M38 WR1 anchor frozen
- WR2+ pool exactly conserved
- TE pool exactly conserved
- team modeled target mass exactly conserved within floating-point tolerance
- no sportsbook
- no Week-5 outcomes
- no fitted parameter

Frozen transformation:
`weight = promoted entitlement * exp(trajectory_delta)`
then renormalize inside protected room.

### WR conclusion

WR **mean opportunity layer is buttoned up enough to shadow prospectively**.

WR is **not fully buttoned up overall** because the newly confirmed player-specific efficiency-uncertainty signal has not yet been turned into a frozen distribution shadow.

---

## TE — opportunity science substantially buttoned up; distribution layer still unfinished

Exact promoted parent:
- TE-R5P.

TE same-player persistence:
- run `37630274063`
- disposition `TE_PLAYER_ERROR_PERSISTENCE_DETECTED`
- 2,005 rows / 121 TEs
- signed error, difficulty, and 30+ miss tendency all replicated.

TE mechanism decomposition:
- branch `research-te-player-mechanism-persistence-v1`
- run `37631011196` SUCCESS
- artifact `11486082927`
- disposition `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`

Pooled:
- signed opportunity persistence rho **+0.2500**
- opportunity-difficulty rho **+0.3936**
- signed efficiency persistence **FAIL**, rho **-0.0567**
- efficiency-difficulty rho **+0.2240**

Target-share trajectory also replicated for TE:
- 2024 rho **-0.10032**
- 2025 rho **-0.06948**
- pooled **-0.09876**
- cluster P(negative)=**0.9998**

TE is included in the same Week-5 trajectory shadow above.

### TE conclusion

Like WR:
- opportunity player-state lane is strong and already locked prospectively;
- efficiency mean should **not** be patched;
- player-specific distribution/uncertainty remains unfinished.

---

# 4. Closed pass-catcher feature family — do not retest

## Situational target earning

Source audit showed third-down, red-zone, two-minute target shares were genuinely nonredundant live player state.

But historical residual test CLOSED:
- branch `research-player-situational-target-residual-v1`
- optimized run `37637017878`
- artifact `11489713078`
- disposition `NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL`

Pooled relationships were near zero / nonreplicating.

Do not rescue:
- red zone for WR only;
- third down;
- two minute;
- alternate windows;
- thresholds;
- role carveouts.

This was a useful null.

---

# 5. New confirmed efficiency-difficulty signal — target-depth dispersion

Branch:
`research-player-target-depth-dispersion-v1`

Run:
`37655282486` SUCCESS

Source SHA:
`7607eb068bb4dad92ec5b0d8497ba5069349ae4d`

Artifact:
`11497703984`

Digest:
`sha256:39110b5d8d344b5beb51f5eab02ba931ffddae41932f8d7dcf5ca5da66824f55`

Result doc was explicitly repaired/frozen after the prior UI interruption:
`docs/research/PLAYER_TARGET_DEPTH_DISPERSION_V1_RESULT.md`

Result commit:
`2aa3a4ce1d9487476368d353c0ab8941f904b790`

Disposition:
`PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED`

Frozen feature:
- individual receiver's strictly-prior target-depth SD;
- latest up to 8 completed receiver target games;
- >=4 prior games;
- >=10 finite air-yard targets.

Target:
- **absolute efficiency-component error**, not signed YPT error.

WR:
- 2023 rho **+0.07713**
- 2024 **+0.09278**
- pooled **+0.08499**
- cluster P(positive)=**0.9990**

TE:
- 2024 **+0.04696**
- 2025 **+0.04930**
- pooled **+0.05414**
- cluster P(positive)=**0.9782**

Combined:
- 6,497 rows / 350 receivers
- rho **+0.18412**
- cluster P(positive)=**1.000**

Interpretation:

> Receivers whose own target depths are more dispersed are systematically harder to translate from opportunity into receiving-yard outcomes.

Because signed efficiency persistence failed, this signal is **not** authorization for a YPT mean change.

Correct downstream use:
- mean-neutral player-specific uncertainty / distribution shape.

Protected closures remain closed:
- WR R7 signed-YPR traits;
- M72;
- M75;
- WR-R3 residual-width calibration;
- TE Width V2;
- QB-receiver pair YPT.

---

# 6. Open branch that has NOT been implemented yet

Branch exists:

`research-player-target-depth-distribution-shadow-v1`

At handoff creation it was created from main but **no contract, code, or run has been added yet**.

This is the most obvious unfinished WR/TE item.

The next chat must not claim this distribution shadow exists or is running.

Required design:
- preserve exact baseline receiving-yard mean;
- preserve target entitlement;
- preserve team pass/target volume;
- use individual target-depth dispersion only to alter uncertainty/distribution;
- no sportsbook inputs;
- no production change;
- score probability/distribution quality, not mean MAE.

---

# 7. Important question about 2026 Weeks 1–4

The user asked whether we can use the four already-live 2026 weeks rather than simply wait for Week 5.

Answer:

## Target-share trajectory shadow

The exact frozen trajectory feature requires:
- 4 prior same-season team games;
- RECENT2 vs EARLIER.

Therefore:
- Week 1: 0 prior
- Week 2: 1
- Week 3: 2
- Week 4: 3
- Week 5: first eligible week.

So **Weeks 1–4 cannot exercise the exact frozen trajectory V1 formula**.

Do not weaken the contract after seeing Weeks 1–4 outcomes by changing to recent1/recent3 or pulling 2025 continuity into the same candidate.

However:
- historical Week-5+ seasons can be used for an exact retrospective integration test;
- a separate early-season trajectory hypothesis could later be frozen prospectively.

## Target-depth dispersion

This can use strictly-prior 2025 history for returning players.

Therefore the four 2026 live weeks can be replayed for a **mean-neutral target-depth distribution shadow** once its contract is frozen.

That is useful current-season evidence.

---

# 8. Exact “all positions before global replay” checklist

This is the immediate roadmap.

## QB
**Status: READY / do not reopen generic mean.**

No blocking player-level work before global replay.

## RB
**Status: NOT READY / main remaining position gap.**

Need a bounded decision on what is scientifically legal and still open:
- preserve Week-5 prospective allocation shadow;
- do not violate M96;
- determine whether any genuinely new RB receiving/efficiency player-state lane is needed and novel;
- otherwise explicitly mark RB as “current production + prospective allocation shadow” for the all-position replay.

Do not invent a retrospective RB feature just for symmetry.

## WR
**Status: OPPORTUNITY READY; DISTRIBUTION NOT READY.**

Need:
- implement/freeze target-depth mean-neutral distribution shadow;
- then WR can enter the global replay with:
  - production baseline;
  - trajectory opportunity layer where eligible;
  - player-specific uncertainty layer if the shadow contract is valid.

## TE
**Status: OPPORTUNITY READY; DISTRIBUTION NOT READY.**

Same remaining work as WR.

## Global all-player replay
**DO NOT START YET** if the user's preference remains “button everything up first.”

First finish:
1. WR/TE target-depth distribution shadow contract + lock/replay path.
2. RB final player-level scope decision / remaining novel audit.
3. Then run the all-position replay.

---

# 9. What the eventual all-player replay should contain

When the user says all positions are ready:

### QB
Use existing M89/M90/C2 production authority unchanged.

### RB
Use existing production authority plus only legally frozen player-state shadow(s).

### WR
Use M38 + WR-R15 baseline.
Add:
- target-share trajectory entitlement shadow where eligible;
- mean-neutral target-depth uncertainty shadow if frozen.

### TE
Use TE-R5P baseline.
Add:
- target-share trajectory entitlement shadow where eligible;
- mean-neutral target-depth uncertainty shadow if frozen.

Do not allow any player layer to:
- change team target/pass volume unless explicitly contracted;
- bypass conservation;
- inject sportsbook state upstream;
- rewrite protected production science.

---

# 10. Week-4 / betting context still authoritative

Week 4 final:
- 220-207
- 51.52%
- -3.38u

Weeks 1–4 cumulative:
- 849-818
- 50.93%
- -44.10u

Raw edge/fair probability remains overconfident and is **not validated confidence**.

Right-Tail Asymmetry V1:
- replicated for rush_yards, rec_yards, receptions, rush_rec_yards in 2024+2025;
- pass_yards did not replicate.

Do not lose this lane, but the user explicitly pivoted current priority toward player-level individualization.

---

# 11. Matchup Transmission / defender / public YPT closures remain binding

Football Matchup Transmission B/C:
- run `37514137803` SUCCESS
- 4 replicated source-level signals identified.

Integration candidates all CLOSED:
- run `37531333007`
- FMT-RB1 worse MAE
- FMT-WR1 worse MAE
- FMT-TE1 worse MAE.

RB opponent-defender injury candidate CLOSED.

Public TE position-YPT additive candidate CLOSED.

Do not reopen these merely because player-centric work is promising.

---

# 12. No active run at handoff time

At handoff creation, the recent relevant Actions were all completed.

Most recent:
- Target Depth Dispersion V1 `37655282486` SUCCESS
- Trajectory Shadow Week-5 lock `37654382316` SUCCESS
- Target Share Trajectory V1 `37638269235` SUCCESS
- WR mechanism `37634044900` SUCCESS
- TE mechanism `37631011196` SUCCESS
- RB player-state lock `37560824479` SUCCESS

Do not tell the user something is “still running” unless live Actions says so.

---

# 13. Immediate next action for the next chat

The user asked to **finish all positions before the all-player replay**.

Do this in order:

1. Query live main, branches, Issue #535 tail, and Actions.
2. Answer the user's position-status question immediately:
   - QB ready;
   - RB main unfinished position;
   - WR/TE opportunity ready but distribution shadow unfinished.
3. Continue work — do not just discuss:
   - freeze and implement `research-player-target-depth-distribution-shadow-v1` as mean-neutral;
   - then resolve RB's remaining legal/open player-level scope without violating M96.
4. Only after those are buttoned up, build the all-position replay.
5. For 2026 W1-4:
   - trajectory V1 cannot legally operate under its frozen 4-prior-game contract;
   - target-depth distribution can potentially replay using 2025 prior history;
   - historical Week-5+ integration can test trajectory transformation.
6. No paid OddsAPI pull unless user explicitly authorizes it.

---

# 14. Memory-efficient read order

Do **not** recursively load old handoffs.

Read only:

1. `AGENTS.md`
2. newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff:
   `docs/handoffs/NFL_HANDOFF_2026-10-07_PLAYER_CENTRIC_ALL_POSITIONS_CURRENT.md`
4. exact result/lock docs only as needed:
   - `docs/research/PLAYER_INDIVIDUALIZATION_AUDIT_V1_FULL_STACK_ADDENDUM.md`
   - `docs/research/RB_PLAYER_STATE_ALLOCATION_SHADOW_V1_WEEK5_LOCK.md`
   - `docs/research/WR_PLAYER_MECHANISM_PERSISTENCE_V1_RESULT.md`
   - `docs/research/TE_PLAYER_MECHANISM_PERSISTENCE_V1_RESULT.md`
   - `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_V1_RESULT.md`
   - `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_WEEK5_LOCK.md`
   - `docs/research/PLAYER_TARGET_DEPTH_DISPERSION_V1_RESULT.md`
5. Issue #535 from comment `6029157126` onward
6. live relevant branches / Actions

Then work.

---

# 15. User style / execution expectations

- Do not ask the user to re-explain the project.
- Do not restart research.
- Do not recursively read old handoffs.
- Do not give repetitive status summaries instead of working.
- Use short GitHub/tool bursts because UI timeouts have been frustrating.
- Hard-checkpoint important results in GitHub quickly.
- Never say something is running unless an Action is actually running.
- User prefers research lead behavior: identify the next scientific question, freeze it, execute, fail closed, and keep moving.
- Preserve prior validated science; player-centric work is a **complement**.
