Pick up my NFL project seamlessly from the previous chat.

GitHub is canonical. Repo: `dkaps6/imtiredofthis`.

Do not ask me to re-explain anything. Do not restart research. Do not recursively read old handoffs. I need this to feel like the exact same chat continuing.

Start by reading, in this exact order:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — **read only the newest top checkpoint**
3. `docs/handoffs/NFL_HANDOFF_2026-10-01_WEEK4_LIVE_BOARD_RECOVERED_CURRENT.md`
4. Issue #535 from comment `5936680436` onward
5. Query GitHub live for current main, open PRs/branches and Actions before doing anything

Critical current state you should expect to verify:

- Current production main before the continuity docs commit was `0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`.
- I explicitly authorized ONE Week-4 live OddsAPI Full Slate.
- Dispatcher run `36934550541` succeeded.
- Paid Full Slate run `36934563481` successfully acquired live odds and built/priced the board, then failed only at QB-C2 lineage stamping because live books had pass-yard offers for CHI Tyson Bagent and WAS Marcus Mariota while football-only starter authority remained Caleb Williams / Jayden Daniels.
- Exact paid artifact: `11197726111`, digest `sha256:202aa5205e71ad0acedf1910f505c2b362f564d8593a0f82dd0928fb45175aae`.
- Do NOT fetch odds again just to continue. Recovery used the preserved paid artifact with no second OddsAPI pull.
- Recovery run `36935917903` succeeded.
- Current recovered Week-4 live board artifact: `11197776900`, digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`.
- Recovered board: 3,460 side rows, 1,730 priced offers, 873 player-market rows, pricing status CURRENT.
- Tyson Bagent and Marcus Mariota were quarantined from the final Week-4 board only; 20 side rows removed.
- PR #671 merged that Week-4 final-board quarantine to main.
- Latest clean no-live main Full Slate after #671: run `36936244158` SUCCESS.
- There is an unfinished dynamic starter-conflict branch `repair-week4-qb-live-starter-conflict-v1@d1003a5809bb17b02592b533eb1f42bc247fb88c`. Its latest verifier `36937252724` correctly identifies Bagent/Mariota dynamically but fails because final-board-quarantined QB rows are missing the quarantine-authority marker required by market-lineage governance. Do not merge it yet and do not refetch odds to test it.
- The immediate game-day priority is to inspect/use recovered artifact `11197776900` and give me the live Week-4 board from that preserved snapshot.
- Raw huge edge is NOT validated betting confidence. Both new historical selector studies closed null:
  - market-relative scalar fair-line V1 = null;
  - offer-level residual probability V1 = null.
  Do not invent another threshold/top-N rescue.
- GSIS RB successor V1 has its first prospective Week-4 V2 private lock sealed before kickoff: 1/6 weeks, 2/10 vacancy team-games, 5/20 successor player-games. Do not alter formula or grade until games are final.
- ESPN availability semantics repair #670 is merged: ESPN injury-log OUT = UNAVAILABLE_REPORTED, not official inactive certification.
- RNG candidate #666 is validated/merged as candidate code but is not silently activated in canonical Full Slate.
- WR/CB remains source-blocked; no blind scans.
- CLV #663 passive; no odds pull solely for CLV.
- RB-PD2 remains HOLD.
- Weeks 1-3 postmortem is CLOSED at 629-611, 50.7%, -40.72u.

After reading/verifying, continue immediately. Do not give me a generic status-only answer if there is actionable work to do.

Most important right now: **use the recovered live Week-4 artifact first; do not spend more OddsAPI credits without asking me.**
