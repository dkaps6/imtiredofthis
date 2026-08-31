# 2026 Full Slate Runtime Contract

The canonical production path is `.github/workflows/full-slate.yml`.

## Football model first

The football projection stack runs independently of sportsbook lines. Live odds are a downstream comparison/pricing layer only. The production path uses the validated 2026 provider/context bridges, Player Identity v3, Team Context v3, the frozen MC/ML/State ensemble, and the M89/M90 QB passing-yards synthesis.

## Preseason and early-week behavior

When current-season history is unavailable, explicit prior-season inputs may be used only through the guarded production bridges with provenance. Missing current player-prop markets are not a football-model failure.

## Live odds contract

When `FETCH_LIVE_ODDS=false`, no OddsAPI credits are used and sportsbook-dependent pricing steps are skipped.

When `FETCH_LIVE_ODDS=true`, `scripts/run_live_odds_gate.py` must:

1. clear previous sportsbook artifacts before fetching;
2. run the existing OddsAPI provider adapter;
3. scope returned events to the authoritative active season/week in `data/team_week_map.csv`;
4. exclude preseason, other-week, or otherwise off-slate events;
5. write `data/live_odds_status.json`;
6. set `available=false` when the active slate has no posted player-prop markets, allowing the football stack to continue while sportsbook pricing is skipped;
7. treat provider/authentication failures as fatal.

The pricing layer may run only when the live-odds gate reports `available=true`.

## Main-branch verification

A push to `main` runs Full Slate with default `FETCH_LIVE_ODDS=false`. This provides an automatic no-credit production rehearsal after production merges without waiting for regular-season prop markets to exist.
