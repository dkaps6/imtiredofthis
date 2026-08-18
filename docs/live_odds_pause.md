# Live Odds API pause

Full Slate defaults `fetch_live_odds` to `false` during preseason development so routine integration runs do not consume limited Odds API credits.

When disabled, Full Slate still runs the non-market pipeline through PlayerForm/history and repository validation. Market-dependent steps (live props/game odds, live opponent-map validation, deterministic market metrics, and Monte Carlo pricing) are skipped.

To restore the complete market-aware run, dispatch Full Slate with `fetch_live_odds=true`. No Odds API code or credentials were removed.
