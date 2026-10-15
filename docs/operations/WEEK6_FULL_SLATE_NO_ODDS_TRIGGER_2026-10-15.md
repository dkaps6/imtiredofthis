# Week 6 Full Slate No-Odds Validation Trigger — 2026-10-15

Purpose: trigger the canonical `.github/workflows/full-slate.yml` from `main` for Thursday Week 6 pregame validation.

Execution contract:
- season: 2026
- prior season: 2025
- slate date: workflow default / nearest upcoming NFL week
- fetch_live_odds: false
- paid OddsAPI calls authorized by this trigger: 0

This is an operational trigger only. It does not change projection science or production model logic.
