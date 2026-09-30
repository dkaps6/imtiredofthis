# WR-CB Exact Opponent Weekly Roster Guard V1
Source-quality audit only, frozen after Claude found 14 source provider IDs with
more than one *schedule-consistent* CB opponent team in the same week. The WR
schedule check is insufficient to validate a CB's own weekly team membership.

Require the CB's exact GSIS ID to have unambiguous same-season/week defensive
roster membership on the reported WR opponent team. Independently confirmed
another team, unknown membership and multi-team weekly rosters all fail closed.

Reuse the preserved PR #665 artifact **run 36735087776**. The only external
data used is nflverse weekly roster identity already in the #665 source audit.
No FantasyAlarm re-acquisition, outcomes, paid odds, models or production change.
This is additive to the reused-provider-ID veto and historical-article
content-version proof requirements; no source/model gate can clear here.
