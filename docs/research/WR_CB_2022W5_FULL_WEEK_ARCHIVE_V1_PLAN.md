# 2022 Week 5 Full-Week Archive Verification V1

Availability metadata pilot found exact FantasyAlarm Week-5 snapshot `20221005195142`, which is **before the first 2022 Week-5 kickoff**. This makes it the best discovery-source candidate found so far.

Verify only source provenance:
- recover exact CDX digest for the same timestamp/URL;
- exact Wayback replay must hash to that digest;
- parse explicit WR↔CB pairings from the digest-matched body/JSON-LD;
- require each row still pregame (expected all valid schedule rows because capture precedes first kickoff);
- require exact 2022 Week-5 WR roster identity and exact opponent defensive roster identity;
- no provider bridge/fallback/manual repair;
- no outcomes, sportsbook inputs, editorial matchup grade, or fitting.

If successful, freeze sanitized GSIS pair/alignment rows as immutable discovery input. If digest/body/identity fails, fail closed.
