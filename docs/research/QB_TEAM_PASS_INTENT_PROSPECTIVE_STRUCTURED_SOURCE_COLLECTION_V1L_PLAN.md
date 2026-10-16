# QB TEAM PASS INTENT — PROSPECTIVE STRUCTURED SOURCE COLLECTION V1L PLAN

Frozen: 2026-10-11  
Initial target: 2026 regular-season Week 6  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent authorities:
- V1K blinded bounded audit: `BOUNDED_MANUAL_PRECISION_REJECTS_BLANKET_V1G_PROMOTION`.
- V1K closed blanket historical semantic automation under the current free/public design.
- V1I-R1 transport remains qualified separately at 141/158 = 89.2405%.
- QB generic mean/YPA, same-data attempts repackaging, schedule/rest D1, and PBP D2 remain CLOSED.
- M89/M90 and QB C2 remain protected.

## Purpose

Start a clean **prospective** collection stream for genuinely new pregame QB
team-pass-opportunity information without reopening failed historical semantic
automation.

V1L is collection infrastructure and a pregame shadow ledger. It is not a
predictive model and does not alter production projections.

The source question is deliberately narrower than historical V1G:

> Before kickoff, did an official team media source contain a directly
> attributable statement from offensive decision-makers describing a specific
> intended offensive approach, adjustment, or player usage for the upcoming
> opponent?

## Prospective universe

Initial collection:
- season: 2026
- regular-season week: 6
- every scheduled team in Week 6
- schedule identity and kickoff time only
- no results, scores, play-by-play, box scores, model residuals, or sportsbook
  information.

The universe is generated from the repository maintained NFL schedule utility.

## Clean source window

For each Week-6 team:
- upper bound: target Week-6 kickoff;
- lower bound: the later of:
  1. target kickoff minus 7 days; or
  2. the team's immediately prior scheduled game kickoff plus 6 hours.

The prior-game time is schedule identity only. No prior-game result is read.

This prevents Week-5 pregame material from entering the Week-6 prospective
ledger merely because it falls inside a calendar lookback.

## Allowed source families

Only first-party official club pages whose canonical URL path contains at least
one of:

- `transcript`
- `transcripts`
- `press-conference`
- `press-conferences`
- `media-availability`
- `media_availability`
- `what-they-said`
- `quotes`

Explicitly excluded even if they contain football vocabulary:
- injury reports;
- game status;
- roster moves / transactions;
- game previews;
- scouting reports;
- storylines;
- keys to victory / what-to-watch analysis;
- staff predictions;
- generic news analysis;
- postgame material.

The source-family restriction is frozen before Week-6 outcomes exist.

## Discovery transport

Reuse the qualified first-party V1C direct transport:
1. official club `robots.txt`;
2. same-domain declared sitemaps;
3. same-domain sitemap fallbacks;
4. same-domain recursive sitemap enumeration.

No public search-engine HTML.
No external search API.
No local-publisher fallback in V1L.

V1L prioritizes precision over historical breadth.

## Candidate qualification

A source enters the prospective review queue only if:

1. first-party club domain;
2. allowed source-family URL;
3. absolute embedded publication time is parseable;
4. publication time is inside the clean source window and strictly pre-kickoff;
5. target opponent is identifiable in URL/title/visible source text using the
   already-frozen team synonym vocabulary;
6. page is not postgame/result material.

No semantic model may auto-finalize intent.

## Candidate ranking / cap

Rank eligible source pages by:
1. explicit opponent token in URL;
2. exact target-week token in URL;
3. publication time closest to kickoff while still pre-kickoff;
4. canonical URL ascending.

Retain at most **3 sources per team-week**.

This cap bounds prospective review burden.

## Prospective human-review schema

Each retained source is reviewed pre-kickoff and assigned exactly one source
decision:

- `EXPLICIT_TARGET_GAME_INTENT_FOUND`
- `NO_EXPLICIT_TARGET_GAME_INTENT`
- `INSUFFICIENT_CONTEXT`

If explicit intent exists, record zero or more structured tags:

- `PASS_VOLUME_UP`
- `PASS_VOLUME_DOWN`
- `RUN_VOLUME_UP`
- `RUN_VOLUME_DOWN`
- `PROTECTION_ADJUSTMENT`
- `TEMPO_UP`
- `TEMPO_DOWN`
- `PERSONNEL_USAGE_CHANGE`
- `TARGET_DISTRIBUTION_CHANGE`
- `OTHER_EXPLICIT_OFFENSIVE_INTENT`

A directional pass/run tag requires direct language supporting that direction.
Do not infer pass rate merely from opponent strengths or analyst recommendations.

Preserve:
- speaker when explicit;
- <=25-word evidence;
- source URL;
- publication time;
- decision timestamp.

## Week-6 initial collection disposition

V1L collection infrastructure is
`PROSPECTIVE_STRUCTURED_COLLECTION_READY`
only if:

1. exact Week-6 scheduled-team universe is generated;
2. every team is attempted;
3. candidate sources satisfy allowed source-family rules;
4. every retained candidate is first-party and timestamp-safe;
5. max 3 candidates/team;
6. review packet contains no outcomes/residuals/sportsbook inputs;
7. no paid OddsAPI calls;
8. no football predictive models fit;
9. production unchanged.

Candidate coverage is **descriptive**, not a pass/fail gate in the first
prospective week.

## Predictive boundary

V1L does not modify QB projections.

No predictive candidate may be tested from Week 6 alone.

Before any predictive test:
- prospective source decisions must be frozen before each kickoff;
- a separately frozen minimum-sample gate must be written before grading;
- outcomes may only be joined after the prospective ledger is immutable;
- the comparison must measure incremental team-pass-opportunity value against
  protected QB C2/M89/M90 without sportsbook inputs.

## Stopping / anti-retest rules

- Do not use Week-5 outcomes to define Week-6 source eligibility.
- Do not backfill V1L rules onto V1G and relabel them historical qualification.
- Do not add source families after seeing Week-6 outcomes.
- Do not use analyst preview prose as direct team intent.
- Do not auto-promote semantic model output.
- Do not alter production.
- Do not pull OddsAPI.

## Integrity

- football outcomes read = 0
- model residuals read = 0
- sportsbook/game-market inputs = false
- paid OddsAPI calls = 0
- Week-5 2026 outcomes read = 0
- predictive football fits = 0
- production changes = false
