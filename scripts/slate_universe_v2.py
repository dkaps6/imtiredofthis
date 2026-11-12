"""Build the active PlayerForm slate universe from football roster + schedule only."""
from __future__ import annotations

from typing import Callable

import os
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.player_identity_v3 import player_name_key
from scripts.utils.eligible_team_set_v1 import expected_current_teams


SKILL_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def _depth_chart_universe(season: int) -> pd.DataFrame:
    """Current-season depth-chart skill players: team, name, position, rank.

    FOOTBALL information only. This is nflverse's published depth chart, not
    sportsbook posting coverage, so it does not breach the contract above --
    which forbids the book from defining the universe, not the use of a second
    football roster source.

    Returns an empty frame on any provider failure. The universe then resolves
    exactly as it did before, so this cannot introduce a new failure mode.
    """
    try:
        import nflreadpy as nfl
        raw = nfl.load_depth_charts(seasons=[int(season)])
        raw = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    need = {"team", "player_name", "pos_abb", "pos_rank"}
    if not need.issubset(x.columns):
        return pd.DataFrame()

    # Latest published chart per TEAM. The provider accumulates every chart of
    # the season; stale placements would otherwise re-rank a team off a
    # preseason snapshot.
    if "dt" in x.columns:
        order = pd.to_datetime(x["dt"], errors="coerce", utc=True)
        x = x.assign(_dt=order).dropna(subset=["_dt"])
        if x.empty:
            return pd.DataFrame()
        x = x.loc[x["_dt"].eq(x.groupby("team")["_dt"].transform("max"))]

    x["team"] = x["team"].map(canon_team)
    x["position"] = x["pos_abb"].astype("string").str.upper().str.strip()
    x["rank"] = pd.to_numeric(x["pos_rank"], errors="coerce")
    name = x["player_name"].astype("string").fillna("").str.strip()
    canon = name.map(canonicalize_player_name_safe)
    x["player"] = canon.map(lambda pair: pair[0])
    x["player_clean_key"] = canon.map(lambda pair: pair[1])
    if "gsis_id" in x.columns:
        gid = x["gsis_id"].astype("string").fillna("").str.strip()
        x = x.loc[gid.str.len() > 0]

    x = x.loc[
        x["position"].isin(SKILL_POSITIONS)
        & x["team"].astype(str).str.len().gt(0)
        & x["player"].astype(str).str.len().gt(0)
        & x["player_clean_key"].astype(str).str.len().gt(0)
        & x["rank"].notna()
    ]
    if x.empty:
        return pd.DataFrame()
    x = x.sort_values(["team", "position", "rank"]).drop_duplicates(
        ["team", "player_clean_key"], keep="first"
    )
    return x[["team", "player", "player_clean_key", "position", "rank"]].reset_index(drop=True)


AVAILABILITY_CSV = Path("data") / "current_player_availability.csv"


def _definitively_unavailable() -> set[tuple[str, str]]:
    """(team, player_clean_key) of everyone the availability certification has
    ruled out.

    THIS is the authority, not a hand-rolled read of the injury report.
    validate_playerform_runtime_strict_prior enforces exactly this set against
    PlayerForm's universe, so anything the union re-admits from here is a
    guaranteed step-17 failure -- which is precisely how run 43317130244 died,
    leaking 15 identities including GB RB1 Josh Jacobs, who availability had
    removed and the depth chart put straight back.

    An earlier attempt at this guard read data/injuries.csv for hard OUT
    designations. That was too narrow: definitive_unavailable also covers
    official inactives and other certified non-participation that never appears
    as an OUT status, so it let most of the leak through.

    Empty when the artifact is absent, which is the pre-availability state in
    scripts that build no slate; the union then adds nothing it could not have
    added before.
    """
    try:
        if not AVAILABILITY_CSV.exists() or AVAILABILITY_CSV.stat().st_size == 0:
            return set()
        av = pd.read_csv(AVAILABILITY_CSV, low_memory=False)
    except Exception:
        return set()
    need = {"team", "player_clean_key", "definitive_unavailable"}
    if av.empty or not need.issubset(av.columns):
        return set()
    flag = pd.to_numeric(av["definitive_unavailable"], errors="coerce").fillna(0).eq(1)
    return {(str(t), str(k)) for t, k in zip(av.loc[flag, "team"].map(canon_team),
                                             av.loc[flag, "player_clean_key"])}


UNAVAILABLE_DESIGNATIONS = {"out", "doubtful", "injured reserve", "ir", "pup",
                            "did not play"}


DEPTH_CHART_UNION_ENV = "SLATE_DEPTH_CHART_UNION"


def _depth_chart_union_enabled() -> bool:
    """Explicit production opt-in, OFF by default.

    The union reaches a live provider, so leaving it always-on made the universe
    builder non-deterministic: unit tests that construct a synthetic roster had
    442 real players injected, and -- worse -- the certified-team-scope gate
    stopped raising on a missing team because the depth chart quietly supplied
    it. That gate exists to catch exactly that, so the union must never be able
    to mask it by accident.

    Full Slate sets this to 1. Everything else, including every test that does
    not ask for it, keeps the previous Ourlads-only behaviour.
    """
    return str(os.getenv(DEPTH_CHART_UNION_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _match_keys(name: str) -> set[str]:
    """Every spelling of one person the two roster sources plausibly disagree on.

    Suffix-stripped, plus the same with a LEADING SINGLE-LETTER INITIAL dropped.
    Week 10 carried "Michael Sturdivant" from Ourlads and "J. Michael
    Sturdivant" from the depth chart -- one GB receiver who entered the universe
    twice, as WR9 and WR5, splitting his target share between two rows.

    Deliberately NOT a surname + first-initial match, which is what the
    pre-spend parity audit uses. That is too aggressive here: Atlanta carries
    both Bijan Robinson and Brian Robinson Jr., two different people it would
    merge into one. A leading lone initial is a formatting variant; a full
    first name is identity.
    """
    base = player_name_key(name, strip_suffix=True)
    keys = {base}
    tokens = str(name).split()
    if len(tokens) > 2 and len(tokens[0].strip(".")) == 1:
        keys.add(player_name_key(" ".join(tokens[1:]), strip_suffix=True))
    return {k for k in keys if k}


def _unavailable_designation_keys() -> set[str]:
    """Suffix-stripped keys of players with a hard unavailable GAME designation.

    Reads the current injuries artifact. Returns an empty set when it is absent
    or carries no designations, which is the legitimate midweek state before
    clubs publish on Friday -- the depth-chart union then relies on the
    availability-certified roles subset alone, exactly as it did before.
    """
    path = Path("data") / "injuries.csv"
    try:
        if not path.exists() or path.stat().st_size == 0:
            return set()
        inj = pd.read_csv(path, low_memory=False)
    except Exception:
        return set()
    if inj.empty or "status" not in inj.columns:
        return set()
    name_col = next((c for c in ("player", "full_name", "player_name") if c in inj.columns), None)
    if name_col is None:
        return set()
    status = inj["status"].astype("string").str.strip().str.lower()
    hit = inj.loc[status.isin(UNAVAILABLE_DESIGNATIONS), name_col].dropna()
    return {player_name_key(str(v), strip_suffix=True) for v in hit}


def _season_usage_keys(season: int) -> set[str]:
    """Suffix-stripped name keys of every player with a current-season stat row.

    Football evidence that a player is actually involved, used to keep the
    depth-chart union from importing never-used deep-roster names.
    Empty on provider failure, which simply makes the rank filter the only test.
    """
    try:
        import nflreadpy as nfl
        w = nfl.load_player_stats(seasons=[int(season)])
        w = w.to_pandas() if hasattr(w, "to_pandas") else pd.DataFrame(w)
    except Exception:
        return set()
    if w is None or w.empty or "player_display_name" not in w.columns:
        return set()
    if "season_type" in w.columns:
        w = w.loc[w["season_type"].astype(str).str.upper().eq("REG")]
    return {player_name_key(v, strip_suffix=True) for v in w["player_display_name"].dropna()}


def augment_roles_with_depth_chart(roles: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    """Union the Ourlads role universe with the current depth chart.

    Closes a real correctness defect. In 2026 week 10 the Ourlads-derived
    universe held NYG RB1 = Devin Singletary while nflverse had Cam Skattebo --
    who practiced in FULL, carries 418 rushing yards, and started that night --
    and it had CLE QB1 = Shedeur Sanders rather than Deshaun Watson. Both
    starters were absent from the universe entirely, so materialize_live_props
    fail-closed on ten priced rows naming five players PlayerForm did not know.
    Ourlads tracks who played LAST week; the depth chart is the forward plan,
    and three of the five had missed recent games.

    Rules, deliberately narrow:
      * a (team, player) the depth chart has and Ourlads lacks is ADDED;
      * within a (team, position) the depth chart covers, ITS rank orders the
        role, so a demoted starter is restored rather than merely present;
      * an Ourlads player the depth chart omits keeps his place, appended after
        the ranked ones in his existing relative order -- nobody is dropped;
      * the result must be a strict superset of the input, or it raises.
    """
    stats = {"depth_rows": 0, "added": 0, "reranked": 0, "available": False,
             "enabled": _depth_chart_union_enabled()}
    if not stats["enabled"]:
        return roles, stats
    if roles is None or roles.empty:
        return roles, stats
    dc = _depth_chart_universe(int(season))
    if dc.empty:
        return roles, stats
    stats.update(available=True, depth_rows=int(len(dc)))

    base = roles.copy()
    have = set(zip(base["team"].astype(str), base["player_clean_key"].astype(str)))
    # Match suffix-INSENSITIVELY when deciding whether Ourlads already has a
    # player. The two sources disagree on generational suffixes and initials
    # ("Pierre Strong" vs "Pierre Strong Jr.", "JJ McCarthy" vs "J.J.
    # McCarthy"), and a strict key comparison would add the same person twice
    # and split his opportunity between two rows.
    have_base = {(str(t), k) for t, n in zip(base["team"], base["player"])
                 for k in _match_keys(str(n))}
    dc = dc.assign(base_key=[player_name_key(v, strip_suffix=True) for v in dc["player"]])
    add = dc.loc[[
        (str(t), str(k)) not in have
        and not (_match_keys(str(n)) & {k2 for (t2, k2) in have_base if t2 == str(t)})
        for t, k, n in zip(dc["team"], dc["player_clean_key"], dc["player"])
    ]].copy()

    # Require football evidence before adding a player, measured on week 10:
    # the raw union added 218, of whom 62 had NO current-season stat row and the
    # tail ran to depth rank 11. A WR8 with no usage still draws an
    # empirical-Bayesian baseline shrunk to a position prior, so he takes a
    # share of a finite target pool he will never see snaps for -- diluting the
    # real contributors this fix exists to restore. Rank <= 4 covers where
    # priced markets actually live, and any usage at all overrides rank.
    if not add.empty:
        before_rank = len(add)
        add = add.loc[(add["rank"] <= 4) | add["base_key"].isin(_season_usage_keys(int(season)))]
        stats["dropped_no_evidence"] = int(before_rank - len(add))

    # NEVER re-admit a player the availability certification withheld.
    #
    # The roles artifact is already an availability/timing-certified subset, so
    # a plain union silently bypasses that gate. Measured on week 10 it
    # re-admitted TreVeyon Henderson (NE, RB2) and Colson Yankoff (WAS, TE4),
    # both designated OUT -- the exact defect class that
    # OUT_DOUBTFUL_OPPORTUNITY_RETENTION = 0.0 was shipped to fix. Restoring a
    # mis-ranked healthy starter must not come bundled with pricing players who
    # are not playing.
    #
    # Only hard game designations exclude. A player who merely did not practise
    # has no designation until Friday and is usually active, and treating
    # midweek practice as unavailability would drop genuine WR1s -- week 10 had
    # nine such, including Puka Nacua and Sam Darnold.
    if not add.empty:
        before_av = len(add)
        # The certified availability artifact is the authority the downstream
        # validator enforces, so it is the one the union must respect.
        certified_out = _definitively_unavailable()
        if certified_out:
            add = add.loc[[
                (str(t), str(k)) not in certified_out
                for t, k in zip(add["team"], add["player_clean_key"])
            ]]
        # Hard injury designations as a second line, for the case where the
        # availability artifact is not on disk yet.
        designated_out = _unavailable_designation_keys()
        if designated_out:
            add = add.loc[~add["base_key"].isin(designated_out)]
        stats["dropped_unavailable"] = int(before_av - len(add))

    if not add.empty:
        new = pd.DataFrame({
            "team": add["team"].to_numpy(),
            "player": add["player"].to_numpy(),
            "player_clean_key": add["player_clean_key"].to_numpy(),
        })
        for col in base.columns:
            if col not in new.columns:
                new[col] = add["position"].to_numpy() if col == "position" else pd.NA
        base = pd.concat([base, new[base.columns]], ignore_index=True, sort=False)
        stats["added"] = int(len(add))

    # Re-rank roles where the depth chart covers the team/position.
    if "role" in base.columns and "position" in base.columns:
        rank_map = {(str(r.team), str(r.player_clean_key)): float(r.rank)
                    for r in dc.itertuples(index=False)}
        rank_base = {(str(r.team), str(r.base_key)): float(r.rank)
                     for r in dc.itertuples(index=False)}
        covered = {(str(r.team), str(r.position)) for r in dc.itertuples(index=False)}
        base["_pos"] = base["position"].astype("string").str.upper().str.strip()
        base["_rank"] = [
            rank_map.get(
                (str(t), str(k)),
                rank_base.get((str(t), player_name_key(str(n), strip_suffix=True)), float("inf")),
            )
            for t, k, n in zip(base["team"], base["player_clean_key"], base["player"])
        ]
        base["_orig"] = range(len(base))
        out = []
        for (team, pos), grp in base.groupby(["team", "_pos"], dropna=False, sort=False):
            if (str(team), str(pos)) in covered and str(pos) in SKILL_POSITIONS:
                grp = grp.sort_values(["_rank", "_orig"]).copy()
                before = grp["role"].astype("string").fillna("").tolist()
                grp["role"] = [f"{pos}{i}" for i in range(1, len(grp) + 1)]
                after = grp["role"].astype("string").fillna("").tolist()
                stats["reranked"] += int(sum(1 for a, b in zip(before, after) if a != b))
            out.append(grp)
        base = pd.concat(out, ignore_index=True, sort=False).sort_values("_orig")
        base = base.drop(columns=["_pos", "_rank", "_orig"])

    base = base.drop_duplicates(["team", "player_clean_key"], keep="first").reset_index(drop=True)
    still = set(zip(base["team"].astype(str), base["player_clean_key"].astype(str)))
    lost = have - still
    if lost:
        raise RuntimeError(
            f"depth-chart augmentation dropped {len(lost)} Ourlads players, which it must "
            f"never do: {sorted(lost)[:10]}"
        )
    return base, stats


def build_slate_universe(
    pf,
    load_schedule: Callable[[], pd.DataFrame],
    season: int,
    week: int,
    *,
    live_odds_enabled: bool,
) -> pd.DataFrame:
    """Return one football-defined player row per team/player for the active week.

    Production contract:
    - sportsbook posting coverage NEVER defines which players the football model knows;
    - when ACTIVE_ROLES_CSV is configured, that football-only availability/timing
      artifact defines the certified current team/player universe;
    - otherwise the active player universe is current Ourlads offensive-skill roles;
    - the authoritative schedule supplies opponent identity;
    - live props, when enabled, are validated/quarantined separately downstream;
    - unresolved current team/opponent/player identity is fatal.

    ``live_odds_enabled`` is retained only for runtime logging/backward-compatible
    call signatures. It has no effect on the PlayerForm universe.
    """
    roles = pf._load_roles()
    if roles.empty:
        raise RuntimeError("Ourlads role universe is empty")
    name_col = "display_name" if "display_name" in roles.columns else "player" if "player" in roles.columns else None
    if name_col is None:
        raise RuntimeError(f"Ourlads normalized role universe missing display/player name column: {list(roles.columns)}")
    required_roles = {"team", "player_clean_key"}
    missing_roles = required_roles - set(roles.columns)
    if missing_roles:
        raise RuntimeError(f"Ourlads role universe missing columns: {sorted(missing_roles)}")

    roles = roles.copy()
    roles["team"] = roles["team"].map(canon_team)
    roles["player"] = roles[name_col].astype("string").fillna("").str.strip()
    player = roles["player"]
    key = roles["player_clean_key"].astype("string").fillna("").str.strip()
    if roles["team"].eq("").any() or player.eq("").any() or key.eq("").any():
        raise RuntimeError("Ourlads role universe contains unresolved team/player identity")
    if roles.duplicated(["team", "player_clean_key"]).any():
        sample = roles.loc[
            roles.duplicated(["team", "player_clean_key"], keep=False),
            ["team", "player", "player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"Ourlads role universe contains duplicate current identities: {sample}")

    # Union with the current depth chart BEFORE the schedule merge, so added
    # players pick up opponent identity and role metadata the same way.
    roles, dc_stats = augment_roles_with_depth_chart(roles, int(season))
    print("[slate_universe_v2] depth_chart_union=" + str(dc_stats))

    schedule = load_schedule()
    required_sched = {"season", "week", "team", "opponent"}
    missing_sched = required_sched - set(schedule.columns)
    if missing_sched:
        raise RuntimeError(f"active schedule missing columns: {sorted(missing_sched)}")
    cur = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(week)),
        ["team", "opponent"],
    ].copy()
    cur["team"] = cur["team"].map(canon_team)
    cur["opponent"] = cur["opponent"].map(canon_team)
    if cur.empty:
        raise RuntimeError(f"No schedule rows available for active slate season={season} week={week}")
    if cur["team"].eq("").any() or cur["opponent"].eq("").any():
        raise RuntimeError("active schedule contains unresolved team/opponent identity")
    if cur["team"].duplicated().any():
        dupes = cur.loc[cur["team"].duplicated(keep=False)].to_dict("records")
        raise RuntimeError(f"Active schedule is not unique by team: {dupes[:20]}")

    # Football model identity is independent of sportsbook posting coverage.
    # The roles source may itself be the certified current availability/timing
    # subset, which is the authoritative football universe for this run.
    base = roles[["player", "player_clean_key", "team"]].copy()
    base = base.merge(cur, on="team", how="inner", validate="many_to_one")
    if base.empty:
        raise RuntimeError("Ourlads + active schedule produced zero PlayerForm players")

    missing = (
        base["team"].isna()
        | base["team"].astype("string").str.strip().eq("")
        | base["opponent"].isna()
        | base["opponent"].astype("string").str.strip().eq("")
        | base["player_clean_key"].isna()
        | base["player_clean_key"].astype("string").str.strip().eq("")
    )
    if missing.any():
        path = pf.DATA / "_debug" / "player_form_unresolved_slate_identity.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        base.loc[missing].to_csv(path, index=False)
        raise RuntimeError(
            f"PlayerForm slate identity unresolved for {int(missing.sum())} rows; see {path}"
        )

    # Attach full current Ourlads metadata after identity/schedule resolution.
    role_meta = roles.drop(columns=["player"], errors="ignore")
    base = base.merge(role_meta, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    base["season"] = int(season)
    base["week"] = int(week)
    base = base.drop_duplicates(["team", "player_clean_key"])

    observed = {canon_team(x) for x in base["team"].dropna().astype(str)}
    observed.discard("")
    expected = expected_current_teams()
    if expected is not None:
        missing_teams = sorted(expected - observed)
        extra_teams = sorted(observed - expected)
        if missing_teams or extra_teams:
            raise RuntimeError(
                "PlayerForm current-team universe != certified current roles; "
                f"missing={missing_teams} extra={extra_teams}"
            )
    elif len(observed) < 24:
        # Legacy/non-availability mode retains the historical corruption floor.
        raise RuntimeError(f"PlayerForm active slate universe has implausible team coverage: {len(observed)}")

    print(
        "[slate_universe_v2] sportsbook_independent=1 "
        f"live_odds_enabled={int(bool(live_odds_enabled))} season={season} week={week} "
        f"players={len(base)} teams={len(observed)} "
        f"source=OURLADS_PLUS_DEPTHCHART_PLUS_SCHEDULE depth_union={dc_stats}"
    )
    return base
