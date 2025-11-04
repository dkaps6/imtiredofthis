"""Name cleaning helpers shared across the pipeline."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

import pandas as pd

try:
    from scripts.utils.team_maps import TEAM_NAME_TO_ABBR
except ImportError:  # pragma: no cover - fallback for partial environments
    TEAM_NAME_TO_ABBR: dict[str, str] = {}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_FIRST_NAME_ALIASES: dict[str, str] = {
    "william": "will",
    "willie": "will",
    "christopher": "chris",
    "nicholas": "nick",
    "nicolas": "nick",
    "joshua": "josh",
    "josh": "josh",
    "matthew": "matt",
    "matt": "matt",
    "dakota": "dak",
    "kenneth": "ken",
    "anthony": "tony",
    "antonio": "tony",
    "trey": "trey",
    "geno": "geno",
    "bo": "bo",
    "brock": "brock",
    "courtland": "courtland",
    "troy": "troy",
    "ashton": "ashton",
}

_INITIAL_LAST_ALIASES: dict[tuple[str, str], str] = {
    ("b", "nix"): "bo",
    ("g", "smith"): "geno",
    ("d", "prescott"): "dak",
    ("k", "walker"): "ken",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACES_RE = re.compile(r"\s+")
_GLUED_INITIAL_RE = re.compile(r"^([A-Z](?:[a-z]?))([A-Z][a-z]+)$")


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
    )


def _strip_suffixes(name: str) -> str:
    t = name.strip()
    t = re.sub(r"[.,]", " ", t)
    parts = [p for p in t.split() if p.lower() not in SUFFIXES]
    return " ".join(parts)


def _norm_spaces(s: str) -> str:
    return _SPACES_RE.sub(" ", s.strip())


def _normalize_for_alias(token: str) -> str:
    token = token or ""
    return re.sub(r"[^a-z]", "", token.lower())


def _alias_first_token(first: str, last: str) -> str:
    if not first:
        return first

    if "-" in first:
        head, *rest = first.split("-")
        context = rest[0] if rest else last
        aliased_head = _alias_first_token(head, context)
        if aliased_head != head:
            return "-".join([aliased_head] + rest)
        return first

    norm_first = _normalize_for_alias(first)
    norm_last = _normalize_for_alias(last)

    if norm_first and len(norm_first) == 1 and norm_last:
        pair = (norm_first, norm_last)
        alias = _INITIAL_LAST_ALIASES.get(pair)
        if alias:
            return alias

    alias = _FIRST_NAME_ALIASES.get(norm_first)
    if alias:
        return alias

    return first


def canonical_player(raw: Optional[str]) -> str:
    """Normalize a player string into a tidy "First Last" representation."""

    if raw is None:
        return ""

    text = _strip_accents(str(raw))
    text = _strip_suffixes(text)

    compact = text.replace(" ", "")
    glued = _GLUED_INITIAL_RE.match(compact)
    if glued and " " not in text:
        prefix = compact[: len(compact) - len(glued.group(2))]
        text = f"{prefix} {glued.group(2)}"

    # Last, First -> First Last
    m = re.match(r"^\s*([A-Za-z'\- ]+)\s*,\s*([A-Za-z'\- ]+)\s*$", text)
    if m:
        text = f"{m.group(2)} {m.group(1)}"

    text = re.sub(r"[^A-Za-z'\- ]+", " ", text)
    text = _norm_spaces(text)
    if not text:
        return ""

    tokens = text.split()
    if tokens:
        last = tokens[1] if len(tokens) > 1 else ""
        tokens[0] = _alias_first_token(tokens[0], last)

    text = " ".join(tokens)
    return text.title()


def canonical_key(name: Optional[str]) -> str:
    """Compact key used across joins (lowercase, punctuation-free)."""

    full = canonical_player(name)
    if not full:
        return ""
    return (
        full.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("'", "")
    )


def initials_last_to_full(
    token: str, team: Optional[str], roster_df: pd.DataFrame
) -> Optional[str]:
    """Resolve tokens like 'J.Williams' → 'Javonte Williams' using team context."""

    if not token or roster_df is None or roster_df.empty:
        return None

    t = str(token).strip()
    if not t:
        return None
    t = t.replace(".", " ")
    t = _norm_spaces(t)
    m = re.match(r"^([A-Za-z])\s*([A-Za-z'\-]+)$", t)
    if not m:
        return None

    init = m.group(1).upper()
    last = m.group(2).title()

    lookup = roster_df
    if "team" in lookup.columns:
        team_norm = None
        if isinstance(team, str):
            team_norm = normalize_team(team)
        elif team is not None and not pd.isna(team):
            team_norm = normalize_team(str(team))
        if team_norm:
            lookup = lookup[lookup["team"] == team_norm]

    cand = lookup[lookup["last_name"] == last]
    if cand.empty:
        return None

    cand = cand[cand["first_name"].str[:1].str.upper() == init]
    if len(cand) == 1:
        return cand.iloc[0]["full_name"]
    return None


def build_roster_lookup(roles: pd.DataFrame) -> pd.DataFrame:
    """Return roster lookup DataFrame with first/last/full names by team."""

    cols = ["team", "first_name", "last_name", "full_name"]
    if roles is None or roles.empty:
        return pd.DataFrame(columns=cols)

    working = roles.copy()
    working.columns = [str(c).lower() for c in working.columns]

    if "player" not in working.columns:
        return pd.DataFrame(columns=cols)

    team_col = next(
        (cand for cand in ("team", "team_abbr", "team_code", "team_name") if cand in working.columns),
        None,
    )

    roster = pd.DataFrame({"player": working["player"]})
    if team_col is not None:
        roster["team"] = working[team_col]
    else:
        roster["team"] = pd.NA

    roster["player"] = roster["player"].astype("string")
    roster["team"] = roster["team"].astype("string")
    roster["team"] = roster["team"].map(normalize_team).astype("string")

    roster["full_name"] = roster["player"].apply(canonical_player)
    roster = roster[roster["full_name"].str.len() > 0].copy()

    if roster.empty:
        return pd.DataFrame(columns=cols)

    sp = roster["full_name"].str.split(" ", n=1, expand=True)
    roster["first_name"] = sp[0].fillna("").astype("string")
    roster["last_name"] = sp[1].fillna("").astype("string")
    roster = roster.drop(columns=["player"])

    roster = roster[cols].drop_duplicates().reset_index(drop=True)
    return roster


def _normalize_tokens(raw: str) -> str:
    raw = raw or ""
    raw = _strip_accents(str(raw))
    raw = _strip_suffixes(raw)
    raw = _PUNCT_RE.sub(" ", raw)
    raw = _SPACES_RE.sub(" ", raw).strip()
    return raw.lower()


def _title(value: str) -> str:
    return " ".join(token.capitalize() for token in value.split())


def _snake(value: str) -> str:
    return _SPACES_RE.sub("_", value.strip().lower())


TEAM_NORMALIZE = {k.upper(): v for k, v in TEAM_NAME_TO_ABBR.items()}
TEAM_NORMALIZE.update({v.upper(): v for v in TEAM_NAME_TO_ABBR.values()})
TEAM_NORMALIZE.update(
    {
        "BYE": "BYE",
        # Cardinals aliases seen across books/APIs/historic sources
        "ARZ": "ARI",
        "AZ": "ARI",
        "ARIZ": "ARI",
        "ARIZONA": "ARI",
        "CRD": "ARI",
        "CARDINALS": "ARI",
        "ARIZONA-CARDINALS": "ARI",
        "ARI CARDINALS": "ARI",
    }
)
def canonicalize(
    df: pd.DataFrame,
    name_cols: Sequence[str],
    team_col: str | None = None,
    roster_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach canonical display/player key columns to *df*.

    Parameters
    ----------
    df: DataFrame to enrich.
    name_cols: Ordered list of potential name columns to use as sources.
    team_col: Optional column name containing team abbreviations (used for roster merge).
    roster_map: Optional lookup DataFrame with columns ['team', 'player_key', 'display_name'].
    """

    if df is None:
        return df

    if not name_cols:
        name_cols = []

    out = df.copy()

    if len(out) == 0:
        out["display_name"] = out.get("display_name", pd.Series(dtype="string"))
        out["player_key"] = out.get("player_key", pd.Series(dtype="string"))
        return out

    base = pd.Series("", index=out.index, dtype="string")
    for col in name_cols:
        if col not in out.columns:
            continue
        series = out[col].fillna("").astype(str).str.strip()
        mask = base.eq("") & series.ne("")
        base = base.where(~mask, series)

    norm = base.apply(_normalize_tokens)
    tokens = norm.str.split()
    two_tok = tokens.apply(lambda ts: " ".join(ts[:2]) if ts else "")

    display = two_tok.apply(_title)
    player_key = two_tok.apply(_snake)

    if "display_name" in out.columns:
        existing = out["display_name"].fillna("").astype(str)
        out["display_name"] = existing.where(existing.str.strip().ne(""), display)
    else:
        out["display_name"] = display

    out["player_key"] = player_key

    if roster_map is not None and not roster_map.empty and team_col and team_col in out.columns:
        roster_cols = [
            col
            for col in ("team", "player_key", "display_name")
            if col in roster_map.columns
        ]
        if set(["team", "player_key", "display_name"]).issubset(roster_cols):
            roster = (
                roster_map.loc[:, ["team", "player_key", "display_name"]]
                .drop_duplicates()
                .rename(columns={"team": "__roster_team"})
            )
            roster["__roster_team"] = roster["__roster_team"].astype(str).str.upper().str.strip()
            roster["player_key"] = roster["player_key"].astype(str)

            working = out.copy()
            working[team_col] = working[team_col].astype(str).str.upper().str.strip()
            working = working.merge(
                roster,
                left_on=[team_col, "player_key"],
                right_on=["__roster_team", "player_key"],
                how="left",
                suffixes=("", "_roster"),
            )
            if "display_name_roster" in working.columns:
                working["display_name"] = working["display_name_roster"].fillna(
                    working["display_name"]
                )
                working.drop(columns=["display_name_roster"], inplace=True)
            working.drop(columns=["__roster_team"], inplace=True, errors="ignore")
            out = working

    return out


def normalize_team(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return x
    k = x.strip().upper().replace(".", "").replace("_", "-")
    return TEAM_NORMALIZE.get(k, TEAM_NORMALIZE.get(k.replace(" ", ""), k))
