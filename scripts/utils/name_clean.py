"""Name cleaning helpers shared across the pipeline."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

import pandas as pd


_SUFFIX_RE = re.compile(r"\b(JR|SR|II|III|IV|V)\b\.?", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACES_RE = re.compile(r"\s+")


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
    )


def _normalize_tokens(raw: str) -> str:
    raw = raw or ""
    raw = _strip_accents(str(raw))
    raw = _SUFFIX_RE.sub("", raw)
    raw = _PUNCT_RE.sub(" ", raw)
    raw = _SPACES_RE.sub(" ", raw).strip()
    return raw.lower()


def _title(value: str) -> str:
    return " ".join(token.capitalize() for token in value.split())


def _snake(value: str) -> str:
    return _SPACES_RE.sub("_", value.strip().lower())


def canonical_player(name: Optional[str]) -> str:
    """Normalize a display name into a lowercase "first last" token."""

    if not name:
        return ""
    normalized = _normalize_tokens(str(name))
    tokens = normalized.split()
    if len(tokens) >= 2:
        normalized = f"{tokens[0]} {tokens[-1]}"
    elif tokens:
        normalized = tokens[0]
    else:
        normalized = ""
    return normalized


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
