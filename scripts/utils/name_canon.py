"""Utilities for canonicalizing player names across disparate sources."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
NONALNUM = re.compile(r"[^a-z0-9]+")


def _strip_suffix(s: str) -> str:
    return SUFFIX_RE.sub("", s).strip()


def _proper(s: str) -> str:
    return " ".join(part.capitalize() for part in s.split())


def make_player_key(name: str) -> str:
    """Create a stable key for matching player strings from multiple sources."""

    if not name:
        return ""
    raw = _strip_suffix(str(name))
    raw = raw.replace(",", " ").replace(".", " ").replace("-", " ")
    tokens = [token for token in raw.split() if token]
    if not tokens:
        return ""
    last = tokens[-1].lower()
    first = tokens[0].lower()
    first_initial = first[0]
    return f"{NONALNUM.sub('', last)}_{NONALNUM.sub('', first_initial)}"


def most_common_full_name(names: Iterable[str]) -> str | None:
    """Return the most common pretty full name from an iterable of spellings."""

    normed = [_proper(_strip_suffix(name)) for name in names if name]
    if not normed:
        return None
    return Counter(normed).most_common(1)[0][0]
