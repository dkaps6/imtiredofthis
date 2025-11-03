import re
import unicodedata
from typing import Optional

_ABBR_FIX = {
    # quick disambiguations
    "d. kmet": "cole kmet",  # example pattern if helpful later
}

def _ascii(x: str) -> str:
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()

def canonical_player(name: Optional[str]) -> str:
    """
    Normalize a player display name to a consistent key:
    - ascii fold, lower, trim
    - drop periods/commas/extra spaces
    - collapse middle initials/suffixes (jr, sr, ii, iii, iv)
    - keep "first last" tokens when possible
    """
    if not name:
        return ""
    s = _ascii(name).lower().strip()
    s = re.sub(r"[.,']", " ", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"\b([a-z])\s+", r"\1 ", s)  # collapse lone initials spacing
    s = re.sub(r"\s+", " ", s).strip()
    # prefer last two tokens when more than 2 (first last)
    toks = s.split()
    if len(toks) >= 2:
        s = f"{toks[0]} {toks[-1]}"
    s = _ABBR_FIX.get(s, s)
    return s
