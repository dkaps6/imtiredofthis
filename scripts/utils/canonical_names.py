# scripts/utils/canonical_names.py
#
# This module defines a canonical map for known player name variants
# and exposes helper(s) that normalize raw strings into a stable identity.
#
# The map keys should be UPPERCASE alphanumeric (no punctuation/spaces),
# and the values should be the official canonical player name in UPPERCASE
# "FIRST LAST" form.

CANONICAL_NAME_MAP = {
    # Davante Adams variants
    "DADAMS": "DAVANTE ADAMS",
    "DAVANTEADAMS": "DAVANTE ADAMS",
    "DADAMSWR": "DAVANTE ADAMS",  # some props boards suffix position
    "D.ADAMS": "DAVANTE ADAMS",
    "DAADAMS": "DAVANTE ADAMS",

    # Justin Jefferson variants
    "JJEFFERSON": "JUSTIN JEFFERSON",
    "J.JEFFERSON": "JUSTIN JEFFERSON",
    "JUSTINJEFFERSON": "JUSTIN JEFFERSON",
    "JJETTAS": "JUSTIN JEFFERSON",  # common alias

    # CeeDee Lamb variants
    "CDLAMB": "CEEDEE LAMB",
    "C.LAMB": "CEEDEE LAMB",
    "CEEDEELAMB": "CEEDEE LAMB",

    # Puka Nacua variants
    "PNACUA": "PUKA NACUA",
    "P.NACUA": "PUKA NACUA",
    "PUKANACUA": "PUKA NACUA",

    # Tyreek Hill variants
    "THILL": "TYREEK HILL",
    "T.HILL": "TYREEK HILL",
    "TYREEKHILL": "TYREEK HILL",

    # Travis Kelce variants
    "TKELCE": "TRAVIS KELCE",
    "T.KELCE": "TRAVIS KELCE",
    "TRAVISKELCE": "TRAVIS KELCE",

    # Amon-Ra St. Brown variants
    "ASTBROWN": "AMON-RA ST. BROWN",
    "A.STBROWN": "AMON-RA ST. BROWN",
    "AMONRASTBROWN": "AMON-RA ST. BROWN",

    # Ja'Marr Chase variants
    "JCHASE": "JA'MARR CHASE",
    "J.CHASE": "JA'MARR CHASE",
    "JAMARRCHASE": "JA'MARR CHASE",

    # Add more as needed. This list will grow over time.
}


def _clean_raw_name(n: str) -> str:
    """
    Take an arbitrary player name string and normalize it to
    uppercase A-Z with no punctuation or whitespace so it can be
    looked up in CANONICAL_NAME_MAP.
    Examples:
      'D.Adams'       -> 'DADAMS'
      'Davante Adams' -> 'DAVANTEADAMS'
      'P. Nacua'      -> 'PNACUA'
    """
    if n is None:
        return ""
    n = str(n).upper()
    # remove punctuation and whitespace
    for bad in [".", ",", "'", '"', "-", "_", " "]:
        n = n.replace(bad, "")
    return n.strip()


def canonicalize_player_name(raw: str) -> tuple[str, str]:
    """
    Returns a canonical player identity:
    - try direct dictionary hit
    - try fuzzy match
    - else fallback to cleaned full uppercase FIRST LAST style
    We also return a SECOND value (clean_key) which is the normalized
    lookup key so other scripts can join on it.
    """
    from difflib import get_close_matches

    clean_key = _clean_raw_name(raw)

    # 1. direct dict match
    if clean_key in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[clean_key], clean_key

    # 2. fuzzy match against known keys
    possible = get_close_matches(
        clean_key,
        list(CANONICAL_NAME_MAP.keys()),
        n=1,
        cutoff=0.88,
    )
    if possible:
        hit = possible[0]
        return CANONICAL_NAME_MAP[hit], clean_key

    # 3. fallback: convert to "FIRST LAST" uppercase-ish
    # e.g. "DAVANTEADAMS" -> "DAVANTE ADAMS"
    # We'll do a naive split by capital letter boundaries? Too heavy.
    # Instead, just return the cleaned_key as-is. Downstream grouping will
    # still work because it's consistent.
    return clean_key, clean_key


def log_unmapped_variant(raw: str, path: str = "data/unmapped_player_names.txt") -> None:
    """
    Append any unseen / unmapped variant to a file so we can grow CANONICAL_NAME_MAP
    over time without breaking the pipeline. We don't crash here.
    """
    try:
        canonical, clean_key = canonicalize_player_name(raw)
        # If canonical == clean_key and clean_key not in CANONICAL_NAME_MAP,
        # that means we didn't have a confident mapping for a "fancy" alias.
        if clean_key not in CANONICAL_NAME_MAP.values() and clean_key not in CANONICAL_NAME_MAP:
            with open(path, "a") as f:
                f.write(f"{raw} -> {clean_key}\n")
    except Exception:
        # never kill pipeline because of logging
        pass
