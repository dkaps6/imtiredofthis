# scripts/_opponent_map.py
import pandas as pd, re

# --- light, safe utilities used by multiple scripts ---

def normalize_team(s: pd.Series) -> pd.Series:
    """Uppercase + trim; keep abbreviations you already use."""
    return s.astype(str).str.upper().str.strip()

def _read_csv_silent(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_opponent_map(coverage_path: str = "data/coverage_cb.csv") -> dict:
    """
    Build offense_team -> defense_team map from coverage_cb.csv.
    Returns {} if file/columns not present.
    """
    cov = _read_csv_silent(coverage_path)
    need = {"offense_team", "defense_team"}
    if not need.issubset(set(cov.columns)):
        return {}
    cov["offense_team"] = normalize_team(cov["offense_team"])
    cov["defense_team"] = normalize_team(cov["defense_team"])
    return dict(zip(cov["offense_team"], cov["defense_team"]))

def attach_opponent(df: pd.DataFrame,
                    team_col: str = "team",
                    coverage_path: str = "data/coverage_cb.csv",
                    opponent_col: str = "opponent",
                    inplace: bool = True) -> pd.DataFrame:
    """
    Fill/overwrite df[opponent_col] using offense->defense schedule map.
    Leaves df untouched if mapping can't be built.
    """
    opp_map = build_opponent_map(coverage_path)
    if not opp_map or team_col not in df.columns:
        return df
    out = df if inplace else df.copy()
    out[team_col] = normalize_team(out[team_col])
    if opponent_col not in out.columns:
        out[opponent_col] = pd.NA
    # only fill where missing or 'ALL'
    mask = out[opponent_col].isna() | (out[opponent_col].astype(str).str.upper().eq("ALL"))
    out.loc[mask, opponent_col] = out.loc[mask, team_col].map(opp_map)
    return out
