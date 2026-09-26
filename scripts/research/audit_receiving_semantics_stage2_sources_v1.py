#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT = Path("data/backtests")
UNIVERSE = ROOT / "pregame_universe"
OUT = Path("outputs/receiving_rule_semantics_stage2_source_audit")

def read_csv(path):
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x

def main():
    team = read_csv(ROOT / "team_weekly_history.csv")
    files = sorted(UNIVERSE.glob("2025_week_*.csv"))
    if not files:
        raise RuntimeError("no 2025 pregame universe files")
    frames = []
    for p in files:
        x = read_csv(p)
        x["source_file"] = p.name
        frames.append(x)
    u = pd.concat(frames, ignore_index=True)
    pos = u.get("position", pd.Series("", index=u.index)).fillna("").astype(str).str.upper().str.strip()
    role = u.get("role", pd.Series("", index=u.index)).fillna("").astype(str).str.upper().str.strip()
    src = u.get("pregame_source", pd.Series("", index=u.index)).fillna("").astype(str)

    wrmask = pos.isin(["WR","LWR","RWR","SWR"]) | pos.str.startswith("WR")
    alignment_like = wrmask & (
        pos.isin(["LWR","RWR","SWR"]) |
        role.str.contains("SWR|LWR|RWR|SLOT", regex=True, na=False)
    )

    summary = {
        "season": 2025,
        "weeks": len(files),
        "team_weekly_rows": int(len(team)),
        "team_weekly_columns": list(team.columns),
        "middle_open_rate_present": bool("middle_open_rate" in team.columns),
        "coverage_man_rate_present": bool("coverage_man_rate" in team.columns or "man_rate" in team.columns),
        "coverage_zone_rate_present": bool("coverage_zone_rate" in team.columns or "zone_rate" in team.columns),
        "pregame_rows": int(len(u)),
        "wr_rows": int(wrmask.sum()),
        "alignment_like_wr_rows": int(alignment_like.sum()),
        "swr_position_rows": int((pos=="SWR").sum()),
        "swr_role_rows": int(role.str.contains("SWR|SLOT", regex=True, na=False).sum()),
        "pregame_sources": {str(k): int(v) for k,v in src.value_counts(dropna=False).to_dict().items()},
        "weeks_with_alignment_like_wr": sorted(
            int(w) for w in pd.to_numeric(u.loc[alignment_like, "week"], errors="coerce").dropna().unique()
        ) if "week" in u.columns else [],
        "sportsbook_inputs_used": 0,
        "target_game_outcomes_read": 0,
        "candidate_variants_scored": 0,
    }
    if not summary["middle_open_rate_present"]:
        a1 = "STAGE2_A1_HISTORICAL_SOURCE_UNAVAILABLE"
    else:
        a1 = "STAGE2_A1_HISTORICAL_SOURCE_AVAILABLE"
    if summary["alignment_like_wr_rows"] > 0 and any("week_tagged_depth_chart" in s for s in summary["pregame_sources"]):
        b1 = "STAGE2_B1_HISTORICAL_SOURCE_AVAILABLE"
    else:
        b1 = "STAGE2_B1_HISTORICAL_SOURCE_UNAVAILABLE"
    summary["a1_disposition"] = a1
    summary["b1_disposition"] = b1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    u.loc[alignment_like].to_csv(OUT / "alignment_like_wr_rows.csv", index=False)
    pd.DataFrame({"column": team.columns}).to_csv(OUT / "team_weekly_columns.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
