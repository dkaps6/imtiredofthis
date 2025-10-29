#!/usr/bin/env python3
# Defensive coverage metrics by TEAM (Man%, Zone%) from Sharp
import io, time, requests, pandas as pd
from pathlib import Path

URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"
UA  = {"User-Agent": "FullSlate/CI (+github-actions)"}

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA",
    "San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}



def fetch_tables() -> list[pd.DataFrame]:
    for attempt in range(3):
        try:
            r = requests.get(URL, headers=UA, timeout=45)
            r.raise_for_status()
            return pd.read_html(io.StringIO(r.text))
        except Exception as e:
            print(f"[cb_team] read_html attempt {attempt+1}/3: {e}")
            time.sleep(2*(attempt+1))
    return []



def pick_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    for t in tables:
        cols = " ".join([str(c).lower() for c in t.columns])
        if "team" in cols and "man" in cols and "zone" in cols:
            return t
    return tables[0] if tables else None



def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()]
    cmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "team" in lc: cmap[c] = "team_name"
        elif "man" in lc: cmap[c] = "man_rate"
        elif "zone" in lc: cmap[c] = "zone_rate"
    df = df.rename(columns=cmap)
    for c in ["man_rate","zone_rate"]:
        if c in df:
            s = (df[c].astype(str).str.replace("%","",regex=False)
                              .str.extract(r"([0-9]+\.?[0-9]*)")[0])
            df[c] = pd.to_numeric(s, errors="coerce")/100.0
    df["team"] = df.get("team_name","").map(TEAM_NAME_TO_ABBR)
    return df.dropna(subset=["team"]) [["team","man_rate","zone_rate"]]



def main():
    out = Path("data") / "cb_coverage_team.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tables = fetch_tables()
    t = pick_table(tables)
    if t is None:
        print("[cb_team] WARN: no tables found, writing header-only CSV.")
        pd.DataFrame(columns=["team","man_rate","zone_rate"]).to_csv(out, index=False); return
    df = clean_numeric(t)
    df.to_csv(out, index=False)
    print(f"[cb_team] wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
