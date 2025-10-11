# scripts/make_metrics.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def _safe_read_csv(p: str) -> pd.DataFrame:
    fp = Path(p)
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()

def _z(col: pd.Series) -> pd.Series:
    x = col.astype(float)
    return (x - x.mean()) / (x.std(ddof=1) + 1e-9)

def _implied_prob_from_american(odds):
    # American to implied probability (vigged). Use later for win prob proxy.
    o = pd.to_numeric(odds, errors="coerce")
    prob = np.where(o > 0, 100 / (o + 100), -o / (-o + 100))
    return pd.to_numeric(prob, errors="coerce")

def build_metrics(season: int) -> pd.DataFrame:
    # base inputs
    team_form = _safe_read_csv("data/team_form.csv")          # expects def_* EPA/sack/pressure/pace/proe/box rates
    player_form = _safe_read_csv("data/player_form.csv")      # shares, yprr, ypc priors, role info
    roles = _safe_read_csv("data/roles.csv")
    injuries = _safe_read_csv("data/injuries.csv")            # player/team/status
    coverage = _safe_read_csv("data/coverage.csv")            # defense_team, tag in {top_shadow,heavy_man,heavy_zone}
    cb_assign = _safe_read_csv("data/cb_assignments.csv")     # defense_team, receiver, cb, quality/penalty
    weather = _safe_read_csv("data/weather.csv")              # event_id, wind_mph, temp_f, precip, altitude_ft
    odds_game = _safe_read_csv("outputs/odds_game.csv")       # event_id, book, market (h2h/spreads/totals), price_american, point

    # ---- Team-level normalizations ----
    if not team_form.empty:
        # Compute z-scores where present
        for c in ["def_pass_epa","def_rush_epa","def_sack_rate","def_pressure_rate",
                  "light_box_rate","heavy_box_rate","pace","proe","ay_per_att"]:
            if c in team_form.columns:
                team_form[f"{c}_z"] = _z(team_form[c])
        # Fallback if proe missing
        if "proe" not in team_form.columns:
            team_form["proe"] = 0.0
            team_form["proe_z"] = 0.0

    # ---- Injuries: cap WR1 + redistribute flags ----
    if not injuries.empty:
        injuries["status_flag"] = injuries["status"].str.lower().isin(["out","doubtful","questionable","limited"]).astype(int)

    # ---- Coverage tags to wide flags ----
    if not coverage.empty:
        tag_pivot = (coverage
                     .assign(val=1)
                     .pivot_table(index="defense_team", columns="tag", values="val", aggfunc="max", fill_value=0)
                     .reset_index())
    else:
        tag_pivot = pd.DataFrame(columns=["defense_team","top_shadow","heavy_man","heavy_zone"])

    # ---- CB assignments kept for WR-specific penalties later ----
    if cb_assign.empty:
        cb_assign = pd.DataFrame(columns=["defense_team","receiver","cb","quality","penalty"])

    # ---- Weather keyed by event_id ----
    if not weather.empty:
        weather["wind_mph"] = pd.to_numeric(weather.get("wind_mph"), errors="coerce")
        weather["temp_f"] = pd.to_numeric(weather.get("temp_f"), errors="coerce")
        weather["precip"] = weather.get("precip").astype(str).str.lower()
    else:
        weather = pd.DataFrame(columns=["event_id","wind_mph","temp_f","precip","altitude_ft"])

    # ---- Get rough win-prob from h2h odds to drive script escalators ----
    # Use first book’s h2h for simplicity
    if not odds_game.empty:
        h2h = odds_game[odds_game["market"]=="h2h"].copy()
        # choose the shortest price per event/team
        h2h["prob"] = _implied_prob_from_american(h2h["price_american"])
        h2h = (h2h.sort_values(["event_id","team","prob"], ascending=[True,True,False])
                  .drop_duplicates(["event_id","team"]))
        winp = h2h[["event_id","team","prob"]].rename(columns={"prob":"win_prob"})
    else:
        winp = pd.DataFrame(columns=["event_id","team","win_prob"])

    # ---- Merge everything to a player-game frame skeleton ----
    # We expect player_form to have at least: player, team, position/role priors
    df = player_form.copy()
    if "team" not in df.columns:
        df["team"] = ""
    if "player" not in df.columns:
        df["player"] = ""

    # merge team-level onto players
    team_keys = [c for c in ["team"] if c in team_form.columns]
    if team_keys:
        df = df.merge(team_form, on="team", how="left", suffixes=("","_team"))

    # attach injuries
    if not injuries.empty:
        df = df.merge(injuries[["player","status","status_flag"]], on="player", how="left")

    # attach roles if provided
    if not roles.empty and "player" in roles.columns:
        df = df.merge(roles, on="player", how="left", suffixes=("","_role"))

    # attach coverage flags by defense opponent if you store opp team column; else keep zeros
    for x in ["top_shadow","heavy_man","heavy_zone"]:
        df[x] = 0
    if "opp_team" in df.columns:
        df = df.merge(tag_pivot, left_on="opp_team", right_on="defense_team", how="left")
        for x in ["top_shadow","heavy_man","heavy_zone"]:
            if x in df.columns:
                df[x] = df[x].fillna(0).astype(int)

    # attach win prob if you store event_id + team
    if "event_id" in df.columns and not winp.empty:
        df = df.merge(winp, left_on=["event_id","team"], right_on=["event_id","team"], how="left")
    else:
        df["win_prob"] = np.nan

    # attach weather
    if "event_id" in df.columns and not weather.empty:
        df = df.merge(weather[["event_id","wind_mph","temp_f","precip"]], on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip"]:
            if c not in df.columns: df[c] = np.nan

    # ---- Derived features (exact rules you specified) ----
    # Pressure-adjusted QB baseline proxy, sack elasticity, funnels, weather multipliers, volatility flags
    def qb_pressure_mult(row):
        # needs def_pressure_rate_z and def_pass_epa_z on opponent
        pz = row.get("def_pressure_rate_z", 0.0)
        ez = row.get("def_pass_epa_z", 0.0)
        m = (1 - 0.35 * (pz if pd.notna(pz) else 0.0)) * (1 - 0.25 * (ez if pd.notna(ez) else 0.0))
        return max(m, 0.6)  # clamp a bit

    def sack_elasticity(row):
        z = row.get("def_sack_rate_z", 0.0)
        return 1 - 0.15 * (z if pd.notna(z) else 0.0)

    def run_funnel(row):
        # run funnel when def_rush_epa is poor (>=60th pct) and pass is good (<=40th pct)
        rp = row.get("def_rush_epa_z", 0.0)
        pp = row.get("def_pass_epa_z", 0.0)
        return int((rp >= 0.253) and (pp <= -0.253))

    def pass_funnel(row):
        rp = row.get("def_rush_epa_z", 0.0)
        pp = row.get("def_pass_epa_z", 0.0)
        return int((pp >= 0.253) and (rp <= -0.253))

    def weather_mult(row, market_hint: str):
        w = 1.0
        wind = row.get("wind_mph")
        precip = str(row.get("precip", "")).lower()
        if pd.notna(wind) and wind >= 15:
            if market_hint in ("pass","rec","rush_rec"):
                w *= 0.94
        if precip in {"rain","snow"}:
            if market_hint in ("rec","rush_rec"):
                w *= 0.97
            if market_hint in ("rush","rush_att"):
                w *= 1.02
        return w

    # Add feature columns
    if "position" not in df.columns:
        df["position"] = df.get("role","")  # fallback

    df["qb_pressure_mult"] = df.apply(lambda r: qb_pressure_mult(r), axis=1)
    df["sack_elasticity"] = df.apply(lambda r: sack_elasticity(r), axis=1)
    df["run_funnel"] = df.apply(lambda r: run_funnel(r), axis=1)
    df["pass_funnel"] = df.apply(lambda r: pass_funnel(r), axis=1)

    # Script escalators: rough – if favored (win_prob >= .55) → RB attempts bump
    df["rb_attempts_escalator"] = np.where(df["win_prob"] >= 0.55, 3.0, 0.0)

    # Weather multipliers as generic hints (specialize downstream per market)
    df["wx_mult_pass"] = df.apply(lambda r: weather_mult(r, "pass"), axis=1)
    df["wx_mult_rec"]  = df.apply(lambda r: weather_mult(r, "rec"), axis=1)
    df["wx_mult_rush"] = df.apply(lambda r: weather_mult(r, "rush"), axis=1)
    df["wx_mult_rush_rec"] = df.apply(lambda r: weather_mult(r, "rush_rec"), axis=1)

    # Volatility flag (pressure mismatch or QB inconsistency proxy)
    df["volatility_flag"] = ((df.get("def_pressure_rate_z",0) > 0.75) | (df.get("def_sack_rate_z",0) > 0.75)).astype(int)

    # Coverage effects: WR vs top shadow / heavy man / heavy zone
    df["wr_shadow_penalty"] = np.where((df["position"].astype(str).str.upper().str.contains("WR")) & (df.get("top_shadow",0)==1), -0.08, 0.0)
    df["slot_zone_boost"]   = np.where((df["position"].astype(str).str.contains("SLOT", case=False)) & (df.get("heavy_zone",0)==1), 0.05, 0.0)

    # Sanity
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    out = "data/metrics_ready.csv"

    df = build_metrics(args.season)
    df.to_csv(out, index=False)
    print(f"[metrics] wrote rows={len(df)} → {out}")

if __name__ == "__main__":
    main()
