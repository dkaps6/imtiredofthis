# scripts/fetch_props_oddsapi.py
import os, argparse, pathlib, pandas as pd, requests, time

CANDIDATE_PLAYER_MARKETS = [
    "player_pass_yards","player_rush_yards","player_rec_yards","player_receptions",
    "player_rush_rec_yards","player_anytime_td","player_pass_tds",
    "player_passing_yards","player_rushing_yards","player_receiving_yards",
    "player_rush_and_receive_yards","player_two_or_more_tds","player_2_or_more_tds",
]
GAME_MARKETS = ["h2h","spreads","totals"]

def _write(df, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _call(url, params, key):
    params = dict(params)  # copy to avoid mutating caller
    params["apiKey"] = key                           # ✅ add the API key
    r = requests.get(url, params=params, timeout=45)
    for k in ("x-requests-remaining","x-requests-used"):
        if k in r.headers:
            print(f"[oddsapi] {k}: {r.headers[k]}")
    if r.status_code == 422:
        raise ValueError("422")
    r.raise_for_status()
    return r.json()

def _fetch_market(url, key, books, api_key):
    params={'regions':'us','oddsFormat':'american','bookmakers':books,'markets':key}
    try:
        print(f"[oddsapi] probe market={key}")
        data=_call(url, params, api_key)
        return True, data
    except ValueError:
        print(f"[oddsapi] 422 unsupported market={key}")
        return False, None
    except requests.HTTPError as e:
        print(f"[oddsapi] http {e.response.status_code} market={key}: {e}")
        return False, None
    except Exception as e:
        print(f"[oddsapi] error market={key}: {e}")
        return False, None

def _flatten_props(payload):
    cols=['player','team','opp_team','event_id','market','line','over_odds','under_odds','book','commence_time','sport_key','position']
    rows=[]
    for g in payload:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                if not str(mkey).startswith("player_"):  # keep only player markets here
                    continue
                for ou in mk.get('outcomes', []):
                    name=ou.get('description') or ou.get('name')
                    team=(ou.get('team') or '').upper()
                    opp=away if team==home else home if team==away else ''
                    line=ou.get('point') if 'point' in ou else (1.0 if 'anytime_td' in mkey else None)
                    side=(ou.get('name','') or '').lower()
                    price=ou.get('price')
                    over_odds=price if side in ('over','yes') else None
                    under_odds=price if side in ('under','no') else None
                    rows.append([name,team,opp,g.get('id'),mkey,line,over_odds,under_odds,bk_key,g.get('commence_time'),g.get('sport_key','nfl'),ou.get('position') or ''])
    return pd.DataFrame(rows, columns=cols)

def _flatten_games(payload):
    cols=['event_id','commence_time','sport_key','home_team','away_team','market','point','book']
    rows=[]
    for g in payload:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                if mkey not in ("h2h","spreads","totals"):
                    continue
                point=None
                if mk.get('outcomes'):
                    point = mk['outcomes'][0].get('point')
                rows.append([g.get('id'), g.get('commence_time'), g.get('sport_key'), home, away, mkey, point, bk_key])
    return pd.DataFrame(rows, columns=cols)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books', default='draftkings,fanduel,betmgm,caesars')
    ap.add_argument('--date', default='')  # kept for CLI compat; not used to avoid 422
    ap.add_argument('--out', default='outputs/props_raw.csv')
    a=ap.parse_args()

    api_key=os.getenv('ODDS_API_KEY','').strip()
    if not api_key:
        print('[oddsapi] key missing; writing empty CSVs')
        _write(pd.DataFrame(), a.out); _write(pd.DataFrame(), "outputs/odds_game.csv"); return 0

    base='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'

    # 1) game markets
    try:
        print(f"[oddsapi] request game markets={','.join(GAME_MARKETS)}")
        game_json=_call(base, {'regions':'us','oddsFormat':'american','bookmakers':a.books,'markets':','.join(GAME_MARKETS)}, api_key)
    except Exception as e:
        print(f"[oddsapi] game error: {e}"); game_json=[]

    games=_flatten_games(game_json)
    _write(games, "outputs/odds_game.csv")
    print(f"[oddsapi] wrote games={len(games)}")

    # 2) per-market player probing
    all_payload=[]; supported=[]
    for mk in CANDIDATE_PLAYER_MARKETS:
        ok, data = _fetch_market(base, mk, a.books, api_key)
        if ok and data:
            supported.append(mk)
            all_payload.extend(data)
            time.sleep(0.35)  # gentle pacing

    if supported:
        print(f"[oddsapi] supported player markets: {supported}")
    else:
        print("[oddsapi] no supported player markets detected")

    props = _flatten_props(all_payload)
    _write(props, a.out)
    print(f"[oddsapi] wrote props={len(props)}")
    return 0

if __name__=='__main__':
    main()
