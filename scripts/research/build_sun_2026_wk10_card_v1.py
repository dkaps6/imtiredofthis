import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
from scripts.research.validate_ticket_card_v1 import validate
S='/tmp/claude-0/-home-user-imtiredofthis/1701f8f3-afc9-57ea-bbc6-cce4dc6b5fbd/scratchpad/'
L=pd.read_csv(S+'sun_legs.csv'); L=L[L.book=='fanduel']
dec=lambda a:1+a/100 if a>0 else 1+100/-a
amer=lambda d:int(round((d-1)*100)) if d>=2 else int(round(-100/(d-1)))
REASON={
 ('Josh Allen',149.5):"Allen has cleared 204+ in all 8 games; 150 is a floor bet in the slate's 2nd-highest total (50.5).",
 ('Brenton Strange',14.5):"JAX TE1, 75% snaps, 8/8 over 14.5 (low 17); model 39.5.",
 ('Devaughn Vele',14.5):"NO's every-down WR (92% snaps), 8/8 over 14.5 (low 17); CAR-NO total 48.5.",
 ('Xavier Hutchinson',9.5):"HOU WR, 8/8 over 9.5 (low 10); model 28.",
 ('AJ Barner',4.5):"SEA TE1 (87% snaps), 9/9 over 4.5; model 25.",
 ('Daniel Jones',99.5):"9/9 over 99.5 (low 108). Only the floor rung: his 246 projection is 60 above the book, so no deeper Jones.",
 ('Aaron Rodgers',149.5):"8/8 over 149.5 (low 172) on 98% snaps; SNF, total 49.5.",
 ('Jahan Dotson',4.5):"ATL WR2, 9/9 over 4.5, with Penix (4 straight starts) under center.",
 ('Ryan Flournoy',19.5):"DAL WR, 9/9 over 19.5 (low 22) in the slate's highest total (55.5).",
 ('Bryce Young',249.5):"Young's median is 284 and he's cleared 249.5 in 6 of 8; NO pass D, total 48.5.",
 ('Bryce Young',274.5):"Same thesis one rung up: 6 of 8 over 274.5.",
 ('Kayshon Boutte',14.5):"8/8 over 14.5 (low 16); model 30.",
 ('Kayshon Boutte',24.5):"5/8 over 24.5; model 30 vs CLE.",
 ('Cooper Kupp',14.5):"SEA slot, 7/9 over 14.5; model 24 vs a rookie-QB LV team likely to trail.",
 ('Keenan Allen',19.5):"SF WR, 7/8 over 19.5, model 34 vs book 24.5; DAL-SF is the highest total (55.5).",
 ('Matthew Stafford',224.5):"43 attempts last week; ARI starts a rookie QB, so LAR should control. 5/9 over 224.5, model 254.",
 ('Matthew Stafford',299.5):"Ceiling leg: 4 of 9 games over 299.5. Model 22% vs a 13.5% price.",
 ('Braelon Allen',39.5):"NYJ lead back, 6/9 over 39.5 (median 59); model 46. Risk: NYJ trailing BUF.",
 ('Dalton Schultz',19.5):"HOU TE, 7/8 over 19.5; model 33.",
 ('Dalton Schultz',29.5):"7 of 8 over 29.5 at +102: his history says this is a better bet than the price.",
 ('Rico Dowdle',29.5):"PIT back, model 39; 3/5 over 29.5. Thin sample, 40% snaps.",
 ('Rico Dowdle',39.5):"Ceiling rung for the same back, 3/5 over 39.5. Thin sample.",
 ('Brenton Strange',5.5):"Receptions ceiling: 2 of 8 games with 6+ catches; model 20% vs 17% price.",
}
CARD={
 'SAFE':[('Josh Allen','pass_yards',149.5),('Brenton Strange','rec_yards',14.5),('Devaughn Vele','rec_yards',14.5),('Xavier Hutchinson','rec_yards',9.5),('AJ Barner','rec_yards',4.5),('Daniel Jones','pass_yards',99.5),('Aaron Rodgers','pass_yards',149.5),('Jahan Dotson','rec_yards',4.5),('Ryan Flournoy','rec_yards',19.5)],
 'MODERATE':[('Bryce Young','pass_yards',249.5),('Kayshon Boutte','rec_yards',14.5),('Cooper Kupp','rec_yards',14.5),('Keenan Allen','rec_yards',19.5)],
 'MEDIUM':[('Matthew Stafford','pass_yards',224.5),('Braelon Allen','rush_yards',39.5),('Dalton Schultz','rec_yards',19.5),('Rico Dowdle','rush_yards',29.5),('Daniel Jones','pass_yards',99.5),('Jahan Dotson','rec_yards',4.5)],
 'HIGH':[('Bryce Young','pass_yards',274.5),('Kayshon Boutte','rec_yards',24.5),('Cooper Kupp','rec_yards',14.5),('Keenan Allen','rec_yards',19.5),('Braelon Allen','rush_yards',39.5),('Aaron Rodgers','pass_yards',149.5)],
 'LOTTERY':[('Matthew Stafford','pass_yards',299.5),('Brenton Strange','receptions',5.5),('Rico Dowdle','rush_yards',39.5),('Dalton Schultz','rec_yards',29.5),('Devaughn Vele','rec_yards',14.5)],
}
rows=[]
for t,legs in CARD.items():
    for p,m,thr in legs:
        r=L[(L.player==p)&(L.market==m)&(L.alt_threshold==thr)]
        if r.empty: raise SystemExit(f"MISSING {t} {p} {m} {thr}")
        r=r.sort_values('offered_american',ascending=False).iloc[0]
        rows.append(dict(ticket=t,tier=t,player=p,team=r.team,opponent=r.opponent,event_id=r.event_id,market=m,side='OVER',threshold=thr,
            price_american=int(r.offered_american),book='fanduel',model_prob=round(float(r.model_prob),4),breakeven=round(float(r.breakeven),4),
            own=f"{int(r.own_hits)}/{int(r.own_games)}",state_label='HEALTHY_FULL',price_basis='naive_product',football_reason=REASON[(p,thr)]))
C=pd.DataFrame(rows)
for t,g in C.groupby('ticket',sort=False):
    assert g.event_id.nunique()==len(g), f"{t}: two legs from one game"
    d=float(np.prod([dec(x) for x in g.price_american])); p=float(np.prod(g.model_prob))
    print(f"\n{t:9s} {amer(d):+7d}  ({d:.1f}x)  {len(g)} legs | model ticket {p*100:.2f}% vs breakeven {100/d:.2f}%")
    for x in g.itertuples(): print(f"   {x.player:18s} {x.team:3s} {x.market:10s} o{x.threshold:<6g} {x.price_american:+6d}  model {x.model_prob*100:4.0f}% vs b/e {x.breakeven*100:4.0f}%  own {x.own}")
print("\nappearances:",C.groupby('player').ticket.nunique().sort_values(ascending=False).head(6).to_dict())
res=validate(C)
print("\nVALIDATOR:"); [print(f"  {s:8s} [{t}] {m}") for s,t,m in res if s=='ERROR']
print(f"  {sum(1 for s,*_ in res if s=='ERROR')} error(s), {sum(1 for s,*_ in res if s=='WARNING')} warning(s) (naive-product price basis on every ticket, as expected)")
C.to_csv(S+'sun_card.csv',index=False)
