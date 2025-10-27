import argparse, yaml
from pathlib import Path
from model.ingest.loaders import load_all
from model.features.build import build_features
from model.pricing.price import price_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    merged, team_frame, cov, lines = load_all(cfg)
    if merged.empty:
        print('No props to price. Check input paths in config.yaml'); return
    m = build_features(merged, team_frame, cov)
    edges = price_all(m)

    out_dir = Path(args.out_dir or cfg['outputs']['dir']); out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / cfg['outputs']['full'], index=False)
    slip = edges[edges['tier'].isin(['ELITE (≥6%)','GREEN (4–6%)'])].sort_values('edge_pct_pts', ascending=False)
    slip.to_csv(out_dir / cfg['outputs']['bet_slip'], index=False)
    print('Wrote outputs to', out_dir)

if __name__ == '__main__':
    main()
