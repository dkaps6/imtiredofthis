# scripts/export_excel.py
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
outdir = root / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# Inputs we already produced earlier steps
paths = {
    "PropsPriced": outdir / "props_priced_clean.csv",
    "ModelPredictions": outdir / "master_model_predictions.csv",
}
# Optional tabs if present
optional = {
    "PropsRaw": outdir / "props_raw.csv",
    "Metrics": root / "data" / "metrics_ready.csv",
}

xlsx_path = outdir / "slate_export.xlsx"
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
    for name, p in paths.items():
        df = pd.read_csv(p)
        df.to_excel(xl, sheet_name=name, index=False)
    for name, p in optional.items():
        if p.exists():
            pd.read_csv(p).to_excel(xl, sheet_name=name, index=False)

print(f"[export] wrote {xlsx_path}")
