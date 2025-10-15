# scripts/export_excel.py
from pathlib import Path
import pandas as pd

# ---- choose an available Excel engine ----
def choose_engine() -> str:
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return "openpyxl"
        except Exception:
            raise SystemExit(
                "[export] ERROR: neither xlsxwriter nor openpyxl is installed. "
                "Add one to requirements.txt (e.g., `xlsxwriter`)."
            )

root = Path(__file__).resolve().parents[1]
outdir = root / "outputs"
outdir.mkdir(parents=True, exist_ok=True)

# required/optional sources
required = {
    "PropsPriced": outdir / "props_priced_clean.csv",
    "ModelPredictions": outdir / "master_model_predictions.csv",
}
optional = {
    "PropsRaw": outdir / "props_raw.csv",
    "Metrics": root / "data" / "metrics_ready.csv",
}

xlsx_main = outdir / "slate_export.xlsx"
xlsx_alt  = outdir / "model_report.xlsx"  # extra copy for downstream tools

engine = choose_engine()

def _write_sheet(xl, name: str, path: Path):
    if not path.exists() or path.stat().st_size == 0:
        print(f"[export] skip {name}: {path.name} missing or empty")
        return
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[export] failed reading {path.name}: {e}")
        return
    try:
        df.to_excel(xl, sheet_name=name, index=False)
        print(f"[export] wrote sheet {name} ({len(df)} rows)")
    except Exception as e:
        print(f"[export] failed writing sheet {name}: {e}")

with pd.ExcelWriter(xlsx_main, engine=engine) as xl:
    for name, p in required.items():
        _write_sheet(xl, name, p)
    for name, p in optional.items():
        _write_sheet(xl, name, p)

# also save a duplicate as model_report.xlsx for compatibility
try:
    if xlsx_alt.exists():
        xlsx_alt.unlink()
    xlsx_alt.write_bytes(xlsx_main.read_bytes())
    print(f"[export] wrote {xlsx_main} and {xlsx_alt}")
except Exception as e:
    print(f"[export] wrote {xlsx_main} (could not duplicate to model_report.xlsx: {e})")
