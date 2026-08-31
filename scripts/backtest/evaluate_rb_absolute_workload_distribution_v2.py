import scripts.backtest.evaluate_rb_absolute_workload_distribution as m

_original_load = m.load_m94d_rb

def _load_filtered(root):
    hold, validation = _original_load(root)
    hold = hold.loc[m.num(hold["week"]).ge(13)].copy()
    return hold, validation

m.load_m94d_rb = _load_filtered

if __name__ == "__main__":
    raise SystemExit(m.main())
