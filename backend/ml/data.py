import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data"
RAW, OUT = BASE / "raw", BASE / "processed"

def build(n_skus=10):
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    cal = pd.read_csv(RAW / "calendar.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")

    sales = sales[(sales.store_id == "CA_1") & (sales.dept_id == "FOODS_3")]
    dcols = [c for c in sales.columns if c.startswith("d_")]
    top = sales.set_index("item_id")[dcols].sum(axis=1).nlargest(n_skus).index
    sales = sales[sales.item_id.isin(top)]

    long = sales.melt(id_vars=["item_id"], value_vars=dcols,
                      var_name="d", value_name="units")
    cal = cal[cal.d.isin(dcols)].copy()
    cal["event"] = cal.event_name_1.notna().astype(int)
    long = long.merge(cal[["d", "wm_yr_wk", "event", "snap_CA"]], on="d")

    full = cal.groupby("wm_yr_wk").size()
    long = long[long.wm_yr_wk.isin(full[full == 7].index)]

    wk = (long.groupby(["item_id", "wm_yr_wk"])
          .agg(units=("units", "sum"), event=("event", "max"),
               snap=("snap_CA", "sum")).reset_index())

    p = prices[prices.store_id == "CA_1"][["item_id", "wm_yr_wk", "sell_price"]]
    wk = wk.merge(p, on=["item_id", "wm_yr_wk"], how="left")
    wk = wk.sort_values(["item_id", "wm_yr_wk"])
    wk["sell_price"] = wk.groupby("item_id").sell_price.transform(lambda s: s.ffill().bfill())
    wk["week_idx"] = wk.groupby("item_id").cumcount()

    OUT.mkdir(parents=True, exist_ok=True)
    wk.to_csv(OUT / "weekly_top10.csv", index=False)
    return wk

if __name__ == "__main__":
    w = build()
    print(w.shape, w.item_id.nunique(), w.week_idx.max() + 1)
    print(w.head())