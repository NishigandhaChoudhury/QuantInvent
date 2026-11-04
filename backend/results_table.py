import pandas as pd
from pathlib import Path

B = Path(__file__).resolve().parent / "data" / "processed"
R = pd.read_csv(B / "results.csv")
inst = R.groupby(["cond", "inst"])[["cost", "sl"]].mean().reset_index()
t = inst.groupby("cond").agg(cost_mean=("cost", "mean"), cost_sd=("cost", "std"),
                             sl_mean=("sl", "mean"), sl_sd=("sl", "std")).round(1)
b4 = inst[inst.cond == "B4 proposed"].set_index("inst")
for c in ["B1 classical", "B2 VQE random", "B3 ML+classical"]:
    o = inst[inst.cond == c].set_index("inst")
    t.loc[c, "B4 cheaper in (of 4)"] = int((b4.cost < o.cost).sum())
t.to_csv(B / "paper_table.csv")
print(t.to_string())