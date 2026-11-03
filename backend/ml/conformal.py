import numpy as np, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
ALPHA, Z = 0.05, 1.645

val = pd.read_csv(BASE / "forecasts_val.csv")
test = pd.read_csv(BASE / "forecasts.csv")

s = ((val.actual - val.mu) / val.sigma).values          # one-sided scores
n = len(s)
q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n))

test["r_gauss"] = Z * test.sigma
test["r_conf"] = np.maximum(q * test.sigma, 0)
test.to_csv(BASE / "forecasts_cal.csv", index=False)

print("calibration points:", n, "| conformal multiplier q:", round(q, 3), "(gaussian: 1.645)")
print("target coverage: 95%")
print("gaussian coverage:", round(((test.actual <= test.mu + test.r_gauss).mean()) * 100, 1), "%")
print("conformal coverage:", round(((test.actual <= test.mu + test.r_conf).mean()) * 100, 1), "%")