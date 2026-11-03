import json, numpy as np, pandas as pd
from skopt import gp_minimize
from skopt.space import Real
import quantum.vqe as V
from quantum.vqe import load_instance, energies, BASE

fc = pd.read_csv(BASE / "forecasts_cal.csv")

def score(d, x, act):
    n, T = d["mu"].shape; x = np.asarray(x).reshape(n, T)
    I = d["I0"].copy(); c = ok = 0
    for t in range(T):
        a = I + d["q"] * x[:, t]; sh = np.maximum(0, act[:, t] - a)
        I = np.maximum(0, a - act[:, t])
        c += (d["h"] * I + d["o"] * x[:, t] + d["pen"] * sh).sum(); ok += (sh == 0).sum()
    return c, ok / (n * T) * 100

def obj(p):
    V.LAM1, V.LAM2 = 10 ** p[0], 10 ** p[1]; tot = 0
    for k in range(3):
        d = load_instance(k); N = d["mu"].size; E = energies(d)
        x = np.array([(int(E.argmin()) >> j) & 1 for j in range(N)])
        f = fc[fc.instance == k]
        act = np.stack([f[f.item_id == i].sort_values("h").actual.values for i in d["items"]])
        c, sl = score(d, x, act); tot += c + 100 * max(0, 95 - sl)
    return tot / 3

r = gp_minimize(obj, [Real(-3, 2), Real(-3, 2)], n_calls=30, acq_func="EI", random_state=0)
l1, l2 = 10 ** r.x[0], 10 ** r.x[1]
print("best lambda1, lambda2:", round(l1, 4), round(l2, 4), "| objective:", round(r.fun, 1))
json.dump(dict(l1=l1, l2=l2), open(BASE / "lambda.json", "w"))