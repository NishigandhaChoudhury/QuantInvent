import json, numpy as np, pandas as pd
import quantum.vqe as V
from quantum.vqe import load_instance, energies, heuristic, ansatz, run_vqe, BASE
from classical.repair import repair, local_search, to_idx, bits

lam = json.load(open(BASE / "lambda.json"))
V.LAM1, V.LAM2 = lam["l1"], lam["l2"]

def score(d, x, act):
    n, T = d["mu"].shape; x = np.asarray(x).reshape(n, T)
    I = d["I0"].copy(); c = ok = 0
    for t in range(T):
        avail = I + d["q"] * x[:, t]; short = np.maximum(0, act[:, t] - avail)
        I = np.maximum(0, avail - act[:, t])
        c += (d["h"] * I + d["o"] * x[:, t] + d["pen"] * short).sum(); ok += (short == 0).sum()
    return c, ok / (n * T) * 100

fc = pd.read_csv(BASE / "forecasts_cal.csv"); rows = []; SEEDS = 3
for k in range(3, 7):
    d = load_instance(k); n, T = d["mu"].shape; N = n * T
    f = fc[fc.instance == k]
    g = lambda c: np.stack([f[f.item_id == i].sort_values("h")[c].values for i in d["items"]])
    act, nv = g("actual"), g("naive")
    E = energies(d); xh = heuristic(d)
    add = lambda name, seed, x: rows.append(dict(inst=k, cond=name, seed=seed, **dict(zip(["cost", "sl"], score(d, x, act)))))
    d1 = dict(d, mu=nv, r=np.zeros_like(nv))
    add("B1 classical", 0, repair(d1, heuristic(d1)))
    add("B3 ML+classical", 0, repair(d, xh))
    add("Exact (ref)", 0, bits(int(E.argmin()), N))
    qc = ansatz(N, 2)
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        th_r = rng.uniform(0, 2 * np.pi, qc.num_parameters)
        th_w = rng.normal(0, 0.05, qc.num_parameters); th_w[:N] += np.pi * (0.1 + 0.8 * xh)
        _, pr = run_vqe(E, qc, th_r)
        s0 = rng.choice(len(E), 2048, p=pr)
        add("B2 VQE random", s, bits(int(s0[np.argmin(E[s0])]), N))
        _, pw = run_vqe(E, qc, th_w)
        smp = rng.choice(len(E), 2048, p=pw); top = np.unique(smp[np.argsort(E[smp])[:10]])
        cands = [local_search(E, repair(d, bits(int(i), N))) for i in top]
        cands.append(local_search(E, repair(d, xh)))
        add("B4 proposed", s, min(cands, key=lambda c: E[to_idx(c)]))
    print("instance", k, "done")

R = pd.DataFrame(rows); R.to_csv(BASE / "results.csv", index=False)
print(R.groupby("cond")[["cost", "sl"]].agg(["mean", "std"]).round(2))