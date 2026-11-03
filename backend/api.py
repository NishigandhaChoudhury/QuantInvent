import json, time, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import quantum.vqe as V
from quantum.vqe import load_instance, energies, heuristic, ansatz, run_vqe, BASE
from classical.repair import repair, local_search, to_idx, bits

app = FastAPI(title="QuantInvent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
lam = json.load(open(BASE / "lambda.json")); V.LAM1, V.LAM2 = lam["l1"], lam["l2"]
fc = pd.read_csv(BASE / "forecasts_cal.csv")

def planned_cost(d, x):
    n, T = d["mu"].shape; x = np.asarray(x).reshape(n, T); I = d["I0"].copy(); c = 0
    for t in range(T):
        a = I + d["q"] * x[:, t]; sh = np.maximum(0, d["mu"][:, t] - a)
        I = np.maximum(0, a - d["mu"][:, t])
        c += (d["h"] * I + d["o"] * x[:, t] + d["pen"] * sh).sum()
    return float(c)

@app.get("/api/results")
def results():
    R = pd.read_csv(BASE / "results.csv")
    g = R.groupby("cond")[["cost", "sl"]].agg(["mean", "std"]).round(2)
    g.columns = ["cost_mean", "cost_std", "sl_mean", "sl_std"]
    return g.reset_index().to_dict("records")

@app.get("/api/forecast")
def forecast(instance: int = 3):
    f = fc[fc.instance == instance]
    return f[["item_id", "h", "mu", "sigma", "actual", "r_gauss", "r_conf"]].round(2).to_dict("records")

@app.get("/api/optimise")
def optimise(instance: int = 3, solver: str = "auto", demand_shock: float = 1.0, band: str = "conformal"):
    d = load_instance(instance, col="r_conf" if band == "conformal" else "r_gauss")
    d["mu"] = d["mu"] * demand_shock; d["r"] = d["r"] * demand_shock
    n, T = d["mu"].shape; N = n * T; E = energies(d); xh = heuristic(d)
    used = solver if solver != "auto" else ("exact" if N <= 12 else "vqe")
    t0 = time.time()
    if used == "heuristic":
        x = repair(d, xh)
    elif used == "exact":
        x = bits(int(E.argmin()), N)
    else:
        qc = ansatz(N, 2); rng = np.random.default_rng(0)
        th = rng.normal(0, 0.05, qc.num_parameters); th[:N] += np.pi * (0.1 + 0.8 * xh)
        _, pr = run_vqe(E, qc, th)
        smp = rng.choice(len(E), 2048, p=pr); top = np.unique(smp[np.argsort(E[smp])[:10]])
        x = min([local_search(E, repair(d, bits(int(i), N))) for i in top], key=lambda c: E[to_idx(c)])
    sched = np.asarray(x).reshape(n, T)
    return dict(solver=used, seconds=round(time.time() - t0, 2), qubits=N,
                planned_cost=round(planned_cost(d, x), 2),
                schedule=[dict(item=i, order_qty=float(q), weeks=s.tolist())
                          for i, q, s in zip(d["items"], d["q"], sched)])