import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
N_SKU, T, LAM1, LAM2 = 3, 5, 0.05, 0.05

def load_instance(k=0, n=N_SKU, col="r_conf"):
    f = pd.read_csv(BASE / "forecasts_cal.csv")
    price = pd.read_csv(BASE / "weekly_top10.csv").groupby("item_id").sell_price.mean()
    f = f[f.instance == k]
    items = f.groupby("item_id").mu.sum().nlargest(n).index
    g = lambda c: np.stack([f[f.item_id == i].sort_values("h")[c].values for i in items])
    mu, r = g("mu"), g(col)
    pr = price[items].values
    q = np.ceil(2 * mu.mean(1))
    return dict(items=list(items), mu=mu, r=r, q=q, I0=mu[:, 0].copy(),
                h=0.1 * pr, o=10 * pr, pen=pr, W=np.ceil(0.67 * q.sum()))

def energies(d):
    n, Tt = d["mu"].shape; N = n * Tt
    idx = np.arange(2 ** N)
    bits = ((idx[:, None] >> np.arange(N)) & 1).reshape(-1, n, Tt)
    I = np.tile(d["I0"], (len(idx), 1)); E = np.zeros(len(idx))
    for t in range(Tt):
        x = bits[:, :, t]
        avail = I + d["q"] * x
        short = np.maximum(0, d["mu"][:, t] - avail)
        I = np.maximum(0, avail - d["mu"][:, t])
        E += (d["h"] * I + d["o"] * x + d["pen"] * short).sum(1)
        E += LAM1 * (np.maximum(0, d["r"][:, t] - I) ** 2).sum(1)
        E += LAM2 * np.maximum(0, (d["q"] * x).sum(1) - d["W"]) ** 2
    return E

def heuristic(d):
    n, Tt = d["mu"].shape; I = d["I0"].copy(); x = np.zeros((n, Tt))
    for t in range(Tt):
        o = I < d["mu"][:, t] + d["r"][:, t]
        x[:, t] = o
        I = np.maximum(0, I + d["q"] * o - d["mu"][:, t])
    return x.ravel()

def ansatz(N, p):
    th = ParameterVector("t", N * (p + 1)); qc = QuantumCircuit(N)
    for l in range(p + 1):
        for k in range(N): qc.ry(th[l * N + k], k)
        if l < p:
            for k in range(N - 1): qc.cz(k, k + 1)
    return qc

def run_vqe(E, qc, theta0, maxiter=200):
    f = lambda th: float(Statevector(qc.assign_parameters(list(th))).probabilities() @ E)
    res = minimize(f, theta0, method="COBYLA", options=dict(maxiter=maxiter))
    probs = Statevector(qc.assign_parameters(list(res.x))).probabilities()
    return res.fun, probs / probs.sum()

if __name__ == "__main__":
    d = load_instance(); E = energies(d); N = len(d["mu"].ravel()); p = 2
    Emin = E.min(); xh = heuristic(d)
    Eh = E[int((xh.astype(int) << np.arange(N)).sum())]
    print("SKUs:", d["items"], "| qubits:", N)
    print("exact optimum:", round(Emin, 2), "| heuristic warm-start:", round(Eh, 2))
    qc = ansatz(N, p)
    for name in ["random", "warm"]:
        gaps = []
        for s in range(5):
            rng = np.random.default_rng(s)
            if name == "random":
                th0 = rng.uniform(0, 2 * np.pi, qc.num_parameters)
            else:
                th0 = rng.normal(0, 0.05, qc.num_parameters)
                th0[:N] += np.pi * (0.1 + 0.8 * xh)
            fun, pr = run_vqe(E, qc, th0)
            best = E[rng.choice(len(E), 2048, p=pr)].min()
            gaps.append((best - Emin) / abs(Emin) * 100)
        print(name, "| best-of-2048 gap to optimum %:", np.round(np.mean(gaps), 2), "+/-", np.round(np.std(gaps), 2))
        