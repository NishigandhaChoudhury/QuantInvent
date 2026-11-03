import numpy as np

def traj(d, x, dem):
    n, T = d["mu"].shape; I = d["I0"].copy(); out = []
    for t in range(T):
        I = np.maximum(0, I + d["q"] * x[:, t] - dem[:, t]); out.append(I.copy())
    return np.array(out).T

def repair(d, x):
    n, T = d["mu"].shape; x = np.array(x, dtype=int).reshape(n, T)
    for t in range(T):
        I = traj(d, x, d["mu"])
        for i in range(n):
            if I[i, t] < d["r"][i, t]: x[i, t] = 1          # service-level repair
        while (d["q"] * x[:, t]).sum() > d["W"]:             # capacity repair
            on = np.where(x[:, t] == 1)[0]
            x[on[np.argmin(d["h"][on])], t] = 0
    return x.ravel()

def to_idx(x):
    return int((np.asarray(x).astype(int) << np.arange(len(x))).sum())

def bits(i, N):
    return np.array([(i >> k) & 1 for k in range(N)])

def local_search(E, x, maxe=200):
    N = len(x); i = to_idx(x); cur = E[i]
    for _ in range(maxe):
        nb = [i ^ (1 << k) for k in range(N)]
        j = nb[int(np.argmin(E[nb]))]
        if E[j] < cur: i, cur = j, E[j]
        else: break
    return bits(i, N)