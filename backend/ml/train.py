import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path

torch.manual_seed(0); np.random.seed(0)
BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
L, H = 12, 5
TRAIN_END, VAL_END = 208, 241

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.lstm = nn.LSTM(8, 64, 2, batch_first=True, dropout=0.2)
        s.att = nn.Linear(64, 1)
        s.out = nn.Linear(64, 2 * H)
    def forward(s, x):
        h, _ = s.lstm(x)
        c = (torch.softmax(s.att(h), 1) * h).sum(1)
        o = s.out(c)
        return x[:, -1, 5:6] + o[:, :H], o[:, H:].clamp(-6, 1)

def nll(mu, ls, y):
    return (ls + 0.5 * ((y - mu) / ls.exp()) ** 2).mean()

def prep(d):
    u = d.units.values.astype(float); p = d.sell_price.values; e = d.event.values
    f = np.zeros((len(u), 8))
    for w in range(51, len(u)):
        f[w] = [u[w], u[w-1], u[w-2], u[w-3], u[w-51], u[w-3:w+1].mean(), p[w], e[w]]
    tr = u[51:TRAIN_END + 1]; lo, rg = tr.min(), np.ptp(tr) + 1e-8
    pt = p[51:TRAIN_END + 1]; pm, pr = pt.min(), np.ptp(pt) + 1e-8
    fn = f.copy(); fn[:, :6] = (f[:, :6] - lo) / rg; fn[:, 6] = (p - pm) / pr
    return fn, (u - lo) / rg, u, lo, rg

def mk(fn, un, ts):
    return (np.stack([fn[t-L:t] for t in ts]), np.stack([un[t:t+H] for t in ts]))

def run():
    df = pd.read_csv(BASE / "weekly_top10.csv")
    D = {i: prep(d.reset_index(drop=True)) for i, d in df.groupby("item_id")}
    tr, va = [], []
    for fn, un, *_ in D.values():
        tr.append(mk(fn, un, range(51 + L, TRAIN_END - H + 2)))
        va.append(mk(fn, un, range(TRAIN_END - H + 2, VAL_END - H + 2)))
    T = lambda z, j: torch.tensor(np.concatenate([a[j] for a in z]), dtype=torch.float32)
    Xtr, Ytr, Xva, Yva = T(tr, 0), T(tr, 1), T(va, 0), T(va, 1)

    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bad, state = 1e9, 0, None
    for ep in range(200):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i+64]; opt.zero_grad(); nll(*m(Xtr[b]), Ytr[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = nll(*m(Xva), Yva).item()
        if v < best: best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 15: break
    m.load_state_dict(state); m.eval()
    print("val nll", round(best, 3), "epochs", ep + 1)

    rows = []
    for item, (fn, un, u, lo, rg) in D.items():
        for k in range(7):
            t = VAL_END + 1 + 5 * k
            x = torch.tensor(fn[t-L:t][None], dtype=torch.float32)
            with torch.no_grad(): mu, ls = m(x)
            mu = mu[0].numpy() * rg + lo; sg = ls[0].exp().numpy() * rg
            naive = u[t-4:t].mean()
            for h in range(H):
                rows.append(dict(item_id=item, instance=k, week=t + h, h=h,
                                 mu=max(mu[h], 0), sigma=sg[h], actual=u[t + h], naive=naive))
    vrows = []
    for item, (fn, un, u, lo, rg) in D.items():
        for t in range(TRAIN_END + 1, VAL_END - H + 2, H):
            x = torch.tensor(fn[t-L:t][None], dtype=torch.float32)
            with torch.no_grad(): mu, ls = m(x)
            mu = mu[0].numpy() * rg + lo; sg = ls[0].exp().numpy() * rg
            for h in range(H):
                vrows.append(dict(item_id=item, h=h, mu=max(mu[h], 0), sigma=sg[h], actual=u[t + h]))
    pd.DataFrame(vrows).to_csv(BASE / "forecasts_val.csv", index=False)
    out = pd.DataFrame(rows); out.to_csv(BASE / "forecasts.csv", index=False)
    print("RMSE model:", np.sqrt(((out.mu - out.actual) ** 2).mean()).round(2),
          "| naive:", np.sqrt(((out.naive - out.actual) ** 2).mean()).round(2))

if __name__ == "__main__":
    run()