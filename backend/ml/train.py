import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path

torch.manual_seed(0); np.random.seed(0)
BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
L, H = 12, 5                  # look-back weeks, forecast horizon
TRAIN_END, VAL_END = 208, 241 # week indices (4y train, ~34w val, 35w test)

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
        return o[:, :H], o[:, H:].clamp(-6, 3)   # mu, log sigma

def nll(mu, ls, y):
    return (ls + 0.5 * ((y - mu) / ls.exp()) ** 2).mean()

def features(d):
    u = d.units.values.astype(float); p = d.sell_price.values; e = d.event.values
    f = np.zeros((len(u), 8))
    for w in range(51, len(u)):
        f[w] = [u[w], u[w-1], u[w-2], u[w-3], u[w-51], u[w-3:w+1].mean(), p[w], e[w]]
    return f, u

def run():
    df = pd.read_csv(BASE / "weekly_top10.csv")
    rows = []
    for item, d in df.groupby("item_id"):
        f, u = features(d.reset_index(drop=True))
        tr = f[51:TRAIN_END + 1]
        mn, mx = tr.min(0), tr.max(0); rg = mx - mn + 1e-8
        fn = (f - mn) / rg
        lo, r0 = mn[0], rg[0]
        un = (u - lo) / r0
        mk = lambda ts: (torch.tensor(np.stack([fn[t-L:t] for t in ts]), dtype=torch.float32),
                         torch.tensor(np.stack([un[t:t+H] for t in ts]), dtype=torch.float32))
        Xtr, Ytr = mk(range(51 + L, TRAIN_END - H + 2))
        Xva, Yva = mk(range(TRAIN_END - H + 2, VAL_END - H + 2))
        m = Net()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        best, bad, state = 1e9, 0, None
        for ep in range(200):
            m.train(); opt.zero_grad()
            loss = nll(*m(Xtr), Ytr); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): v = nll(*m(Xva), Yva).item()
            if v < best: best, bad, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        m.load_state_dict(state); m.eval()
        for k in range(7):                       # 7 test instances x 5 weeks
            t = VAL_END + 1 + 5 * k
            x = torch.tensor(fn[t-L:t][None], dtype=torch.float32)
            with torch.no_grad(): mu, ls = m(x)
            mu = mu[0].numpy() * r0 + lo; sg = ls[0].exp().numpy() * r0
            naive = u[t-4:t].mean()
            for h in range(H):
                rows.append(dict(item_id=item, instance=k, week=t + h, h=h,
                                 mu=max(mu[h], 0), sigma=sg[h], actual=u[t + h], naive=naive))
        print(item, "val nll", round(best, 3), "epochs", ep + 1)
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "forecasts.csv", index=False)
    print("RMSE model:", np.sqrt(((out.mu - out.actual) ** 2).mean()).round(2),
          "| naive:", np.sqrt(((out.naive - out.actual) ** 2).mean()).round(2))

if __name__ == "__main__":
    run()