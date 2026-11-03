import { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, LineChart, Line, Legend, ErrorBar } from "recharts";

const API = "http://127.0.0.1:8000/api";
const PRI = "#4f46e5", OK = "#059669";
const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });

function useGet(url) {
  const [s, set] = useState({ data: null, err: null, loading: true });
  useEffect(() => {
    set({ data: null, err: null, loading: true });
    fetch(url).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => set({ data, err: null, loading: false }))
      .catch(e => set({ data: null, err: String(e.message || e), loading: false }));
  }, [url]);
  return s;
}
const Err = ({ e }) => <div className="banner">Cannot reach the API ({e}). Start it: <code>cd backend; uvicorn api:app --reload</code></div>;
const Kpi = ({ l, v, d, cls }) => <div className="card kpi"><div className="l">{l}</div><div className={"v " + (cls || "")}>{v}</div><div className="d">{d}</div></div>;

function Overview() {
  const { data, err, loading } = useGet(`${API}/results`);
  if (err) return <Err e={err} />;
  if (loading) return <div className="spin">Loading…</div>;
  const by = Object.fromEntries(data.map(r => [r.cond, r]));
  const b1 = by["B1 classical"], b4 = by["B4 proposed"], b3 = by["B3 ML+classical"], ex = by["Exact (ref)"];
  const red = ((b1.cost_mean - b4.cost_mean) / b1.cost_mean) * 100;
  const gap = ((b4.cost_mean - ex.cost_mean) / ex.cost_mean) * 100;
  const col = (r) => r.cond.startsWith("B4") ? PRI : r.cond.startsWith("Exact") ? "#94a3b8" : "#c7d2fe";
  return (<>
    <div className="grid g4">
      <Kpi l="Cost vs classical" v={`-${red.toFixed(0)}%`} d={`${fmt(b1.cost_mean)} → ${fmt(b4.cost_mean)} per plan`} cls="good" />
      <Kpi l="Service level (proposed)" v={`${b4.sl_mean.toFixed(0)}%`} d={`classical: ${b1.sl_mean.toFixed(0)}%`} cls="good" />
      <Kpi l="Gap to exact optimum" v={`${gap.toFixed(1)}%`} d="after classical repair + search" />
      <Kpi l="ML + classical (no quantum)" v={fmt(b3.cost_mean)} d="cost; strong baseline at this scale" />
    </div>
    <div className="grid g2">
      <div className="card"><h3>Average plan cost (lower is better)</h3>
        <ResponsiveContainer width="100%" height={280}><BarChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="cond" tick={{ fontSize: 11 }} /><YAxis /><Tooltip formatter={fmt} />
          <Bar dataKey="cost_mean" name="Cost" radius={[6, 6, 0, 0]}>{data.map((r, i) => <Cell key={i} fill={col(r)} />)}<ErrorBar dataKey="cost_std" stroke="#475569" /></Bar></BarChart></ResponsiveContainer></div>
      <div className="card"><h3>Service level % (target ≥ 95)</h3>
        <ResponsiveContainer width="100%" height={280}><BarChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="cond" tick={{ fontSize: 11 }} /><YAxis domain={[0, 100]} /><Tooltip formatter={(v) => v.toFixed(1)} />
          <Bar dataKey="sl_mean" name="Service level" radius={[6, 6, 0, 0]}>{data.map((r, i) => <Cell key={i} fill={r.sl_mean >= 95 ? OK : "#f59e0b"} />)}</Bar></BarChart></ResponsiveContainer></div>
    </div>
    <div className="card"><h3>Full results (mean ± SD, test instances 3–6)</h3>
      <table><thead><tr><th>Method</th><th>Cost</th><th>Service level %</th></tr></thead>
        <tbody>{data.map(r => <tr key={r.cond}><td>{r.cond.startsWith("B4") ? <b>{r.cond} <span className="pill">ours</span></b> : r.cond}</td>
          <td>{fmt(r.cost_mean)} ± {fmt(r.cost_std)}</td><td>{r.sl_mean.toFixed(1)} ± {r.sl_std.toFixed(1)}</td></tr>)}</tbody></table></div>
    <div className="note">Honest note: at 15 qubits the ML + classical baseline is competitive with the quantum pipeline. The quantum layer is a proof of concept for larger, harder problems.</div>
  </>);
}

function Forecast() {
  const [inst, setInst] = useState(3), [item, setItem] = useState("");
  const { data, err, loading } = useGet(`${API}/forecast?instance=${inst}`);
  const items = useMemo(() => data ? [...new Set(data.map(r => r.item_id))] : [], [data]);
  useEffect(() => { if (items.length && !items.includes(item)) setItem(items[0]); }, [items]);
  if (err) return <Err e={err} />;
  const cov = (k) => data ? (data.filter(r => r.actual <= r.mu + r[k]).length / data.length * 100).toFixed(1) : "-";
  const rows = (data || []).filter(r => r.item_id === item).map(r => ({ week: `W${r.h + 1}`, Actual: r.actual, Forecast: r.mu, "Gaussian 95%": +(r.mu + r.r_gauss).toFixed(1), "Conformal 95%": +(r.mu + r.r_conf).toFixed(1) }));
  return (<>
    <div className="grid g4">
      <Kpi l="Gaussian coverage" v={`${cov("r_gauss")}%`} d="share of actuals under the band" />
      <Kpi l="Conformal coverage" v={`${cov("r_conf")}%`} d="target 95%; smaller safety stock" cls="good" />
    </div>
    <div className="card"><div className="row" style={{ marginBottom: 12 }}>
      <label>Instance<select value={inst} onChange={e => setInst(+e.target.value)}>{[0, 1, 2, 3, 4, 5, 6].map(i => <option key={i}>{i}</option>)}</select></label>
      <label>SKU<select value={item} onChange={e => setItem(e.target.value)}>{items.map(i => <option key={i}>{i}</option>)}</select></label></div>
      {loading ? <div className="spin">Loading…</div> :
        <ResponsiveContainer width="100%" height={340}><LineChart data={rows}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="week" /><YAxis /><Tooltip /><Legend />
          <Line dataKey="Actual" stroke="#0f172a" strokeWidth={2.5} dot /><Line dataKey="Forecast" stroke={PRI} strokeWidth={2} dot />
          <Line dataKey="Gaussian 95%" stroke="#f59e0b" strokeDasharray="5 5" dot={false} /><Line dataKey="Conformal 95%" stroke={OK} strokeDasharray="5 5" dot={false} /></LineChart></ResponsiveContainer>}
    </div></>);
}

function Planner() {
  const [p, setP] = useState({ instance: 3, solver: "exact", demand_shock: 1.0, band: "conformal" });
  const [res, setRes] = useState(null), [base, setBase] = useState(null), [sweep, setSweep] = useState(null);
  const [busy, setBusy] = useState(false), [err, setErr] = useState(null);
  const set = (k, v) => setP({ ...p, [k]: v });
  const call = async (q) => { const r = await fetch(`${API}/optimise?` + new URLSearchParams(q)); if (!r.ok) throw new Error(r.status); return r.json(); };
  const run = async () => {
    setBusy(true); setErr(null); setSweep(null);
    try { const [a, b] = await Promise.all([call(p), call({ ...p, solver: "heuristic" })]); setRes(a); setBase(b); }
    catch (e) { setErr(String(e.message)); } setBusy(false);
  };
  const runSweep = async () => {
    setBusy(true); setErr(null);
    try {
      const out = [];
      for (const s of [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]) { const r = await call({ ...p, solver: "exact", demand_shock: s }); out.push({ shock: `${s}x`, cost: r.planned_cost }); }
      setSweep(out);
    } catch (e) { setErr(String(e.message)); } setBusy(false);
  };
  const save = res && base ? ((base.planned_cost - res.planned_cost) / base.planned_cost * 100) : 0;
  return (<>
    {err && <Err e={err} />}
    <div className="card"><h3>Scenario</h3><div className="row">
      <label>Planning instance<select value={p.instance} onChange={e => set("instance", e.target.value)}>{[3, 4, 5, 6].map(i => <option key={i}>{i}</option>)}</select></label>
      <label>Solver<select value={p.solver} onChange={e => set("solver", e.target.value)}>{["exact", "vqe", "heuristic", "auto"].map(i => <option key={i}>{i}</option>)}</select></label>
      <label>Safety stock method<select value={p.band} onChange={e => set("band", e.target.value)}>{["conformal", "gaussian"].map(i => <option key={i}>{i}</option>)}</select></label>
      <label>Demand disruption: {p.demand_shock}x<input type="range" min="0.7" max="1.6" step="0.05" value={p.demand_shock} onChange={e => set("demand_shock", e.target.value)} /></label>
      <button className="btn" onClick={run} disabled={busy}>{busy ? "Working…" : "Optimise"}</button>
      <button className="btn sec" onClick={runSweep} disabled={busy}>Sensitivity sweep</button></div></div>
    {res && <>
      <div className="grid g4">
        <Kpi l="Planned cost" v={fmt(res.planned_cost)} d={`solver: ${res.solver}`} />
        <Kpi l="Savings vs rule-based" v={`${save.toFixed(0)}%`} d={`rule-based: ${fmt(base.planned_cost)}`} cls={save > 0 ? "good" : "badc"} />
        <Kpi l="Orders placed" v={res.schedule.reduce((a, s) => a + s.weeks.reduce((x, y) => x + y, 0), 0)} d="across all SKUs and weeks" />
        <Kpi l="Run" v={`${res.seconds}s`} d={`${res.qubits} qubits`} />
      </div>
      <div className="card"><h3>Replenishment schedule</h3>
        <table><thead><tr><th>SKU</th><th>Order qty</th>{[1, 2, 3, 4, 5].map(w => <th key={w}>Week {w}</th>)}</tr></thead>
          <tbody>{res.schedule.map(s => <tr key={s.item}><td>{s.item}</td><td>{s.order_qty}</td>
            {s.weeks.map((x, i) => <td key={i}>{x ? <span className="ord">ORDER</span> : <span className="no">–</span>}</td>)}</tr>)}</tbody></table></div></>}
    {sweep && <div className="card"><h3>Cost sensitivity to demand disruption (exact solver)</h3>
      <ResponsiveContainer width="100%" height={260}><LineChart data={sweep}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="shock" /><YAxis /><Tooltip formatter={fmt} /><Line dataKey="cost" stroke={PRI} strokeWidth={2.5} /></LineChart></ResponsiveContainer></div>}
  </>);
}

const PAGES = { Overview: [Overview, "Benchmark of all methods"], Forecast: [Forecast, "Demand forecast and calibrated safety stock"], Planner: [Planner, "Optimise orders and test disruptions"] };
export default function App() {
  const [tab, setTab] = useState("Overview"); const [Page, desc] = PAGES[tab];
  return (<div className="app">
    <aside className="side"><div className="logo">QuantInvent</div><div className="tag">ML · Quantum · Operations Research</div>
      {Object.keys(PAGES).map(t => <button key={t} className={"nav " + (tab === t ? "on" : "")} onClick={() => setTab(t)}>{t}</button>)}</aside>
    <main className="main"><h1>{tab}</h1><p className="sub">{desc}. Data: M5 Walmart, store CA_1, FOODS_3.</p><Page /></main>
  </div>);
}