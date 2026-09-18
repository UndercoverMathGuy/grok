"""rep_table.py -- score heracles/FINAL_RUNS/REP against the frozen predictions in heracles/rep_pred.json (written before scoring).
Instruments are nsum.py's (N_.05, K_5pct, K_2pct, clean fraction, margin, acc).  Every seed of every cell is listed;
hit rates are over ALL seeds in the cell (collapsed / unfit seeds count as misses).
Usage: uv run python heracles/rep_table.py [--runs heracles/FINAL_RUNS/REP]"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(ROOT / "FINAL_RUNS" / "REP"))
a = ap.parse_args()
PRED = json.loads((ROOT / "rep_pred.json").read_text())

cells = defaultdict(list)
for f in sorted(Path(a.runs).glob("*/seed*/metrics.json")):
    cells[f.parent.parent.name].append(json.loads(f.read_text()))


def pred(p, q):
    d = PRED.get(str(p), {}).get(f"{q:g}")
    return d


def dist(xs):
    return " ".join(str(x) for x in sorted(xs))


def med(xs):
    return st.median(xs) if xs else float("nan")


rows = []
for name, rs in cells.items():
    rs.sort(key=lambda r: r["seed"])
    r0 = rs[0]; p, act, q = r0["p"], r0["act"], r0["q"]
    P = pred(p, q) if act != "relu" else dict(K_star="dense", plat74=[None, None], plat95=[None, None], nf=(p - 1) // 2)
    nf = (p - 1) // 2
    fit = [r for r in rs if r["acc"] >= 0.999]
    line = dict(cell=name, p=p, act=act, q=q, lam=r0["lam"], init=r0["init_scale"], resumed=("_from" in name), n=len(rs), n_fit=len(fit),
                acc_min=min(r["acc"] for r in rs), margin_med=med([r["margin"] for r in fit]), clean_med=med([r["clean_frac"] for r in fit]),
                K_star=P and P["K_star"], plat74=P and P["plat74"], plat95=P and P["plat95"], nf=nf)
    for k in ("N_05", "K_5pct", "K_2pct"):
        xs = [r[k] for r in rs]
        line[k] = dist(xs); line[k + "_med"] = med([r[k] for r in fit])
        if P and P["K_star"] == "dense":
            line[k + "_hit1"] = sum(1 for r in rs if r["acc"] >= 0.999 and r[k] >= 0.8 * nf)      # dense criterion: count >= 0.8 nf
            if P["plat74"][0] is not None:
                line[k + "_hit74"] = sum(1 for r in rs if r["acc"] >= 0.999 and P["plat74"][0] <= r[k] <= P["plat74"][1])
                line[k + "_hit95"] = sum(1 for r in rs if r["acc"] >= 0.999 and P["plat95"][0] <= r[k] <= P["plat95"][1])
        elif P:
            ks = P["K_star"]
            line[k + "_hit74"] = sum(1 for r in rs if r["acc"] >= 0.999 and P["plat74"][0] <= r[k] <= P["plat74"][1])
            line[k + "_hit95"] = sum(1 for r in rs if r["acc"] >= 0.999 and P["plat95"][0] <= r[k] <= P["plat95"][1])
            line[k + "_hit1"] = sum(1 for r in rs if r["acc"] >= 0.999 and abs(r[k] - ks) <= 1)
    rows.append(line)

rows.sort(key=lambda l: (l["act"], l["resumed"], l["p"], l["q"], l["lam"], l["init"]))
print("| cell | n (fit) | acc min | margin med | clean med | K* | plat.74 | plat.95 | N_.05 med [all seeds] | K_5pct med [all] | K_2pct med [all] | hits in plat.74 (N05/K5/K2) | in plat.95 | K*+-1 (dense: >= 0.8 nf) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for l in rows:
    def h(sfx):
        return "/".join(str(l.get(k + sfx, "-")) for k in ("N_05", "K_5pct", "K_2pct"))
    pl = lambda x: f"{x[0]}..{x[1]}" if x and x[0] is not None else "-"
    print(f"| {l['cell']} | {l['n']} ({l['n_fit']}) | {l['acc_min']:.3f} | {l['margin_med']:.2f} | {l['clean_med']:.2f} | {l['K_star']} of {l['nf']} | {pl(l['plat74'])} | {pl(l['plat95'])} | "
          f"{l['N_05_med']} [{l['N_05']}] | {l['K_5pct_med']} [{l['K_5pct']}] | {l['K_2pct_med']} [{l['K_2pct']}] | {h('_hit74')} | {h('_hit95')} | {h('_hit1')} |")
(ROOT / "rep_scored.json").write_text(json.dumps(rows, indent=1))
