"""Score T2-C (product law, eps) and T2-E (frac-matched task table) per
flight/PREREG_T2.md. P-E2 (checkpoint margin-per-norm Deff) is scored by a
separate script committed before unblinding those checkpoints."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_flight"


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    o = np.argsort(e)[::-1]
    g = e[o[:-1]] - e[o[1:]]
    return o[:int(np.argmax(g[:min(10, len(g))])) + 1].tolist()


def t_ident(ep, E, comm):
    """Largest-gap instrument: first sustained (x2) snapshot whose energy
    ranking separates the final committee at AUC >= 0.9."""
    K = len(comm)
    isc = np.zeros(E.shape[1], bool)
    isc[comm] = True
    aucs = []
    for t in range(E.shape[0]):
        r = np.argsort(np.argsort(E[t]))
        pos, neg = r[isc], r[~isc]
        aucs.append((pos[:, None] > neg[None, :]).mean())
    ok = np.array(aucs) >= 0.9
    i = fs(ok)
    return int(ep[i]) if i >= 0 else None


def load_run(rd):
    z = np.load(rd / "spectra.npz")
    ep, E, ta, te = z["epochs"], z["energy"], z["train_acc"], z["test_acc"]
    cfg = json.loads((rd / "config.json").read_text())
    comm = committee_final(E)
    junk = [j for j in range(E.shape[1]) if j not in comm]
    r = E[:, junk].sum(1) / E.sum(1)
    im, ig = fs(ta >= 0.99), fs(te >= 0.9)
    out = dict(lr=cfg["lr"], wd=cfg["weight_decay"], eps=cfg["adam_eps"],
               fn=cfg["fn_name"], p=cfg["p"], frac=cfg["frac_train"],
               t_mem=int(ep[im]) if im >= 0 else None,
               t_gen=int(ep[ig]) if ig >= 0 else None,
               t_id=t_ident(ep, E, comm), lam=None, r=r, ep=ep, im=im, ig=ig)
    if im >= 0 and ig >= 0 and ig - im >= 5:
        x = ep[im:ig + 1].astype(float)
        out["lam"] = -float(np.polyfit(
            x, np.log(np.clip(r[im:ig + 1], 1e-12, None)), 1)[0])
    return out


# ---------------- T2-C ----------------
grid = [load_run(rd) for rd in sorted((RUNS / "t2c").glob("*/*"))
        if (rd / "spectra.npz").exists()]
gen = [g for g in grid if g["t_gen"] is not None and g["lam"] and g["lam"] > 0]
cen = [g for g in grid if g["t_gen"] is None]
print(f"T2-C grid: {len(grid)} runs, {len(gen)} generalized, {len(cen)} censored")
by = defaultdict(list)
for g in gen:
    by[(g["lr"], g["wd"])].append(g["lam"])
print(f"{'lr':>7} {'wd':>4} {'n':>2} {'med lam/kep':>11} {'lam/(2 lr wd)':>13}")
for k in sorted(by):
    v = np.median(by[k]) * 1e3
    print(f"{k[0]:>7} {k[1]:>4} {len(by[k]):>2} {v:>11.3f} "
          f"{v / (2 * k[0] * k[1] * 1e3):>13.3f}")
x = np.log([g["lr"] * g["wd"] for g in gen])
y = np.log([g["lam"] for g in gen])
slope = np.polyfit(x, y, 1)[0]
ok1 = 0.7 <= slope <= 1.3
pair = None
if (1e-3, 3.0) in by and (3e-3, 1.0) in by:
    pair = np.median(by[(1e-3, 3.0)]) / np.median(by[(3e-3, 1.0)])
    ok1 = ok1 and (1 / 1.3 <= pair <= 1.3)
print(f"P-C1: slope log-lam~log(lr*wd) = {slope:.2f} (bar [0.7,1.3]); "
      f"matched-product ratio = {pair and round(pair, 2)} (bar [0.77,1.3]) -> "
      f"{'PASS' if ok1 else 'FAIL'}")
tid = [g["t_id"] for g in grid if g["t_id"] is not None]
tg = [g["t_gen"] for g in gen]
ok2 = (max(tid) / max(min(tid), 1) < 2) and (max(tg) / min(tg) > 20)
print(f"P-C2: t_ident span x{max(tid)/max(min(tid),1):.2f} (<2 bar), "
      f"t_gen span x{max(tg)/min(tg):.0f} (>20 bar) -> "
      f"{'PASS' if ok2 else 'FAIL'}  [t_id range {min(tid)}-{max(tid)}]")
print("P-C3 (exploratory) eps arms:")
eps_runs = [load_run(rd) for rd in sorted((RUNS / "t2c_eps").glob("*/*"))
            if (rd / "spectra.npz").exists()]
base = [g for g in gen if g["lr"] == 1e-3 and g["wd"] == 1.0]
for e in sorted({g["eps"] for g in eps_runs} | {1e-8}):
    v = base if e == 1e-8 else [g for g in eps_runs if g["eps"] == e]
    lam = [g["lam"] for g in v if g["lam"]]
    tgs = [g["t_gen"] for g in v if g["t_gen"]]
    tis = [g["t_id"] for g in v if g["t_id"] is not None]
    print(f"  eps={e}: n={len(v)}  med lam/kep "
          f"{np.median(lam)*1e3 if lam else float('nan'):.3f}  med t_gen "
          f"{np.median(tgs) if tgs else float('nan'):.0f}  med t_ident "
          f"{np.median(tis) if tis else float('nan'):.0f}")

# ---------------- T2-E ----------------
R_STAR_TASK = {("add", 113): 10 ** -2.25, ("add", 96): 10 ** -1.94,
               ("sub", 113): 10 ** -1.73}
rows = [load_run(rd) for rd in sorted((RUNS / "t2e").glob("*/*"))
        if (rd / "spectra.npz").exists()]
print(f"\nT2-E: {len(rows)} runs")
med = {}
for (fn, p) in [("add", 113), ("add", 96), ("sub", 113)]:
    for frac in (0.3, 0.6):
        v = [g["lam"] / (2 * g["lr"] * g["wd"]) for g in rows
             if g["fn"] == fn and g["p"] == p and g["frac"] == frac
             and g["lam"] and g["lam"] > 0 and g["t_gen"]]
        med[(fn, p, frac)] = float(np.median(v)) if v else None
        cenn = sum(1 for g in rows if g["fn"] == fn and g["p"] == p
                   and g["frac"] == frac and g["t_gen"] is None)
        print(f"  {fn}{p} f{frac}: kappa*Deff median "
              f"{med[(fn, p, frac)] and round(med[(fn, p, frac)], 3)} "
              f"(n={len(v)}, censored {cenn})")
a, b, c = (med.get(("add", 113, 0.3)), med.get(("add", 96, 0.3)),
           med.get(("sub", 113, 0.3)))
if None not in (a, b, c):
    ok = a > b > c
    print(f"P-E1: add113 {a:.3f} > add96 {b:.3f} > sub113 {c:.3f} -> "
          f"{'PASS' if ok else 'FAIL'}")
else:
    print(f"P-E1: not scorable (a cell has no cleanup-limited runs): "
          f"{a} / {b} / {c}")
# P-E3: sub f0.3 law predictions at t_mem+500, own-task r*
print("P-E3 (sub113 f0.3, prediction at t_mem+500 with r*=10^-1.73):")
for g in rows:
    if not (g["fn"] == "sub" and g["frac"] == 0.3):
        continue
    if g["im"] < 0:
        continue
    ep, r, im = g["ep"], g["r"], g["im"]
    iw = np.where((ep >= g["t_mem"]) & (ep <= g["t_mem"] + 500))[0]
    lam = -float(np.polyfit(ep[iw].astype(float),
                            np.log(np.clip(r[iw], 1e-12, None)), 1)[0])
    rs = R_STAR_TASK[("sub", 113)]
    if r[im] <= np.e * rs:
        pred = g["t_mem"] + 132
    elif lam <= 0:
        pred = "CENSORED"
    else:
        w = np.log(r[im] / rs) / lam
        pred = "CENSORED" if g["t_mem"] + w > 50_000 else round(g["t_mem"] + w)
    print(f"  pred {pred} vs realized {g['t_gen'] or 'CENSORED'}")
