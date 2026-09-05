"""Score the Adam-eps dial sweep against notes/prereg_eps_dial.md (P1-P5).

    python nudial/score_eps_dial.py --root runs_eps [--base runs_sweep/base]
                                    [--out notes/eps_dial_scored.json]

Per run: eps, grok epoch, final test acc, committee (committee_from_coeffs on
final spectra), K, final train CE, lock epoch, per-tensor crossing epochs
from the tensor tap (first snapshot with median |g| < eps; and with median
sqrt(v) < eps), gradient plateau level. Per cell: grok rate, K histogram,
mean K, median grok epoch, geometric-mean CE floor, median crossings.
Cross-cell: P2 log-log slope of floor vs eps; P3 low-vs-high pooled
permutation test on K, Spearman(K, log eps), paired-by-seed deltas vs base.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
from train_semifinal_torch import committee_from_coeffs, grok_epoch

SIG = {"embed.W_E": "E", "blocks.0.attn.W_V": "V", "blocks.0.attn.W_O": "O",
       "blocks.0.mlp.W_in": "in", "blocks.0.mlp.W_out": "out",
       "unembed.W_U": "U", "blocks.0.attn.W_Q": "Q", "blocks.0.attn.W_K": "K"}
# P1 table (median of the two eps=1e-12 reference seeds), per tensor group
P1 = {1e-4: (0, 150), 1e-5: (225, 337), 1e-6: (450, 575), 1e-7: (687, 812),
      1e-8: (925, 1037), 1e-9: (1150, 1287), 1e-10: (1387, 1512),
      1e-12: (1862, 2012)}
LOW, HIGH = (1e-12, 1e-10, 1e-9), (1e-7, 1e-6, 1e-5, 1e-4)


def lock_epoch(z):
    comms = [tuple(committee_from_coeffs(z["coeffs"][i]))
             for i in range(len(z["epochs"]))]
    for i in range(len(comms) - 1, -1, -1):
        if comms[i] != comms[-1]:
            return int(z["epochs"][i + 1])
    return int(z["epochs"][0])


def read_run(rd):
    rd = Path(rd)
    z = np.load(rd / "spectra.npz")
    cfg = json.load(open(rd / "config.json"))
    m = json.load(open(rd / "metrics.json"))
    eps = cfg["adam_eps"]
    tr = np.array(m["train_losses"])
    r = dict(run=rd.name, cell=rd.parent.name, seed=cfg["data_seed"], eps=eps,
             grok=grok_epoch(z), acc=float(z["test_acc"][-1]),
             committee=committee_from_coeffs(z["coeffs"][-1]),
             lock=lock_epoch(z), final_ce=float(tr[-1]),
             floor_ce=float(np.exp(np.log(tr[-500:]).mean())),
             max_ce_after_10k=float(tr[10000:].max()) if len(tr) > 10000 else None)
    r["K"] = len(r["committee"])
    # gap-detector-free readouts: threshold count and participation ratio
    a = np.abs(z["coeffs"][-1]); e2 = a ** 2
    r["K_thr"] = int((a >= 0.1 * a.max()).sum())
    r["k_eff"] = float(e2.sum() ** 2 / (e2 ** 2).sum())
    if "ttap" in z.files:
        keys, st = list(z["ttap_keys"]), list(z["ttap_stats"])
        T, ep = z["ttap"], z["epochs"]
        cr, crv, plateau = {}, {}, {}
        for k, s in SIG.items():
            j = keys.index(k)
            g = T[:, j, st.index("g_q50")]
            v = T[:, j, st.index("sqrtv_q50")]
            hit = np.flatnonzero(g < eps)
            hitv = np.flatnonzero(v < eps)
            cr[s] = int(ep[hit[0]]) if len(hit) else None
            crv[s] = int(ep[hitv[0]]) if len(hitv) else None
            plateau[s] = float(np.median(g[-50:]))
        r["cross_g"], r["cross_v"], r["g_plateau"] = cr, crv, plateau
    return r


def perm_test(a, b, n=20000, seed=0):
    """one-sided: mean(a) > mean(b)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if pool[:len(a)].mean() - pool[len(a):].mean() >= obs:
            cnt += 1
    return float(obs), (cnt + 1) / (n + 1)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_eps")
    ap.add_argument("--base", default="runs_sweep/base")
    ap.add_argument("--out", default="notes/eps_dial_scored.json")
    args = ap.parse_args()
    runs = [read_run(Path(f).parent) for f in
            sorted(glob.glob(f"{args.root}/*/*/spectra.npz"))]
    base = ([read_run(Path(f).parent) for f in
             sorted(glob.glob(f"{args.base}/*/spectra.npz"))]
            if Path(args.base).exists() else [])
    cells = {}
    for r in runs:
        cells.setdefault(r["eps"], []).append(r)
    print(f"{len(runs)} runs in {len(cells)} cells; base n={len(base)}")
    summ = {}
    print("\n%-8s %3s %5s  %-28s %6s %8s %9s  %s" % (
        "eps", "n", "grok", "K hist (grokked)", "meanK", "med grok", "floorCE",
        "median crossing in/out | E/V/O/U (g crit) [P1 table]"))
    for eps in sorted(cells, reverse=True):
        rs = cells[eps]
        g = [r for r in rs if r["acc"] >= 0.99]
        ks = [r["K"] for r in g]
        hist = {k: ks.count(k) for k in sorted(set(ks))}
        s = dict(n=len(rs), n_grok=len(g), K_hist=hist,
                 mean_K=float(np.mean(ks)) if ks else None,
                 mean_K_thr=float(np.mean([r["K_thr"] for r in g])) if g else None,
                 mean_k_eff=float(np.mean([r["k_eff"] for r in g])) if g else None,
                 med_grok=float(np.median([r["grok"] for r in g])) if g else None,
                 floor_ce=float(np.exp(np.mean([np.log(r["floor_ce"]) for r in g]))) if g else None,
                 max_ce_after_10k=float(np.median([r["max_ce_after_10k"] for r in g
                                                    if r["max_ce_after_10k"] is not None])) if g else None)
        cr = ""
        if rs and "cross_g" in rs[0]:
            med = lambda key, t: (float(np.median([r[key][t] for r in rs
                                                   if r[key][t] is not None]))
                                  if any(r[key][t] is not None for r in rs) else None)
            s["cross_g"] = {t: med("cross_g", t) for t in SIG.values()}
            s["cross_v"] = {t: med("cross_v", t) for t in SIG.values()}
            io = np.nanmedian([s["cross_g"][t] or np.nan for t in ("in", "out")])
            ev = np.nanmedian([s["cross_g"][t] or np.nan for t in ("E", "V", "O", "U")])
            cr = f"{io:6.0f} | {ev:6.0f}  [{P1.get(eps, ('?', '?'))}]"
        summ[eps] = s
        print("%-8.0e %3d %5d  %-28s %6s %8s %9s  %s" % (
            eps, s["n"], s["n_grok"], hist,
            f"{s['mean_K']:.2f}/{s['mean_K_thr']:.2f}/{s['mean_k_eff']:.2f}" if ks else "-",
            f"{s['med_grok']:.0f}" if g else "-",
            f"{s['floor_ce']:.2e}" if g else "-", cr))
    if base:
        bg = [r for r in base if r["acc"] >= 0.99]
        bk = [r["K"] for r in bg]
        print("base 1e-8 (runs_sweep): n_grok %d hist %s meanK %.2f floor %.2e" % (
            len(bg), {k: bk.count(k) for k in sorted(set(bk))}, np.mean(bk),
            np.exp(np.mean([np.log(r["floor_ce"]) for r in bg]))))
    # P2
    pts = [(np.log10(e), np.log10(s["floor_ce"])) for e, s in summ.items()
           if s["floor_ce"] and s["n_grok"] >= 12]
    out = dict(cells={f"{e:.0e}": s for e, s in summ.items()}, runs=runs)
    if len(pts) >= 2:
        x, y = np.array(pts).T
        slope = float(np.polyfit(x, y, 1)[0])
        out["P2_slope"] = slope
        print(f"\nP2 floor ~ eps^slope: slope = {slope:.3f} (call 1.00 +- 0.15) over {len(pts)} cells")
    # P3
    lo = [r["K"] for e in LOW for r in cells.get(e, []) if r["acc"] >= 0.99
          and summ[e]["n_grok"] >= 12]
    hi = [r["K"] for e in HIGH for r in cells.get(e, []) if r["acc"] >= 0.99
          and summ[e]["n_grok"] >= 12]
    mid = [r["K"] for r in cells.get(1e-8, []) if r["acc"] >= 0.99]
    if lo and hi:
        d, p = perm_test(lo, hi)
        out["P3"] = dict(mean_low=float(np.mean(lo)), mean_mid=float(np.mean(mid)) if mid else None,
                         mean_high=float(np.mean(hi)), n_low=len(lo), n_high=len(hi),
                         delta=d, p_one_sided=p)
        print(f"\nP3 low {LOW} meanK {np.mean(lo):.2f} (n={len(lo)})  vs  high {HIGH} "
              f"meanK {np.mean(hi):.2f} (n={len(hi)}): delta {d:+.2f}, one-sided perm p = {p:.4f}"
              + (f";  1e-8 meanK {np.mean(mid):.2f}" if mid else ""))
    allk = [(np.log10(r["eps"]), r["K"]) for e, rs in cells.items() for r in rs
            if r["acc"] >= 0.99 and summ[e]["n_grok"] >= 12]
    if len(allk) > 10:
        x, y = np.array(allk).T
        rho = spearman(x, y)
        out["P3_spearman_K_logeps"] = rho
        print(f"Spearman(K, log eps) over {len(allk)} grokked runs = {rho:+.3f} (call: negative)")
    # paired vs base by seed
    if base:
        bmap = {r["seed"]: r for r in base if r["acc"] >= 0.99}
        print("\npaired-by-seed mean(K_cell - K_base):")
        for eps in sorted(cells, reverse=True):
            dd = [r["K"] - bmap[r["seed"]]["K"] for r in cells[eps]
                  if r["acc"] >= 0.99 and r["seed"] in bmap]
            if dd:
                summ[eps]["paired_dK_vs_base"] = float(np.mean(dd))
                print(f"  {eps:.0e}: {np.mean(dd):+.2f} (n={len(dd)}), "
                      f"same K in {sum(1 for v in dd if v == 0)}/{len(dd)}")
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
