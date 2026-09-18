"""rank_share.py -- does a trained heracles ReLU net share rank across frequencies?

For each mod-add run: X = [U E; V E] (E = product of pre-layers, identity if none), Omega = W W_npost ... W_1.
    ratio_X = ||X||_* / sum_k ||X Pi_k||_*      Pi_k = projector onto span{cos_k, sin_k} (k = 0: constants)
    ratio_O = ||Omega||_* / sum_k ||Pi_k Omega||_*
Both equal 1 exactly when no rank is shared across frequency slices (Lemma 9, THEORY_relu.md); a fully locked
K-frequency population sits at 1/sqrt(K). Also per-neuron frequency concentration (energy share of the top
frequency of u_i, weighted by neuron norm) and the number of frequencies per neuron above 5% of its energy.
Usage: uv run python heracles/rank_share.py --cells H1 H3 --out heracles/rank_share.json
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent


def projectors(p):
    th = 2 * np.pi * np.arange(p) / p
    nf = (p - 1) // 2
    P = [np.full((p, p), 1.0 / p)]
    for k in range(1, nf + 1):
        c, s = np.cos(k * th), np.sin(k * th)
        c, s = c / np.linalg.norm(c), s / np.linalg.norm(s)
        P.append(np.outer(c, c) + np.outer(s, s))
    return P


def nuc(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def analyse(run, Pcache):
    cfg = json.loads((run / "config.json").read_text())
    w = load_file(str(run / "final.safetensors"))
    w = {k: v.astype(np.float64) for k, v in w.items()}
    p, nu = cfg["p"], cfg["nu"]
    n_pre = sum(1 for k in w if k.startswith("E"))
    n_post = sum(1 for k in w if k.startswith("W") and k != "W")
    E = np.eye(p)
    for i in range(1, n_pre + 1):
        E = w[f"E{i}"] @ E
    X = np.vstack([w["U"] @ E, w["V"] @ E])                # (2m, p)
    Om = w["W"]
    for i in range(n_post, 0, -1):
        Om = Om @ w[f"W{i}"]                                # (p, m)
    if p not in Pcache:
        Pcache[p] = projectors(p)
    Ps = Pcache[p]
    sx = sum(nuc(X @ Pk) for Pk in Ps)
    so = sum(nuc(Pk @ Om) for Pk in Ps)
    rX, rO = nuc(X) / sx, nuc(Om) / so
    # per-neuron frequency concentration of u_i (rows of U E)
    Ue = w["U"] @ E
    F = np.fft.rfft(Ue, axis=1)                              # (m, nf+1)
    en = np.abs(F[:, 1:]) ** 2                               # drop DC
    tot = en.sum(1) + 1e-30
    top = en.max(1) / tot
    wts = tot / tot.sum()
    top_share = float((top * wts).sum())
    nfreq = float(((en > 0.05 * en.max(1, keepdims=True)).sum(1) * wts).sum())
    return dict(run=str(run.relative_to(ROOT / "runs")), p=p, nu=nu, wd=cfg["weight_decay"], frac=cfg.get("frac_train"),
                ratio_X=rX, ratio_Omega=rO, top_freq_share=top_share, freqs_per_neuron=nfreq,
                X_nuc=nuc(X), X_slices=sx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["H1", "H3"])
    ap.add_argument("--out", default=str(ROOT / "rank_share.json"))
    args = ap.parse_args()
    Pcache, rows = {}, []
    for cell in args.cells:
        for run in sorted((ROOT / "runs" / cell).glob("**/final.safetensors")):
            run = run.parent
            cfg = json.loads((run / "config.json").read_text())
            if cfg.get("task", "add") not in ("add", None) and not str(cfg.get("task", "")).startswith("add"):
                continue
            try:
                rows.append(analyse(run, Pcache)); rows[-1]["cell"] = cell
            except Exception as e:
                print("skip", run, e)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["p"], r["nu"], r["wd"])].append(r)
    print(f"{'cell':5} {'p':>4} {'nu':>2} {'wd':>7} {'n':>3} | ratio_X med (min)   ratio_Om med (min)   top-freq share   freqs/neuron")
    summary = []
    for key in sorted(groups):
        g = groups[key]
        rx = np.array([r["ratio_X"] for r in g]); ro = np.array([r["ratio_Omega"] for r in g])
        ts = np.array([r["top_freq_share"] for r in g]); nf = np.array([r["freqs_per_neuron"] for r in g])
        summary.append(dict(cell=key[0], p=key[1], nu=key[2], wd=key[3], n=len(g), ratio_X_med=float(np.median(rx)),
                            ratio_X_min=float(rx.min()), ratio_O_med=float(np.median(ro)), ratio_O_min=float(ro.min()),
                            top_share_med=float(np.median(ts)), freqs_per_neuron_med=float(np.median(nf))))
        print(f"{key[0]:5} {key[1]:>4} {key[2]:>2} {key[3]:>7} {len(g):>3} | {np.median(rx):.3f} ({rx.min():.3f})      "
              f"{np.median(ro):.3f} ({ro.min():.3f})        {np.median(ts):.3f}          {np.median(nf):.2f}")
    Path(args.out).write_text(json.dumps(dict(summary=summary, runs=rows), indent=1))
    print("wrote", args.out)
