"""ladder_loss.py -- CAUSAL frequency count by LOGIT-SPACE ablation (Nanda et al. 2301.05217 §4.4 /
Fig. 6 protocol), plus the Morwani et al. 2311.07568 Fig. 4/7 neuron-Fourier histogram.

Why this instrument and not the margin one.  The margin-ablation count (ladder_abl.py) measures the
relative loss of the MINIMUM margin when a frequency is removed.  When the trained min-margin is
thin -- which it is at nu = 4 (0.8-1.9 against logits of order 10^2) and negative at nu = 5 -- that
ratio is meaningless: cutting anything at all makes the margin negative.  Cross-entropy over the
whole table has no such problem, which is why Nanda et al. count with it.

(1) K_loss -- per-frequency loss ablation.
    The logit table z(row, c) is a function of the output class c.  Frequency k is removed by
    projecting the two directions cos(2 pi k c / p), sin(2 pi k c / p) out of every row's logit
    vector along the CLASS axis.  (For the exact families the clean logits depend only on
    D = sum_j a_j - c, so this removes exactly frequency k's contribution and nothing else.)
    Then the full-table cross-entropy is recomputed:
        dCE_k = CE(logits with frequency k projected out) - CE(logits).
    Counts reported at three stated thresholds:
        K_loss_10   #{k : dCE_k >= 0.10 * max_j dCE_j}      (relative)
        K_loss_25   #{k : dCE_k >= 0.25 * max_j dCE_j}      (relative)
        K_loss_abs  #{k : dCE_k >= 0.01 * ln p}             (absolute, 1% of the uniform-guess loss)
    plus the whole sorted dCE curve so the gap is visible, and Nanda's two summary numbers
        RESTRICTED loss  CE when only the counted frequencies are kept (all others projected out)
        EXCLUDED loss    CE when exactly the counted frequencies are projected out.

    SCALE.  A trained net's logits are large, so its CE can be ~1e-3 and every dCE_k is dominated by
    the scale rather than the structure.  Every quantity is therefore reported twice: at the
    TRAINED scale, and at a NORMALISED scale, where the logits are multiplied by a single scalar
    alpha chosen so that the unablated CE equals `--ce-target` (0.1 nats by default).  The
    normalised numbers are the ones to read; both are stored.

(2) Morwani neuron-Fourier histogram.
    For every neuron, take its token-indexed vectors (the n operand embeddings and the readout
    column), Fourier transform each along the token axis, sum the power over the vectors, drop the
    DC term, and form
        max normalized power  =  max_k power_k / sum_{k >= 1} power_k,   argmax frequency = the neuron's frequency.
    Report the histogram of neurons per frequency, unweighted and weighted by the neuron's weight
    norm ||theta_i||, and the number of frequencies holding >= 5% of the total weighted mass.
    Expectation to CHECK, not assume: near-uniform over frequencies at q = nu, concentrated on
    about K* frequencies at q = 2.

Usage
  python3 -u heracles/ladder_loss.py --runs heracles/runs/LAD/* --out .../train/loss_scored.json
  python3 -u heracles/ladder_loss.py --validate 59            # constructed nets: must return K
"""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEV = os.environ.get("NSUM_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
DT = torch.float64


# ------------------------------------------------------------------ task
def rows_and_weights(p, n, tied):
    """Rows of the task and their multiplicities.  With tied embeddings the net is symmetric in the
    operands, so the multiset table with multinomial weights gives exactly the full-table loss."""
    if not tied:
        r = torch.arange(p ** n)
        ops = torch.stack([(r // p ** j) % p for j in range(n)], 1)
        return ops, torch.ones(len(ops), dtype=DT)
    combs = np.fromiter(itertools.chain.from_iterable(itertools.combinations_with_replacement(range(p), n)),
                        dtype=np.int64).reshape(-1, n)
    fact = [math.factorial(i) for i in range(n + 1)]
    w = np.empty(len(combs))
    for r_, row in enumerate(combs):
        _, cnt = np.unique(row, return_counts=True)
        w[r_] = fact[n] / np.prod([fact[c] for c in cnt])
    return torch.from_numpy(combs), torch.from_numpy(w)


def logits_of(E, W, ops, p, n, act="prod", chunk=200_000):
    """Full logit table (R, p) in fp64."""
    out = torch.empty(len(ops), p, dtype=E.dtype, device=E.device)
    for s in range(0, len(ops), chunk):
        o = ops[s:s + chunk].to(E.device)
        if act == "prod":
            h = torch.ones(o.shape[0], E.shape[1], dtype=E.dtype, device=E.device)
            for j in range(n):
                h = h * E[0 if E.shape[0] == 1 else j][:, o[:, j]].T
        else:
            x = torch.zeros(o.shape[0], E.shape[1], dtype=E.dtype, device=E.device)
            for j in range(n):
                x = x + E[0 if E.shape[0] == 1 else j][:, o[:, j]].T
            h = x ** n
        out[s:s + chunk] = h @ W.T
    return out


def ce_of(z, y, w, alpha=1.0):
    """Weighted mean cross-entropy of alpha * z, in nats."""
    zz = z * alpha
    ls = torch.logsumexp(zz, 1) - zz.gather(1, y[:, None]).squeeze(1)
    return float((ls * w).sum() / w.sum())


def basis(p, k, device, dtype):
    x = torch.arange(p, dtype=dtype, device=device)
    c = torch.cos(2 * math.pi * k * x / p)
    s = torch.sin(2 * math.pi * k * x / p)
    return c / c.norm(), s / s.norm()


def cut_logits(z, p, ks, keep=False):
    """Project the cos/sin of every k in ks out of the CLASS axis of the logit table (keep=False),
    or keep only those frequencies plus the class-constant part (keep=True)."""
    if keep:
        out = z.mean(1, keepdim=True).expand_as(z).clone()          # the class-constant part
        for k in ks:
            c, s = basis(p, k, z.device, z.dtype)
            out = out + (z @ c)[:, None] * c[None, :] + (z @ s)[:, None] * s[None, :]
        return out
    out = z.clone()
    for k in ks:
        c, s = basis(p, k, z.device, z.dtype)
        out = out - (out @ c)[:, None] * c[None, :] - (out @ s)[:, None] * s[None, :]
    return out


def alpha_for_ce(z, y, w, target):
    """Scalar alpha with CE(alpha z) = target, by bisection on log alpha (CE is decreasing in alpha)."""
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ce_of(z, y, w, math.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def loss_ablation(z, y, w, p, alpha, abs_frac=0.01, log=print, tag=""):
    nf = (p - 1) // 2
    ce0 = ce_of(z, y, w, alpha)
    d = np.array([ce_of(cut_logits(z, p, [k]), y, w, alpha) - ce0 for k in range(1, nf + 1)])
    dmax = float(d.max()) if len(d) else 0.0
    thr_abs = abs_frac * math.log(p)
    k10 = [int(k + 1) for k in np.nonzero(d >= 0.10 * dmax)[0]] if dmax > 0 else []
    k25 = [int(k + 1) for k in np.nonzero(d >= 0.25 * dmax)[0]] if dmax > 0 else []
    kabs = [int(k + 1) for k in np.nonzero(d >= thr_abs)[0]]
    counted = k25 or k10
    restricted = ce_of(cut_logits(z, p, counted, keep=True), y, w, alpha) if counted else float("nan")
    excluded = ce_of(cut_logits(z, p, counted), y, w, alpha) if counted else float("nan")
    order = np.argsort(-d)
    out = dict(alpha=alpha, ce0=ce0, dce=[float(x) for x in d], dce_max=dmax, thr_abs=thr_abs,
               K_loss_10=len(k10), K_loss_25=len(k25), K_loss_abs=len(kabs),
               freqs_10=k10, freqs_25=k25, freqs_abs=kabs,
               sorted_dce=[[int(order[i] + 1), float(d[order[i]])] for i in range(min(12, len(d)))],
               restricted_loss=restricted, excluded_loss=excluded)
    log(f"    {tag}: CE0 {ce0:.4g} | K_loss 10%/25%/abs = {out['K_loss_10']}/{out['K_loss_25']}/"
        f"{out['K_loss_abs']} | top dCE {[round(v, 4) for _, v in out['sorted_dce'][:6]]} | "
        f"restricted {restricted:.4g} excluded {excluded:.4g}")
    return out


# ------------------------------------------------------------------ (2) Morwani neuron histogram
def neuron_fourier(E, W, p):
    """Per-neuron max normalized Fourier power and argmax frequency, over all token-indexed vectors."""
    nf = (p - 1) // 2
    m = E.shape[1]
    pw = torch.zeros(m, nf, dtype=DT, device=E.device)
    for j in range(E.shape[0]):
        f = torch.fft.rfft(E[j], dim=1)                       # (m, p//2+1)
        pw += (f[:, 1:nf + 1].abs() ** 2).to(DT)
    f = torch.fft.rfft(W.T, dim=1)
    pw += (f[:, 1:nf + 1].abs() ** 2).to(DT)
    tot = pw.sum(1)
    ok = tot > 0
    maxp = torch.zeros(m, dtype=DT, device=E.device)
    arg = torch.zeros(m, dtype=torch.long, device=E.device)
    maxp[ok] = pw[ok].max(1).values / tot[ok]
    arg[ok] = pw[ok].argmax(1) + 1
    nrm = (E ** 2).sum((0, 2)) * (E.shape[0] == 1 and 1 or 1) + (W ** 2).sum(0)
    nrm = nrm.sqrt()
    hist = np.zeros(nf)
    whist = np.zeros(nf)
    for i in range(m):
        if arg[i] > 0:
            hist[int(arg[i]) - 1] += 1
            whist[int(arg[i]) - 1] += float(nrm[i])
    wm = whist.sum()
    share = whist / wm if wm > 0 else whist
    return dict(max_norm_power_median=float(maxp[ok].median()) if ok.any() else float("nan"),
                max_norm_power_mean=float(maxp[ok].mean()) if ok.any() else float("nan"),
                hist=[int(x) for x in hist], whist=[float(x) for x in whist],
                n_freq_5pct_mass=int((share >= 0.05).sum()),
                top_freq_share=[[int(k + 1), float(share[k])] for k in np.argsort(-share)[:10]],
                uniformity=float(1.0 / (share ** 2).sum()) if wm > 0 else float("nan"))


# ------------------------------------------------------------------ driver
def score_one(E, W, p, n, tied, act, name, ce_target, log=print):
    ops, w = rows_and_weights(p, n, tied)
    y = (ops.sum(1) % p).to(E.device)
    w = w.to(E.device)
    z = logits_of(E, W, ops, p, n, act)
    ce_tr = ce_of(z, y, w, 1.0)
    a_norm = alpha_for_ce(z, y, w, ce_target)
    log(f"  {name}: CE(trained) {ce_tr:.4g}, alpha for CE={ce_target} is {a_norm:.4g}")
    res = dict(name=name, p=p, n=n, tied=tied, ce_trained=ce_tr, alpha_norm=a_norm,
               trained=loss_ablation(z, y, w, p, 1.0, log=log, tag="trained scale"),
               normalised=loss_ablation(z, y, w, p, a_norm, log=log, tag="normalised scale"),
               neuron=neuron_fourier(E, W, p))
    nu = n + 1
    log(f"    Morwani: median max-norm-power {res['neuron']['max_norm_power_median']:.3f}, "
        f"freqs with >=5% of weighted mass {res['neuron']['n_freq_5pct_mass']}, "
        f"effective #freqs (1/sum share^2) {res['neuron']['uniformity']:.1f}")
    return res


def validate(p, ns=(2, 3), Ks=(3, 5), ce_target=0.1, log=print):
    """A constructed junk-free K-net's logits live exactly on its K frequencies, so K_loss must be K."""
    from ladder_core import best_K_support, build_net, gmat
    out = []
    for n in ns:
        for K in Ks:
            g = gmat(p)
            bk = best_K_support(g, K, (n + 1) / 2.0, shortlist=40, polish=False)
            act = [k for k, b in zip(bk["S"], bk["b"]) if b > 1e-4]
            E, W = build_net("chain", p, n, bk["S"], bk["b"], 2.0)
            E = torch.tensor(E, dtype=DT, device=DEV)
            W = torch.tensor(W, dtype=DT, device=DEV)
            r = score_one(E, W, p, n, n >= 4, "prod", f"constructed p{p} n{n} K{len(act)}", ce_target, log=log)
            ok = r["normalised"]["K_loss_25"] == len(act) and sorted(r["normalised"]["freqs_25"]) == sorted(act)
            log(f"    -> truth {len(act)} {sorted(act)}; K_loss_25 {r['normalised']['K_loss_25']} "
                f"{sorted(r['normalised']['freqs_25'])}  {'OK' if ok else 'MISMATCH'}")
            out.append(dict(p=p, n=n, K_true=len(act), S_true=sorted(act),
                            K_loss_25=r["normalised"]["K_loss_25"], K_loss_10=r["normalised"]["K_loss_10"],
                            K_loss_abs=r["normalised"]["K_loss_abs"],
                            freqs_25=sorted(r["normalised"]["freqs_25"]), ok=bool(ok),
                            neuron=r["neuron"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "LADDER" / "train" / "loss_scored.json"))
    ap.add_argument("--ce-target", type=float, default=0.1)
    ap.add_argument("--validate", type=int, default=0)
    a = ap.parse_args()
    if a.validate:
        out = validate(a.validate, ce_target=a.ce_target)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(Path(a.out).with_name("loss_validate.json"), "w"), indent=1)
        print("wrote", Path(a.out).with_name("loss_validate.json"))
        return
    from safetensors.torch import load_file
    rows = []
    for r in a.runs:
        rp = Path(r)
        if not rp.is_dir():
            continue
        for sd in sorted(rp.glob("seed*")):
            f = sd / "final.safetensors"
            if not f.exists():
                continue
            st = load_file(str(f))
            mm = json.loads((sd / "metrics.json").read_text())
            E = st["E"].to(DT).to(DEV)
            W = st["W"].to(DT).to(DEV)
            p, n = mm["p"], mm["n"]
            q = mm.get("q") or (n + 1 if mm.get("penalty") == "l2nu" else 2.0)
            try:
                res = score_one(E, W, p, n, bool(mm.get("tied", False)), mm.get("act", "prod"),
                                f"{rp.name}/{sd.name}", a.ce_target)
            except Exception as e:                                       # noqa: BLE001
                print(f"  !! {sd}: {e}", flush=True)
                continue
            res.update(run=rp.name, seed=sd.name, q=q, lam=mm.get("lam"), nu=n + 1,
                       K_5pct=mm.get("K_5pct"), N_05=mm.get("N_05"), clean_frac=mm.get("clean_frac"),
                       acc=mm.get("acc"), margin=mm.get("margin"))
            rows.append(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
