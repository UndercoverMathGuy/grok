"""fvf_fixed.py -- de-circularising controls for the FIXED-SUPPORT ladder (RESULTS_NU4 T1).

The archived ladder starts every run AT the analytic junk-free construction, so "measured = predicted" could be
true by construction.  This script runs the same optimiser and the same Fourier projection but changes the START:

  --start con    : the analytic junk-free (n+1)-neuron common-phase construction + relative weight noise --noise
                   (this is the archived recipe when --noise 0.05)
  --start rand   : RANDOM Gaussian weights projected onto the frequency span of S_K.  Nothing about the theory's
                   construction is used; only the support is fixed.  Width is chosen by --width.
  --start randfull: random Gaussian weights, NO projection at all (free support, free junk) -- the "free GD with a
                   good optimiser" arm, for reference.

Symbols.  p = prime modulus; n = number of operands; nu = n+1 = homogeneity degree (multiplying every weight by s
multiplies the logits by s^nu); K = number of Fourier frequencies allowed; S_K = the coverage-optimal K-set
(kstar.best_subset, maximiser of M_K = min over wrong offsets D != 0 of sum_{k in S} (1 - cos 2 pi k D / p));
gamma = (smallest margin over the full p^n-row table) / ||theta||^nu, the l2-normalised margin, fp64;
clean fraction = share of the centred-logit energy that is a function of (sum of operands - answer) alone
(1 - junk); N_.05 = number of frequencies holding the top 95% of the clean-logit energy.

Optimiser (identical to nu4_fixed.py): normalised gradient descent (step = eta * ||theta|| * g / ||g||) on
CE + lam ||theta||^2, with an exact golden-section line search on the global scale every --ls_every steps, and lam
annealed through --lams.  Projection (when on) is applied after every step.

Usage: uv run python heracles/fvf_fixed.py --n 2 --p 71 --start rand --Ks 2 3 4 5 6 8 --seeds 0 1
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import nsum  # noqa: E402
from kstar import best_subset  # noqa: E402
from nu4_fixed import construct, make_proj, gap_ce, line_search, short  # noqa: E402

DEV = nsum.DEV


def rand_init(n, m, p, S, rng, proj, scale):
    """Random Gaussian weights, projected onto the Fourier span of S (when proj is given), then rescaled so that
    ||theta|| = scale.  Built on the CPU and projected there, so the caller moves the result to the device."""
    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    E = torch.randn(n, m, p, generator=g, dtype=torch.float64)
    W = torch.randn(p, m, generator=g, dtype=torch.float64)
    if proj is not None:
        proj(E, W)
    nrm = math.sqrt(float(E.square().sum() + W.square().sum()))
    return E * (scale / nrm), W * (scale / nrm)


def run(a, K, S, seed, data, data64):
    n, p, nu = a.n, a.p, a.n + 1
    tag = f"{a.start}_{a.act}_n{n}_p{p}_{a.pop}{'_np' if a.no_proj else ''}{a.tag}"
    out = Path(a.out) / tag / f"K{K}_seed{seed}"
    if (out / "metrics.json").exists():
        print("skip", out, flush=True); return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 * K + seed)
    proj = None if (a.no_proj or a.start == "randfull") else make_proj(p, S, DEV)
    if a.start == "con":
        E, W = construct(a.act, p, n, S, False, a.reps, rng, a.noise, a.pop)
        E, W = E.to(DEV), W.to(DEV)
        if proj is not None:
            with torch.no_grad():
                proj(E, W)
    else:
        m = a.width if a.width else (n + 1) * K * a.reps
        E, W = rand_init(n, m, p, S, rng, make_proj(p, S, "cpu") if a.start == "rand" else None, a.scale)
        E, W = E.to(DEV), W.to(DEV)
    stages = []
    r0 = nsum.analyse(E, W, data64, a.act)
    stages.append(dict(stage="start", lam=None, **short(r0)))
    print(f"[{tag}] K{K} s{seed} m{E.shape[1]} start: gamma {r0['gamma']:.4e} margin {r0['margin']:.3f} clean {r0['clean_frac']:.3f}", flush=True)
    E = E.float().requires_grad_(True); W = W.float().requires_grad_(True)
    t0 = time.time()
    for lam in a.lams:
        line_search(data, E, W, a.act, lam, nu)
        for step in range(1, a.steps + 1):
            eta = a.eta * (a.eta_min / a.eta) ** (step / a.steps)
            gE = torch.zeros_like(E); gW = torch.zeros_like(W)
            for slabs, y, w in data.chunks:
                l = gap_ce(data.forward(E, W, slabs, a.act).T, y, w, data.N_full)
                dE, dW = torch.autograd.grad(l, (E, W)); gE += dE; gW += dW
            with torch.no_grad():
                gE += 2 * lam * E; gW += 2 * lam * W
                gn = torch.sqrt(gE.square().sum() + gW.square().sum()); tn = torch.sqrt(E.square().sum() + W.square().sum())
                E -= (eta * tn / (gn + 1e-30)) * gE; W -= (eta * tn / (gn + 1e-30)) * gW
                if proj is not None:
                    proj(E, W)
            if step % a.ls_every == 0:
                line_search(data, E, W, a.act, lam, nu)
        r = nsum.analyse(E, W, data64, a.act)
        stages.append(dict(stage=f"lam{lam:g}", lam=lam, **short(r)))
        print(f"[{tag}] K{K} s{seed} lam {lam:g}: margin {r['margin']:7.3f} gamma {r['gamma']:.4e} clean {r['clean_frac']:.4f} "
              f"N05 {r['N_05']:2d} top_share {r['top_share']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    res = dict(start=a.start, act=a.act, n=n, p=p, nu=nu, K=K, S=list(S), seed=seed, m=int(E.shape[1]), pop=a.pop,
               noise=a.noise, scale=a.scale, no_proj=a.no_proj, lams=a.lams, steps=a.steps, eta=a.eta, stages=stages,
               gamma_best=max(s["gamma"] for s in stages), train_time=time.time() - t0)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--p", type=int, default=71)
    ap.add_argument("--act", default="prod", choices=["prod", "pow"])
    ap.add_argument("--start", default="rand", choices=["con", "rand", "randfull"])
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 10, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-5, 1e-6, 1e-7])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=3e-3); ap.add_argument("--eta_min", type=float, default=3e-5)
    ap.add_argument("--ls_every", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=0, help="rand/randfull width (0 = (n+1)*K*reps, matched to the construction)")
    ap.add_argument("--scale", type=float, default=60.0, help="||theta|| at a random start (the line search fixes it anyway)")
    ap.add_argument("--no_proj", action="store_true")
    ap.add_argument("--pop", default="common", choices=["cube", "common"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=400_000)
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "FVF"))
    a = ap.parse_args()
    print("device", DEV, vars(a), flush=True)
    nf = (a.p - 1) // 2; sets = {}
    for K in a.Ks:
        sets[K] = list(range(1, nf + 1)) if K >= nf else best_subset(a.p, K, np.random.default_rng(0))[1]
    data = nsum.Data(a.p, a.n, False, a.chunk, DEV, prod=(a.act == "prod"), grid=True)
    data64 = nsum.Data(a.p, a.n, False, a.chunk, DEV, dtype=torch.float64, prod=(a.act == "prod"), grid=True)
    for K in a.Ks:
        for s in a.seeds:
            run(a, K, sets[K], s, data, data64)
