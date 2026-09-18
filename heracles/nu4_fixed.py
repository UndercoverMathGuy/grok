"""nu4_fixed.py -- measured FIXED-SUPPORT ladder gamma(K) on the exact nu families (nsum.py: --act prod / pow).

For each K: take the ladder's M_K-optimal frequency set S_K (kstar.py), build the junk-free K-frequency network
analytically, then fine-tune it on CE + lam ||theta||^2 with every embedding row and readout column PROJECTED onto the
Fourier span of S_K after each step (so the support stays exactly S_K; neurons may mix frequencies inside S_K).
gamma = min-margin / ||theta||^nu (nsum.analyse, fp64) is recorded at the construction and after each lam stage
(lam -> 0 continuation). Compare gamma(K)/max with the predicted M_K / K^{nu/2} / max.

Constructions (frequency k, x_j = 2 pi k a_j / p):
  prod untied: 2^n neurons, e_j = cos(x_j + phi_j), phi in {0, pi/2}^n, w = cos(x_c + sum phi)   (THEORY_nsum sec. 6)
  prod tied  : n+1 neurons, e = cos(x + pi t/(n+1)), w = cos(x_c + n pi t/(n+1)), t = 0..n
  pow untied : 2^n phases x 2^(n-1) signs: e_j = s_j cos(x_j + phi_j), w = (prod s) cos(x_c + sum phi)  (kills u^2 v, u^3)
  pow tied   : not constructed (use untied)
Optimiser: normalised GD (step = eta_t ||theta|| g/||g||, eta cosine-decayed per stage) + exact 1-D line search on the
global scale every --ls_every steps (the net is nu-homogeneous, so the slow weight-decay norm mode is solved exactly).
CE is computed in the gap form log(1 + sum_o exp(z_o - z_y)), stable in fp32 at large margins.

Usage: python3 heracles/nu4_fixed.py --act prod --n 3 --p 71 --Ks 2 3 4 5 6 7 8 10 --seeds 0 1 2
"""
import argparse, itertools, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsum  # noqa: E402
from kstar import best_subset  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEV = nsum.DEV


def construct(act, p, n, S, tied, reps, rng, noise, pop="cube"):
    a = np.arange(p); Es, Ws = [], []
    for k in S:
        x = 2 * np.pi * k * a / p
        for _ in range(reps):
            d = rng.uniform(0, 2 * np.pi, n) if noise > 0 else np.zeros(n)   # population-wide phase shift: keeps exactness
            if tied or pop == "common":               # (n+1)-neuron common-phase population (PROOF_AUDIT finding 1)
                assert act == "prod"
                for t in range(n + 1):
                    ph = np.pi * t / (n + 1) + d[0]
                    Es.append([np.cos(x + ph)] * (1 if tied else n)); Ws.append(np.cos(x + n * ph))
            else:
                signs = [(1,) + s for s in itertools.product((1, -1), repeat=n - 1)] if act == "pow" else [(1,) * n]
                for phi in itertools.product((0.0, np.pi / 2), repeat=n):
                    for sg in signs:
                        Es.append([sg[j] * np.cos(x + phi[j] + d[j]) for j in range(n)])
                        Ws.append(np.prod(sg) * np.cos(x + sum(phi) + d.sum()))
    E = torch.tensor(np.array(Es)).permute(1, 0, 2).contiguous()      # (ne, m, p)
    W = torch.tensor(np.array(Ws)).T.contiguous()                    # (p, m)
    if noise > 0:
        g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
        E = E * (1 + noise * torch.randn(E.shape, generator=g, dtype=E.dtype)); W = W * (1 + noise * torch.randn(W.shape, generator=g, dtype=W.dtype))
    return E, W


def make_proj(p, S, dev):
    mask = torch.zeros(p // 2 + 1, device=dev); mask[list(S)] = 1
    def proj(E, W):
        E.copy_(torch.fft.irfft(torch.fft.rfft(E, dim=-1) * mask, n=p, dim=-1))
        W.copy_(torch.fft.irfft(torch.fft.rfft(W, dim=0) * mask[:, None], n=p, dim=0))
    return proj


def gap_ce(z, y, w, N):
    """sum_rows w * log(1 + sum_{o != y} exp(z_o - z_y)) / N, stable at large margins."""
    d = z - z.gather(1, y[:, None])
    d = d.scatter(1, y[:, None], -float("inf"))
    return (F.softplus(torch.logsumexp(d, 1)) * w).sum() / N


def ce_and_logits(data, E, W, act):
    return [(data.forward(E, W, slabs, act).T, y, w) for slabs, y, w in data.chunks]


def line_search(data, E, W, act, lam, nu):
    """argmin_s CE(s^nu z) + lam s^2 ||theta||^2 (golden section on log s), applied in place."""
    with torch.no_grad():
        zs = ce_and_logits(data, E, W, act); n2 = float(E.square().sum() + W.square().sum())
        f = lambda t: sum(float(gap_ce(math.exp(nu * t) * z, y, w, data.N_full)) for z, y, w in zs) + lam * math.exp(2 * t) * n2
        lo, hi = -1.0, 1.0; gr = (math.sqrt(5) - 1) / 2
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo); fc, fd = f(c), f(d)
        for _ in range(28):
            if fc < fd:
                hi, d, fd = d, c, fc; c = hi - gr * (hi - lo); fc = f(c)
            else:
                lo, c, fc = c, d, fd; d = lo + gr * (hi - lo); fd = f(d)
        s = math.exp((lo + hi) / 2); E *= s; W *= s
    return s


def short(res):
    return {k: res[k] for k in ("margin", "acc", "norm2", "gamma", "clean_frac", "K_2pct", "K_5pct", "N_05", "N_10", "top_share", "amp")}


def run(a, K, S, seed, data, data64, ckpt=None):
    act, n, p = a.act, a.n, a.p; nu = n + 1
    out = Path(a.out) / f"{'polish' if ckpt else 'free' if a.no_proj else 'fixed'}_{act}_n{n}_p{p}{'_tied' if a.tied else ''}{'_common' if a.pop == 'common' and not a.tied else ''}{a.tag}" / (f"K{K}_seed{seed}" if ckpt is None else ckpt[0])
    if (out / "metrics.json").exists():
        print("skip", out); return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 * K + seed)
    if ckpt is None:
        E, W = construct(act, p, n, S, a.tied, a.reps, rng, a.noise, a.pop)
    else:
        from safetensors.torch import load_file
        st = load_file(ckpt[1]); E, W = st["E"].double(), st["W"].double()
    E, W = E.to(DEV), W.to(DEV)
    proj = (lambda E, W: None) if a.no_proj else make_proj(p, S, DEV)
    stages = []
    with torch.no_grad():
        proj(E, W)
    r0 = nsum.analyse(E, W, data64, act)
    stages.append(dict(stage="construction", lam=None, **short(r0)))
    print(f"K{K} s{seed} m{E.shape[1]} construction: margin>0 {r0['margin'] > 0} gamma {r0['gamma']:.4e} clean {r0['clean_frac']:.3f}", flush=True)
    E = E.float().requires_grad_(True); W = W.float().requires_grad_(True)
    t0 = time.time()
    for lam in a.lams:
        line_search(data, E, W, act, lam, nu)
        for step in range(1, a.steps + 1):
            eta = a.eta * (a.eta_min / a.eta) ** (step / a.steps) if a.decay == "exp" else a.eta_min + 0.5 * (a.eta - a.eta_min) * (1 + math.cos(math.pi * step / a.steps))
            gE = torch.zeros_like(E); gW = torch.zeros_like(W); ce = 0.0
            for slabs, y, w in data.chunks:
                l = gap_ce(data.forward(E, W, slabs, act).T, y, w, data.N_full)
                dE, dW = torch.autograd.grad(l, (E, W)); gE += dE; gW += dW; ce += float(l.detach()) if step % 500 == 0 else 0.0
            with torch.no_grad():
                gE += 2 * lam * E; gW += 2 * lam * W
                gn = torch.sqrt(gE.square().sum() + gW.square().sum()); tn = torch.sqrt(E.square().sum() + W.square().sum())
                sc = eta * tn / (gn + 1e-30)
                E -= sc * gE; W -= sc * gW
                proj(E, W)
            if step % a.ls_every == 0:
                line_search(data, E, W, act, lam, nu)
            if step % 500 == 0:
                print(f"  K{K} s{seed} lam {lam:g} step {step} ce {ce:.3e} |g|/|th| {float(gn / tn):.2e} [{time.time() - t0:.0f}s]", flush=True)
        r = nsum.analyse(E, W, data64, act)
        stages.append(dict(stage=f"lam{lam:g}", lam=lam, **short(r)))
        print(f"K{K} s{seed} lam {lam:g}: margin {r['margin']:.3f} gamma {r['gamma']:.4e} (x{r['gamma'] / r0['gamma']:.3f} constr) acc {r['acc']:.4f} "
              f"clean {r['clean_frac']:.3f} N05 {r['N_05']} [{time.time() - t0:.0f}s]", flush=True)
    res = dict(act=act, n=n, p=p, nu=nu, tied=a.tied, K=K, S=list(S), seed=seed, m=int(E.shape[1]), reps=a.reps, noise=a.noise,
               lams=a.lams, steps=a.steps, eta=a.eta, stages=stages, gamma_best=max(s["gamma"] for s in stages), train_time=time.time() - t0)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    from safetensors.torch import save_file
    save_file({"E": E.detach().cpu(), "W": W.detach().cpu()}, str(out / "final.safetensors"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="prod", choices=["prod", "pow"])
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--p", type=int, default=71)
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-5, 1e-6, 1e-7])
    ap.add_argument("--steps", type=int, default=3000, help="per lam stage")
    ap.add_argument("--eta", type=float, default=3e-3); ap.add_argument("--eta_min", type=float, default=3e-5)
    ap.add_argument("--decay", default="exp", choices=["exp", "cos"])
    ap.add_argument("--ls_every", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1, help="copies of the cancelling population per frequency")
    ap.add_argument("--noise", type=float, default=0.05, help="relative weight noise on the construction (0 = exact, seed-independent)")
    ap.add_argument("--no_proj", action="store_true", help="no Fourier projection: the construction (or checkpoint) is free to change support")
    ap.add_argument("--ckpts", nargs="*", default=[], help="polish mode: nsum final.safetensors files (implies --no_proj); same optimiser, no construction")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pop", default="cube", choices=["cube", "common"], help="prod untied: 2^n phase cube, or the (n+1)-neuron common-phase population")
    ap.add_argument("--chunk", type=int, default=400_000)
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "NU4"))
    a = ap.parse_args()
    print("device", DEV, vars(a), flush=True)
    nf = (a.p - 1) // 2; sets = {}
    for K in ([] if a.ckpts else a.Ks):            # M_K-optimal sets (kstar.best_subset: exhaustive when small, else local search); K = nf: all
        sets[K] = list(range(1, nf + 1)) if K >= nf else best_subset(a.p, K, np.random.default_rng(0))[1]
        print(f"S_{K} = {sets[K]}", flush=True)
    grid = True                                # untied: broadcast grid; tied prod: index gathers (nsum.Data)
    data = nsum.Data(a.p, a.n, a.tied, a.chunk, DEV, prod=(a.act == "prod"), grid=grid)
    data64 = nsum.Data(a.p, a.n, a.tied, a.chunk, DEV, dtype=torch.float64, prod=(a.act == "prod"), grid=grid)
    if a.ckpts:
        a.no_proj = True
        for c in a.ckpts:
            cp = Path(c); run(a, 0, [], 0, data, data64, ckpt=(f"{cp.parent.parent.name}__{cp.parent.name}", str(cp)))
    else:
        for K in a.Ks:
            for s in a.seeds:
                run(a, K, sets[K], s, data, data64)
