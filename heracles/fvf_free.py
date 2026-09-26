"""fvf_free.py -- can FREE training (random init, no support projection) be driven to the max-margin optimum?

Cell: the gated chain z(c; a_1..a_n) = sum_i w_i(c) prod_j e_ij(a_j) on the full table of a_1+...+a_n = c (mod p);
homogeneity degree nu = n+1 (scaling every weight by s scales the logits by s^nu).  Plain weight decay means the
penalty lam * ||theta||^2 = lam * (sum of squares of every weight).

Routes (--mode):
  l2       plain weight decay from random init -- the baseline that lands in the "junk basin".
  qnu      penalty lam_q * sum_i (||theta_i||^2)^(nu/2) (= the L_{2,nu} norm Morwani/Li et al. price, for which a
           frequency may be split over many neurons for free, so junk-free is free) for --steps, then plain l2 for
           --then_l2 steps.  This is the archived "q = nu first, then plain l2" recipe.
  qanneal  the same but with the penalty exponent q moved CONTINUOUSLY from nu down to 2 over the first
           --anneal_frac of the run, then held at 2 (a homotopy instead of a hard switch).
  junk     plain l2 plus mu * (junk energy of the logit table) for the first --junk_frac of the run, then pure l2.
           "Junk" = the part of the centred logit table that is NOT a function of (sum of operands - answer);
           the theory's junk lemma says it can only cost margin, so this penalty removes the thing the basin carries.
  ascent   direct margin ascent: maximise a softmin of (correct logit - wrong logit) / ||theta||^nu with Adam and an
           annealed softmin temperature (no cross-entropy, no weight decay).  Scale-free, so it optimises gamma itself.

Every route ends with the SAME measurement: gamma = (min margin over the full table) / ||theta||^nu in fp64, the
clean fraction (1 - junk share of the centred-logit energy), and N_.05 (number of frequencies carrying the top 95%
of the clean-logit energy).  With --polish the run additionally gets the fixed-support optimiser (normalised GD +
exact line search on the scale, lam annealed to 1e-7, NO projection) so that gamma can be compared with the
constructed ceiling on equal footing.

Usage: uv run python heracles/fvf_free.py --mode l2 --n 2 --p 71 --seeds 0 1
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import nsum  # noqa: E402
from nu4_fixed import gap_ce, line_search, short  # noqa: E402

DEV = nsum.DEV
LR_DOUBLE = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def junk_energy(z, y, p):
    """JUNK FRACTION: the share of the centred-logit energy that is NOT a function of (sum a_j - c).
    z: (N, p) logits, y: (N,) correct answers.  g(D) = mean over rows of z[row, y - D] is the clean part.
    Scale-free (in [0, 1]), so penalising it does not fight the global scale."""
    zc = z - z.mean(1, keepdim=True)
    D = torch.arange(p, device=z.device)
    idx = (y[:, None] - D[None, :]) % p
    g = zc.gather(1, idx).mean(0)                       # (p,) clean profile g(D)
    c = torch.arange(p, device=z.device)
    gt = g[(y[:, None] - c[None, :]) % p]               # (N, p) clean table
    return (zc - gt).square().sum() / (zc.square().sum() + 1e-30)


def penalty(E, W, q, lam, nu):
    if q == 2.0:
        return lam * (E.square().sum() + W.square().sum())
    n2 = E.square().sum((0, 2)) + W.square().sum(0)
    return lam * n2.pow(q / 2).sum()


def polish(a, E, W, data, data64, nu, lams=(1e-5, 1e-6, 1e-7), steps=3000, eta=3e-3, eta_min=3e-5):
    E = E.detach().clone().float().requires_grad_(True); W = W.detach().clone().float().requires_grad_(True)
    for lam in lams:
        line_search(data, E, W, "prod", lam, nu)
        for step in range(1, steps + 1):
            et = eta * (eta_min / eta) ** (step / steps)
            gE = torch.zeros_like(E); gW = torch.zeros_like(W)
            for slabs, y, w in data.chunks:
                l = gap_ce(data.forward(E, W, slabs, "prod").T, y, w, data.N_full)
                dE, dW = torch.autograd.grad(l, (E, W)); gE += dE; gW += dW
            with torch.no_grad():
                gE += 2 * lam * E; gW += 2 * lam * W
                gn = torch.sqrt(gE.square().sum() + gW.square().sum()); tn = torch.sqrt(E.square().sum() + W.square().sum())
                E -= (et * tn / (gn + 1e-30)) * gE; W -= (et * tn / (gn + 1e-30)) * gW
            if step % 100 == 0:
                line_search(data, E, W, "prod", lam, nu)
    return nsum.analyse(E, W, data64, "prod"), E, W


def ascent(a, E, W, data64, p, n, nu, steps):
    """Direct margin ascent on gamma = softmin_rows (z_y - z_c) / ||theta||^nu (scale-free), Adam + temperature anneal."""
    ops = torch.stack([(torch.arange(p ** n) // (p ** j)) % p for j in range(n)], 1).to(DEV)
    y = (ops.sum(1) % p).to(DEV)
    E = E.detach().clone().requires_grad_(True); W = W.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=0.01)
    oh = F.one_hot(y, p).bool()
    best, bestEW = -1e30, None
    for t in range(steps):
        cols = [E[j][:, ops[:, j]] for j in range(n)]
        h = cols[0]
        for j in range(1, n):
            h = h * cols[j]
        z = (W @ h).T
        n2 = E.square().sum() + W.square().sum()
        d = (z.gather(1, y[:, None]) - z)[~oh] / n2 ** (nu / 2)
        sc = d.detach().abs().mean() + 1e-30
        beta = 2.0 * (1 + 400 * (t / steps) ** 2)
        sm = -torch.logsumexp(-beta * d / sc, 0) / beta * sc
        opt.zero_grad(); (-sm / sc).backward(); opt.step()
        if t % 200 == 0 or t == steps - 1:
            v = float(d.min())
            if v > best:
                best, bestEW = v, (E.detach().clone(), W.detach().clone())
        if t == int(0.7 * steps):
            for gp in opt.param_groups:
                gp["lr"] = 0.002
    return bestEW


def main(a):
    n, p, nu = a.n, a.p, a.n + 1
    data = nsum.Data(p, n, False, a.chunk, DEV, prod=True, grid=True)
    data64 = nsum.Data(p, n, False, a.chunk, DEV, dtype=torch.float64, prod=True, grid=True)
    lnp = math.log(p)
    for seed in a.seeds:
        tag = f"{a.mode}_n{n}_p{p}_m{a.m}_init{a.init:g}_lam{a.lam:g}{a.tag}"
        out = Path(a.out) / tag / f"seed{seed}"
        if (out / "metrics.json").exists():
            print("skip", out, flush=True); continue
        out.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator().manual_seed(seed)
        E = a.init * (torch.rand(n, a.m, p, generator=gen) * 2 - 1) / math.sqrt(n * p)
        W = a.init * (torch.rand(p, a.m, generator=gen) * 2 - 1) / math.sqrt(a.m)
        E = E.to(DEV).requires_grad_(True); W = W.to(DEV).requires_grad_(True)
        t0 = time.time(); rec = []
        if a.mode == "ascent":
            E, W = ascent(a, E, W, data64, p, n, nu, a.steps)
            r = nsum.analyse(E, W, data64, "prod")
            rec.append(dict(stage="ascent", **short(r)))
            print(f"[{tag} s{seed}] ascent: gamma {r['gamma']:.4e} margin {r['margin']:.3f} clean {r['clean_frac']:.3f} N05 {r['N_05']}", flush=True)
        else:
            total = a.steps + (a.then_l2 if a.mode in ("qnu", "qanneal", "junk") else 0)
            lr, lrcap = 0.05, a.lrmax
            warm = a.warmup > 0
            for step in range(1, total + 1):
                if step in LR_DOUBLE:
                    lr = min(lr * 2, lrcap)
                if a.mode == "l2":
                    q, lam, mu = 2.0, a.lam, 0.0
                elif a.mode == "qnu":
                    q, lam, mu = (float(nu), a.lam_q, 0.0) if step <= a.steps else (2.0, a.lam, 0.0)
                elif a.mode == "qanneal":
                    fr = min(1.0, step / max(1, int(a.anneal_frac * a.steps)))
                    q = nu + (2.0 - nu) * fr
                    lam = a.lam_q * (a.lam / a.lam_q) ** fr
                    mu = 0.0
                else:                                   # junk
                    q, lam = 2.0, a.lam
                    mu = a.mu if step <= a.junk_frac * a.steps else 0.0
                gE = torch.zeros_like(E); gW = torch.zeros_like(W); ce = 0.0
                for slabs, yy, w in data.chunks:
                    z = data.forward(E, W, slabs, "prod").T
                    l = (F.cross_entropy(z, yy, reduction="none") * w).sum() / data.N_full
                    if mu > 0:
                        l = l + mu * junk_energy(z, yy, p)
                    dE, dW = torch.autograd.grad(l, (E, W)); gE += dE; gW += dW; ce += float(l.detach())
                if warm and (ce < a.escape_frac * lnp or step >= a.warmup):
                    warm = False; lrcap = a.lrmax2; lr = min(lr, lrcap)
                    print(f"  warm-up ends at step {step} (ce {ce:.3f})", flush=True)
                if not warm:
                    pen = penalty(E, W, q, lam, nu)
                    dE, dW = torch.autograd.grad(pen, (E, W)); gE += dE; gW += dW
                with torch.no_grad():
                    gn = torch.sqrt(gE.square().sum() + gW.square().sum())
                    if a.clip and gn > a.clip:
                        gE *= a.clip / gn; gW *= a.clip / gn
                    E -= lr * gE; W -= lr * gW
                if step % a.report == 0 or step == total:
                    r = nsum.analyse(E, W, data64, "prod")
                    rec.append(dict(stage=f"step{step}", q=q, lam=lam, mu=mu, **short(r)))
                    print(f"[{tag} s{seed}] step {step:6d} q {q:.2f} ce {ce:.4f} margin {r['margin']:8.3f} gamma {r['gamma']:.4e} "
                          f"clean {r['clean_frac']:.3f} N05 {r['N_05']:2d} K5 {r['K_5pct']:2d} [{time.time()-t0:.0f}s]", flush=True)
        if a.polish:
            r, E, W = polish(a, E, W, data, data64, nu)
            rec.append(dict(stage="polish", **short(r)))
            print(f"[{tag} s{seed}] POLISH: margin {r['margin']:.3f} gamma {r['gamma']:.4e} clean {r['clean_frac']:.4f} "
                  f"N05 {r['N_05']} K5 {r['K_5pct']} [{time.time()-t0:.0f}s]", flush=True)
        (out / "metrics.json").write_text(json.dumps(dict(mode=a.mode, n=n, p=p, nu=nu, m=a.m, seed=seed, args=vars(a),
                                                          stages=rec, train_time=time.time() - t0), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="l2", choices=["l2", "qnu", "qanneal", "junk", "ascent"])
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--p", type=int, default=71)
    ap.add_argument("--m", type=int, default=150); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--then_l2", type=int, default=15000)
    ap.add_argument("--lam", type=float, default=1e-5); ap.add_argument("--lam_q", type=float, default=1e-6)
    ap.add_argument("--mu", type=float, default=1.0, help="junk-penalty weight (mode junk)")
    ap.add_argument("--junk_frac", type=float, default=0.5); ap.add_argument("--anneal_frac", type=float, default=0.6)
    ap.add_argument("--init", type=float, default=4.0); ap.add_argument("--lrmax", type=float, default=12.8)
    ap.add_argument("--lrmax2", type=float, default=12.8); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=20000); ap.add_argument("--escape_frac", type=float, default=0.8)
    ap.add_argument("--report", type=int, default=5000); ap.add_argument("--polish", action="store_true")
    ap.add_argument("--chunk", type=int, default=400_000); ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "FVF"))
    a = ap.parse_args()
    print("device", DEV, vars(a), flush=True)
    main(a)
