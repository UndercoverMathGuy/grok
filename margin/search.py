"""Option 2: hunt for NON-Fourier max-margin solutions of mod-p addition in a
one-hidden-layer net  f_c(a,b) = sum_i w_ic * relu(u_ia + v_ib)  (one-hot in,
2-homogeneous).  Normalized margin gamma = min_{a,b}[f_y - max_{c!=y} f_c] / ||theta||^2.
Morwani et al. 2023 prove the max-margin solution is Fourier-sparse for
QUADRATIC activation; ReLU is open.  We maximize a smoothed gamma from many
random restarts (batched), then score each solution's Fourier sparsity.
A high-margin, low-sparsity solution = counterexample candidate; its margin is
then recomputed EXACTLY in f64 (the certificate)."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=11)
ap.add_argument("--m", type=int, default=0, help="width (0 -> 4p)")
ap.add_argument("--restarts", type=int, default=64)
ap.add_argument("--fourier", type=int, default=16, help="Fourier-initialized restarts")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--act", default="relu", choices=["relu", "quad"])
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="notes/margin")
a = ap.parse_args()
torch.manual_seed(a.seed); torch.set_num_threads(2)
p, m = a.p, a.m or 4 * a.p
R = a.restarts + a.fourier
A, B = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
A, B = A.reshape(-1), B.reshape(-1); Y = (A + B) % p
act = (lambda x: torch.relu(x)) if a.act == "relu" else (lambda x: x * x)

def fourier_init(n):
    k = torch.randint(1, p // 2 + 1, (n, m, 1)).double()
    ph = torch.rand(n, m, 1).double() * 2 * math.pi
    t = 2 * math.pi * torch.arange(p).double() / p
    u = torch.cos(k * t + ph); v = torch.cos(k * t + ph)
    w = torch.cos(k * t - 2 * ph)
    return u, v, w

U = torch.randn(R, m, p).double(); V = torch.randn(R, m, p).double()
W = torch.randn(R, m, p).double()
if a.fourier:
    fu, fv, fw = fourier_init(a.fourier)
    U[a.restarts:], V[a.restarts:], W[a.restarts:] = fu, fv, fw
U, V, W = (t + 0.01 * torch.randn_like(t) for t in (U, V, W))

def normalize(U, V, W):
    n = (U.pow(2).sum((1, 2)) + V.pow(2).sum((1, 2)) + W.pow(2).sum((1, 2))).sqrt()
    return U / n[:, None, None], V / n[:, None, None], W / n[:, None, None]

def margins(U, V, W):
    """per-input margins (R, p*p) of the unit-norm net."""
    U, V, W = normalize(U, V, W)
    h = act(U[:, :, A] + V[:, :, B])              # (R, m, N)
    f = torch.einsum("rmn,rmc->rnc", h, W)         # (R, N, p)
    fy = f.gather(2, Y[None, :, None].expand(R, -1, 1)).squeeze(2)
    fo = f.masked_fill(torch.nn.functional.one_hot(Y, p).bool()[None], -1e9)
    return fy, fo

params = [t.clone().requires_grad_() for t in (U, V, W)]
opt = torch.optim.Adam(params, lr=a.lr)
t0 = time.time()
for it in range(a.steps):
    beta = 10.0 * (1 + 20 * it / a.steps)         # anneal softness
    fy, fo = margins(*params)
    mo = torch.logsumexp(beta * fo, 2) / beta      # soft max over wrong classes
    g = fy - mo                                     # (R, N)
    soft_min = -torch.logsumexp(-beta * g, 1) / beta
    loss = -soft_min.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0 or it == a.steps - 1:
        with torch.no_grad():
            fy, fo = margins(*params)
            exact = (fy - fo.max(2).values).min(1).values
        print(f"it {it:5d} beta {beta:6.1f} best exact gamma {exact.max():.5f} "
              f"median {exact.median():.5f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- exact scoring + Fourier sparsity
with torch.no_grad():
    Un, Vn, Wn = normalize(*[t.detach() for t in params])
    fy, fo = margins(Un, Vn, Wn)
    gamma = (fy - fo.max(2).values).min(1).values          # exact, f64
    def sparsity(X):   # (R, m, p) -> per-restart neuron-weighted top-freq power share
        F = torch.fft.rfft(X, dim=2)[:, :, 1:]             # drop DC
        P = F.abs().pow(2)
        share = P.max(2).values / P.sum(2).clamp_min(1e-30)
        wgt = X.pow(2).sum(2)
        return (share * wgt).sum(1) / wgt.sum(1)
    sp = (sparsity(Un) + sparsity(Vn) + sparsity(Wn)) / 3
    order = torch.argsort(gamma, descending=True)
    rows = []
    for i in order.tolist():
        rows.append(dict(restart=i, fourier_init=i >= a.restarts,
                         gamma=float(gamma[i]), sparsity=float(sp[i])))
    print(f"\np={p} m={m} act={a.act}: top solutions (exact gamma, Fourier top-freq share)")
    for r in rows[:12]:
        print(f"  #{r['restart']:3d} {'F' if r['fourier_init'] else 'R'}  gamma {r['gamma']:.5f}  sparsity {r['sparsity']:.3f}")
    best_rand = max((r for r in rows if not r["fourier_init"]), key=lambda r: r["gamma"])
    best_four = max((r for r in rows if r["fourier_init"]), key=lambda r: r["gamma"], default=None)
    print(f"best random-init gamma {best_rand['gamma']:.5f} (sparsity {best_rand['sparsity']:.3f}); "
          f"best Fourier-init gamma {best_four['gamma'] if best_four else float('nan'):.5f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"p{p}_m{m}_{a.act}_s{a.seed}"
    torch.save(dict(U=Un, V=Vn, W=Wn, gamma=gamma, sparsity=sp), f"{a.out}/{tag}.pt")
    Path(f"{a.out}/{tag}.json").write_text(json.dumps(dict(p=p, m=m, act=a.act, rows=rows), indent=1))
