"""MAIS-O58 / O52 / O61 in the POSED ensemble: one-hidden-layer net
f(a,b)(c) = sum_i sigma(u_i(a)+v_i(b)) w_i(c), full addition table, CE + wd,
N(0,tau^2) init, full-batch optimizer.  Batched over seeds.  Records per run:
key set K_eps (frequency classes carrying >= eps of centered-logit L2 norm),
spectral count N_delta, and tests (i) is K_eps a uniform k-set given |K|=k?
(ii) rate of additive 'collision trios' {i,j,k} with i±j≡±k (mod p) versus the
uniform-k-set null -- the sum-free veto law from our transformer cohorts."""
import argparse, json, math, itertools, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=11)
ap.add_argument("--m", type=int, default=0)          # 0 -> 4p
ap.add_argument("--act", default="relu", choices=["relu", "quad"])
ap.add_argument("--opt", default="adam", choices=["adam", "gd"])
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--decoupled", action="store_true", help="AdamW-style p*=(1-lr*wd) instead of L2 penalty in loss")
ap.add_argument("--tau", type=float, default=0.0)    # 0 -> p^-1/2
ap.add_argument("--seeds", type=int, default=200)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--eps", type=float, default=0.005)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="notes/mais")
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)
p, S = a.p, a.seeds
m = a.m or 4 * p
tau = a.tau or p ** -0.5
dev = torch.device(a.device)
torch.manual_seed(p * 1000 + S)
torch.set_num_threads(4)
A, B = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
A, B = A.reshape(-1).to(dev), B.reshape(-1).to(dev); Y = (A + B) % p
act = torch.relu if a.act == "relu" else (lambda x: x * x)

U = (tau * torch.randn(S, m, p, device=dev)).requires_grad_()
V = (tau * torch.randn(S, m, p, device=dev)).requires_grad_()
W = (tau * torch.randn(S, m, p, device=dev)).requires_grad_()
params = [U, V, W]
opt = (torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8) if a.opt == "adam"
       else torch.optim.SGD(params, lr=a.lr))

def logits(U, V, W):
    h = act(U[:, :, A] + V[:, :, B])                  # (S, m, N)
    return torch.einsum("smn,smc->snc", h, W)         # (S, N, p)

t0 = time.time()
prev = None
for it in range(a.steps):
    f = logits(U, V, W)
    ce = torch.nn.functional.cross_entropy(f.reshape(-1, p), Y.repeat(S), reduction="none").view(S, -1).mean(1)
    loss = ce.sum() + (0 if a.decoupled else a.wd * sum((t * t).sum((1, 2)) for t in params).sum())
    opt.zero_grad(); loss.backward(); opt.step()
    if a.decoupled:
        with torch.no_grad():
            for t in params: t.mul_(1 - a.lr * a.wd)
    if it % 2000 == 0 or it == a.steps - 1:
        with torch.no_grad():
            acc = (f.argmax(2) == Y).float().mean(1)
        print(f"it {it:6d} ce med {ce.median():.3e} min {ce.min():.2e} max {ce.max():.2e} "
              f"acc<1: {(acc < 1).sum().item()}/{S}  ({time.time()-t0:.0f}s)", flush=True)

# ---- spectral readout of the CENTERED logits on (a,b,c): characters of a+b-c
with torch.no_grad():
    Uc, Vc, Wc = (t.detach().cpu() for t in (U, V, W))
    A, B, Y = A.cpu(), B.cpu(), Y.cpu(); dev = torch.device("cpu")
    torch.save(dict(U=Uc, V=Vc, W=Wc), f"{a.out}/_w_{p}_{a.act}_{a.opt}_wd{a.wd:g}_s{S}.pt") if Path(a.out).exists() else None
    f = logits(Uc, Vc, Wc).double()                    # (S, N, p)
    f = f - f.mean(2, keepdim=True)                    # center over c per input pair
    total = f.pow(2).sum((1, 2))
    nf = (p - 1) // 2
    z = (A + B - torch.arange(p, device=dev)[:, None]).T % p      # (N, p): a+b-c
    mass = torch.zeros(S, nf, dtype=torch.float64, device=dev)
    for k in range(1, nf + 1):
        c = torch.cos(2 * math.pi * k * z.double() / p); s = torch.sin(2 * math.pi * k * z.double() / p)
        # projection of f onto span{cos,sin of k(a+b-c)} : these are orthogonal on G^3
        pc = (f * c).sum((1, 2)) / c.pow(2).sum(); ps = (f * s).sum((1, 2)) / s.pow(2).sum()
        mass[:, k - 1] = pc.pow(2) * c.pow(2).sum() + ps.pow(2) * s.pow(2).sum()
    frac = mass / total[:, None]                        # share of centered L2 norm per class
    resid = 1 - frac.sum(1)                             # mass outside the a+b-c characters
    acc = (logits(Uc, Vc, Wc).argmax(2) == Y).float().mean(1)

frac = frac.cpu().numpy(); resid = resid.cpu().numpy(); acc = acc.cpu().numpy()
def fold(x): x %= p; return min(x, p - x)
def collision(Sset):
    Sset = sorted(Sset)
    for i, j in itertools.combinations(Sset, 2):
        for k in (fold(i + j), fold(i - j)):
            if k in Sset and k not in (i, j): return True
    return False
rows = []
for s in range(S):
    K = [k + 1 for k in range(nf) if frac[s, k] >= a.eps]
    order = np.argsort(frac[s])[::-1]; cum = np.cumsum(frac[s][order]) / frac[s].sum()
    N = int(np.searchsorted(cum, 1 - a.delta) + 1)
    rows.append(dict(seed=s, K=K, k=len(K), N=N, collision=collision(K),
                     resid=float(resid[s]), acc=float(acc[s]), top=[float(x) for x in np.sort(frac[s])[::-1][:6]]))
# ---- null: uniform k-sets
rng = np.random.default_rng(0)
summary = dict(p=p, m=m, act=a.act, opt=a.opt, decoupled=a.decoupled, lr=a.lr, wd=a.wd, tau=tau, seeds=S, steps=a.steps,
               acc1=float(np.mean([r["acc"] == 1 for r in rows])),
               mean_resid=float(np.mean([r["resid"] for r in rows])),
               k_hist={}, N_hist={}, collision={})
for k in sorted(set(r["k"] for r in rows)):
    rr = [r for r in rows if r["k"] == k]
    obs = np.mean([r["collision"] for r in rr])
    null = np.mean([collision(rng.choice(np.arange(1, nf + 1), k, replace=False).tolist()) for _ in range(5000)]) if 0 < k <= nf else float("nan")
    summary["k_hist"][k] = len(rr)
    summary["collision"][k] = dict(n=len(rr), obs_rate=float(obs), null_rate=float(null),
                                   obs_count=int(sum(r["collision"] for r in rr)))
for N in sorted(set(r["N"] for r in rows)):
    summary["N_hist"][N] = sum(r["N"] == N for r in rows)
# set-frequency table (for the uniform-k-set test)
from collections import Counter
summary["set_counts"] = {str(k): v for k, v in Counter(tuple(r["K"]) for r in rows).most_common(20)}
print(json.dumps(summary, indent=1))
Path(a.out).mkdir(parents=True, exist_ok=True)
tag = f"p{p}_m{m}_{a.act}_{a.opt}_wd{a.wd:g}{'dec' if a.decoupled else ''}_s{S}"
Path(f"{a.out}/{tag}.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
