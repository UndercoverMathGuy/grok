# Global search over arbitrary product networks (not slot-symmetric, all frequencies, DC allowed) at small (p,n,b),
# with diagnostics of each local maximum:  gamma/gamma_IF, feature count vs K*, cheapness, split margin, Sigma-orbit,
# decoupling ratio with the energy split (b = 2 only meaningful), coordination gain (class-C pieces).
import sys, json, numpy as np, itertools
from lib import *
from transitions import kstar

def features(net, rel=1e-5):
    A = np.abs(net.clean()); return A, [j + 1 for j in range(len(A)) if A[j] > rel * A.max()]

def sigma_orbit(net, trials, rng):
    """margins of random sign-flip images sigma theta, sigma in Sigma = (Z_2)^{F x n} (output follows the product)."""
    p = net.p; F = (p - 1) // 2; out = []
    Pk = [freq_proj(p, k) for k in range(1, F + 1)]
    for _ in range(trials):
        sg = rng.choice([-1, 1], size=(F, net.n))
        U = net.U.copy(); W = net.W.copy()
        for k in range(F):
            for t in range(net.n):
                if sg[k, t] < 0: U[:, t] -= 2 * net.U[:, t] @ Pk[k]
            if np.prod(sg[k]) < 0: W -= 2 * net.W @ Pk[k]
        out.append(Net(U, W, net.b).margin())
    return np.array(out)

def energy_split(net):
    """P_k = sum_i sum_factors |P_k v|^2 (frequency-k energy), P_0 = DC energy (b=2: cost = sum_k P_k + P_0)."""
    p = net.p; F = (p - 1) // 2; E = []
    for k in range(0, F + 1):
        Pk = freq_proj(p, k)
        E.append(((net.U @ Pk) ** 2).sum() + ((net.W @ Pk) ** 2).sum())
    return np.array(E)

def diagnostics(net, Gam, ks, rng):
    p, n, b = net.p, net.n, net.b; nu = n + 1; s = nu / b
    gIF = gamma_if(p, n, b, Gam); g = net.gamma()
    A, feats = features(net); Aq, Q, a = net.cheapness()
    sp = split(net); gsp = sp.gamma()
    sig = sigma_orbit(net, 16, rng); h = net.margin()
    E = energy_split(net); K = kappa_mat(p)
    # decoupling ratio, IF price, split: each neuron's cost distributed over frequencies by energy (DC pro rata)
    Pj = np.zeros((p - 1) // 2); costs = ((net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)) ** (b / 2)
    for k in range(1, (p - 1) // 2 + 1):
        Pk = freq_proj(p, k)
        Eik = ((net.U @ Pk) ** 2).sum((1, 2)) + ((net.W @ Pk) ** 2).sum(1)
        Pj[k - 1] = 0.0 + (costs * Eik / np.maximum((net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1) - 0 * Eik, 1e-300)).sum()
    Pj = Pj / Pj.sum() * net.cost()
    dec = h / (nu ** (1 - s) * Bp(p, n) * (K @ (Pj ** s)).min())
    return dict(ratio=g / gIF, count=len(feats), feats=feats, Kstar=ks, amps=np.round(A / A.max(), 6).tolist(),
                cheap=[round(float(Aq[j] / a[j]), 4) if (j + 1) in feats else None for j in range(len(Aq))], split_ratio=gsp / g,
                sigma_min=float(sig.min() / h), sigma_mean=float(sig.mean() / h), dc_frac=float(E[0] / E.sum()),
                dec_ratio=float(dec), width=net.m)

def start(kind, p, n, m, rng, b):
    F = (p - 1) // 2
    if kind == 'rand':
        return rng.normal(size=(m, n, p)), rng.normal(size=(m, p))
    K = int(kind[2:]); S = rng.choice(np.arange(1, F + 1), size=min(K, F), replace=False)
    y = np.zeros(F); y[S - 1] = rng.uniform(0.5, 1.5, len(S))
    net = if_network(p, n, y, b); U, W = net.U, net.W
    sc = np.sqrt((U ** 2).sum((1, 2)).mean() / (n * p))
    U = U + 0.15 * sc * rng.normal(size=U.shape); W = W + 0.15 * sc * rng.normal(size=W.shape)
    if U.shape[0] < m:
        U = np.concatenate([U, 0.3 * sc * rng.normal(size=(m - U.shape[0], n, p))]); W = np.concatenate([W, 0.3 * sc * rng.normal(size=(m - W.shape[0], p))])
    return U[:m], W[:m]

if __name__ == "__main__":
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    m = int(sys.argv[4]); R = int(sys.argv[5]); kinds = sys.argv[6].split(","); seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    rng = np.random.default_rng(seed); nu = n + 1; s = nu / b
    ks, Gam, ex = kstar(p, s)
    print(f"# p={p} n={n} b={b} s={s:.4f} Gamma={Gam:.6f} K*={ks} supports={[ex[k][1] for k in ks]}", flush=True)
    res = []
    for r in range(R):
        kind = kinds[r % len(kinds)]
        U, W = start(kind, p, n, m, rng, b)
        net = optimise(U, W, b)
        net = prune(net, 1e-10)
        d = diagnostics(net, Gam, ks, rng); d['start'] = kind; res.append(d)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
        np.savez(f"gs_p{p}_n{n}_b{b}_m{m}_r{r}_s{seed}.npz", U=net.U, W=net.W, b=b)
    best = max(res, key=lambda d: d['ratio'])
    print(f"# BEST ratio {best['ratio']:.6f} count {best['count']} K* {ks}", flush=True)
