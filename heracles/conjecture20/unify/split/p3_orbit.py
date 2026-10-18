"""p = 3 checks of the split / sign-flip picture on the certified counterexamples (b = 2).

For each certified network theta (slot-symmetric, p = 3):
  * exact (rational) margin h(theta), cost P, gamma/gamma_IF;
  * the split theta_split (drop the DC components of inputs and outputs; at p = 3 F = {1}):  exact h, P_split,
    gamma(theta_split)/gamma_IF, DC energy fraction;
  * the Sigma-orbit: Sigma = {+-1}^n flips the frequency-1 component of the inputs in a slot set S and multiplies
    the output frequency-1 component by (-1)^{|S|}; by slot symmetry h(sigma theta) depends on k = |S| only.
    Exact margins for every k, and E_sigma h(sigma theta) = 2^{-n} sum_k C(n,k) h_k  (<= h(theta_split), Lemma S3);
  * a sample of per-neuron flips (each neuron its own slot set; float arithmetic over all 3^n inputs, n <= 13).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy --with mpmath python p3_orbit.py
"""
import sys, json, math, itertools
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '../..')
from exact_certificate import NETWORKS
import certify_1d

p = 3


def counts(n):
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            yield (N0, N1, n - N0 - N1)


def margin_groups(groups, W, n_by_group):
    """Exact margin of a network whose slots are partitioned into groups; within group g every slot of neuron i
    uses the vector groups[g][i].  W[i] = output vector (Fractions)."""
    m = len(W)
    h = None
    lists = [list(counts(ng)) for ng in n_by_group]
    for combo in itertools.product(*lists):
        y = sum(N[1] + 2 * N[2] for N in combo) % p
        z = []
        for i in range(m):
            v = Fr(1)
            for g, N in enumerate(combo):
                u = groups[g][i]
                v *= u[0] ** N[0] * u[1] ** N[1] * u[2] ** N[2]
            z.append(v)
        f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(p)]
        g_ = min(f[y] - f[c] for c in range(p) if c != y)
        h = g_ if h is None or g_ < h else h
    return h


def cost2(U, W, n):
    return sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(len(U)))


def analyse(U, W, n, label, per_neuron_samples=0):
    nu = n + 1
    m = len(U)
    gIF = Fr(3 * nu) / Fr(6 * nu * nu) ** (nu // 2) if nu % 2 == 0 else None
    gIF_f = nu ** (1 - nu / 2) * 2 * (6 * nu) ** (-nu / 2) * 1.5
    mean = lambda v: sum(v) / 3
    U0 = [[mean(u)] * 3 for u in U]; U1 = [[a - mean(u) for a in u] for u in U]
    W0 = [[mean(w)] * 3 for w in W]; W1 = [[a - mean(w) for a in w] for w in W]
    P = cost2(U, W, n); Ps = cost2(U1, W1, n)
    dcE = P - Ps
    h = margin_groups([U], W, [n])
    hs = margin_groups([U1], W1, [n])
    ratio = lambda hh, PP: float(hh) / float(PP) ** (nu / 2) / gIF_f
    print(f"== {label}: n={n}, width {m}")
    print(f"   h(theta) = {float(h):.6e}, P = {float(P):.6f}, gamma/gamma_IF = {ratio(h, P):.6f}")
    print(f"   DC energy fraction (b=2 split slack / P) = {float(dcE / P):.4f}")
    print(f"   h(split) = {float(hs):.6e}  (sign {'+' if hs > 0 else '-'}), P_split = {float(Ps):.6f}, "
          f"gamma(split)/gamma_IF = {ratio(hs, Ps):.6f}")
    # Sigma orbit: k flipped slots
    Um = [[U0[i][a] - U1[i][a] for a in range(3)] for i in range(m)]
    hk = []
    for k in range(n + 1):
        Wk = [[W0[i][a] + (-1) ** k * W1[i][a] for a in range(3)] for i in range(m)]
        hk.append(margin_groups([Um, U], Wk, [k, n - k]))
    avg = sum(math.comb(n, k) * hk[k] for k in range(n + 1)) / 2 ** n
    print("   Sigma-orbit gamma/gamma_IF by k=|S|: " + ", ".join(f"{ratio(x, P):+.3g}" for x in hk))
    print(f"   E_sigma h(sigma theta)/(gamma_IF P^s) = {ratio(avg, P):+.4g}   <=  h(split)/(gamma_IF P^s) = {ratio(hs, P):+.4g}"
          f"   [check: {avg <= hs}]")
    out = dict(label=label, n=n, m=m, ratio=ratio(h, P), dc_frac=float(dcE / P), split_ratio=ratio(hs, Ps),
               split_ratio_sameP=ratio(hs, P), orbit=[ratio(x, P) for x in hk], orbit_avg=ratio(avg, P),
               check_avg_le_split=bool(avg <= hs))
    if per_neuron_samples:
        rng = np.random.default_rng(0)
        Uf = np.array([[float(a) for a in u] for u in U]); Wf = np.array([[float(a) for a in w] for w in W])
        U0f = Uf.mean(1, keepdims=True) * np.ones((1, 3)); U1f = Uf - U0f
        W0f = Wf.mean(1, keepdims=True) * np.ones((1, 3)); W1f = Wf - W0f
        idx = np.indices((3,) * n).reshape(n, -1); y = idx.sum(0) % 3
        vals = []
        for s_ in range(per_neuron_samples):
            S = rng.choice([1, -1], size=(m, n))
            F_ = np.zeros((idx.shape[1], 3))
            for i in range(m):
                z = np.ones(idx.shape[1])
                for t in range(n):
                    u = U0f[i] + S[i, t] * U1f[i]
                    z *= u[idx[t]]
                w = W0f[i] + np.prod(S[i]) * W1f[i]
                F_ += np.outer(z, w)
            rows = np.arange(len(y))
            g = np.min(np.stack([F_[rows, y] - F_[rows, (y - d) % 3] for d in (1, 2)]), 0)
            vals.append(g.min() / float(P) ** (nu / 2) / gIF_f)
        vals = np.array(vals)
        print(f"   per-neuron flips ({per_neuron_samples} samples): gamma/gamma_IF min {vals.min():+.3g}, median {np.median(vals):+.3g}, "
              f"max {vals.max():+.3g};  fraction > 0: {(vals > 0).mean():.3f}")
        out['per_neuron'] = dict(min=float(vals.min()), median=float(np.median(vals)), max=float(vals.max()),
                                 frac_pos=float((vals > 0).mean()))
    return out


if __name__ == "__main__":
    res = []
    for n, net in NETWORKS.items():
        U = [[Fr(a) for a in row] for row in net["u"]]; W = [[Fr(a) for a in row] for row in net["w"]]
        res.append(analyse(U, W, n, f"exact_certificate n={n}", per_neuron_samples=60 if n <= 11 else 0))
    for n in range(11, 19):
        d = json.load(open(f'../../oned_certified/p3_n{n}.json'))
        x = np.concatenate([d['lam'], d['logV'], d['argV']])
        U, W = certify_1d.build(x, 3, n, 9)
        res.append(analyse(U, W, n, f"oned_certified p3_n{n}"))
    json.dump(res, open('p3_orbit.json', 'w'), indent=1)
