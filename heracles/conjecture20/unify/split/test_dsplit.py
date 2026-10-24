"""Test of the canonical-split decoupling inequalities on many networks (p >= 5), with the conjectured price nu^{1-s}:

 (D_split)  h(theta) <= nu^{1-s} B_p (P / P_feat)^s  min_delta sum_{j in feat} P_j^s kappa_j(delta),
            P_j = sum over neurons with beta_ij != 0 of ||omega_i^{(j)}||^b (live frequency-j pieces), P_feat = sum_j P_j;
 (D_Q)      h(theta) <= nu^{1-s} (C_b P / sum_l Q_l)^s  min_delta sum_l Q_l^s kappa_l(delta),   Q_l = sum_i |beta_il|^r.
Both imply the exact count for every global maximiser (Prop. 'decoupling suffices', with pi replaced by the price used);
with the true price pi_{n,b}(p) in place of nu^{1-s} both are trivially true at p = 3.  We report the ratio
LHS/RHS (must be <= 1) over: random networks (unoptimised), optimised random starts, IF + mixed noise starts.
usage: python test_dsplit.py p n b widths seeds
"""
import sys, json
import numpy as np, torch
from netopt import *

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
widths = [int(v) for v in sys.argv[4].split(',')]; seeds = int(sys.argv[5])
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
Bp_ = Bp(p, n); Cb = Bp_ ** (1 / s)
model = Full(p, n)
F = list(range(1, (p - 1) // 2 + 1))
K = kappa_mat(p)                                  # rows delta = 1..p-1


def ratios(U, W):
    with torch.no_grad():
        h = float(model.gaps(torch.tensor(U), torch.tensor(W)).min()); P = float(model.cost(torch.tensor(U), torch.tensor(W), b))
    at = own(U, W, p, n)
    beta = np.array([at[j].sum() for j in F]); Q = np.array([(np.abs(at[j]) ** r).sum() for j in F])
    amax = np.abs(beta).max()
    feat = np.abs(beta) > 1e-7 * amax
    # live pieces
    Pj = np.zeros(len(F))
    for i in range(len(U)):
        for jj, j in enumerate(F):
            if abs(at[j][i]) > 1e-12 * max(1e-300, amax):
                piece = np.concatenate([proj(U[i], j, p).ravel(), proj(W[i], j, p)])
                Pj[jj] += (piece ** 2).sum() ** (b / 2)
    Pj = Pj * feat
    Pf = Pj.sum()
    rhs_split = nu ** (1 - s) * Bp_ * (P / Pf) ** s * (K @ (Pj ** s)).min()
    rhs_Q = nu ** (1 - s) * (Cb * P / Q.sum()) ** s * (K @ (Q ** s)).min()
    if_val = gamma_IF(p, n, b, Gam) * P ** s
    return h / rhs_split, h / rhs_Q, h / if_val, int(feat.sum())


rng = np.random.default_rng(int(100 * b) + p + 7 * n)
rows = []
for m in widths:
    for sd in range(seeds):
        U0 = rng.normal(size=(m, n, p)); W0 = rng.normal(size=(m, p))
        rows.append(('raw',) + ratios(U0, W0))
        U, W, h, P = optimise(model, U0, W0, b)
        rows.append(('opt',) + ratios(U, W))
pos = [r_ for r_ in rows if r_[3] > 0]
print(f"p={p} n={n} b={b}: {len(rows)} networks ({len(pos)} with h>0)")
for tag in ['raw', 'opt']:
    sel = [r_ for r_ in rows if r_[0] == tag and r_[3] > 0]
    if sel:
        print(f"  {tag}: max h/RHS(D_split) = {max(x[1] for x in sel):.4f}   max h/RHS(D_Q) = {max(x[2] for x in sel):.4f}   "
              f"max gamma/gamma_IF = {max(x[3] for x in sel):.4f}   (n_pos={len(sel)})")
viol = [r_ for r_ in pos if r_[1] > 1 + 1e-6 or r_[2] > 1 + 1e-6]
print("  violations (ratio > 1):", len(viol), viol[:5])
json.dump(rows, open(f'test_dsplit_p{p}_n{n}_b{b}.json', 'w'))
