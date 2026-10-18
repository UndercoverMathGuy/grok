"""Referee checks of SPLIT R1 (frequency split lemma) and R2 (sign-flip averaging), independent code.
Random + adversarial (sparse / dead pieces / heavy DC / single-frequency) networks.
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python v_r1_r2.py
"""
import itertools, numpy as np
from vlib import *

rng = np.random.default_rng(20261018)
worst = {}
def rec(k, v):
    worst[k] = max(worst.get(k, -np.inf), float(v))

cases = [(3, 2, 3), (3, 3, 2), (5, 2, 3), (5, 3, 2), (7, 2, 3), (7, 3, 2), (11, 2, 2), (5, 4, 2)]
for (p, n, m) in cases:
    F = F_(p); nu = n + 1
    for kind in ['gauss', 'sparse', 'heavydc']:
        for trial in range(3):
            U, W = rand_net(rng, m, n, p, kind)
            f = table(U, W)
            Us, Ws = split(U, W)
            fs = table(Us, Ws)
            # linearity: split table = diagonal-mode projection of the table
            rec('R1 |table(split) - diagproj(table)|', np.abs(fs - diag_proj_table(f)).max())
            e = piece_energies(U, W)          # e[:,0] = DC energy, e[:,k] = piece k
            # energy decomposition
            rec('R1 |sum_k e_k - ||omega||^2|', np.abs(e.sum(1) - ((U ** 2).sum((1, 2)) + (W ** 2).sum(1))).max())
            for b in [2.0, 2.3, 2.5, 3.0, float(nu)]:
                q = b / 2
                P, Ps = cost(U, W, b), cost(Us, Ws, b)
                rec('R1a cost(split)-cost  (<=0)', Ps - P)
                # per-neuron slack lower bound (R1b)
                for i in range(m):
                    e0 = e[i, 0]; ek = e[i, 1:]; E = ek.sum(); M = ek.max()
                    slack = (E + e0) ** q - (ek ** q).sum()
                    lb = q * E ** (q - 1) * e0 + (q - 1) * M ** (q - 1) * (E - M)
                    rec('R1b lower bound - slack (<=0)', lb - slack)
                    if b == 2.0:
                        rec('R1b |slack_b2 - e0|', abs(slack - e0))
                # P_live <= P_split <= P
                Pl = sum(P_live_by_freq(U, W, b).values())
                rec('R1c P_live - P_split (<=0)', Pl - Ps)
            # own data preserved: from the TABLES (FFT), not from atoms
            b1, bT1 = own_from_table(f); b2, bT2 = own_from_table(fs)
            rec('R1c |beta - beta_split| (table FFT)', max(abs(b1[j] - b2[j]) for j in F))
            rec('R1c |betaT - betaT_split| (table FFT)', max(abs(bT1[k] - bT2[k]) for k in bT1))
            # atoms: neuron i's atom at j == atom of piece (i, j); pieces (i,k!=j) have zero j-atom
            A = atoms(U, W); As = atoms(Us, Ws)
            for j in F:
                for i in range(m):
                    for kk, k in enumerate(F):
                        a_piece = As[j][i * len(F) + kk]
                        rec('R1c |atom piece - atom neuron| (k=j) or |atom| (k!=j)',
                            abs(a_piece - A[j][i]) if k == j else abs(a_piece))
            # E_x g preserved
            rec('R1c |E_x g - E_x g_split|', np.abs(gaps(f).mean(tuple(range(n))) - gaps(fs).mean(tuple(range(n)))).max())
            # beta from atoms == beta from table (check of conventions)
            rec('conv |sum atoms - beta(table)|', max(abs(A[j].sum() - b1[j]) for j in F))

# equality cases R1a
p, n, m = 7, 2, 3
U, W = rand_net(rng, m, n, p)
U0, W0 = fproj(U, F_(p), p), fproj(W, F_(p), p)       # no DC
Us, Ws = split(U0, W0)
print('R1a equality, b=2 no DC:        cost-cost_split = %.3e' % (cost(U0, W0, 2) - cost(Us, Ws, 2)))
print('R1a strict,   b=2.5 no DC 3 fr: cost-cost_split = %.3e' % (cost(U0, W0, 2.5) - cost(Us, Ws, 2.5)))
U1, W1 = fproj(U, [3], p), fproj(W, [3], p)
Us, Ws = split(U1, W1)
print('R1a equality, b=3 single freq:  cost-cost_split = %.3e' % (cost(U1, W1, 3) - cost(Us, Ws, 3)))
U2, W2 = fproj(U, [0, 3], p), fproj(W, [3], p)
Us, Ws = split(U2, W2)
print('R1a strict,   b=2 freq+DC:      cost-cost_split = %.3e' % (cost(U2, W2, 2) - cost(Us, Ws, 2)))
# edge case: pure DC neuron (E = 0) -> slack = e0^q > 0, table is the all-DC block
U3, W3 = fproj(U, [0], p), fproj(W, [0], p)
Us, Ws = split(U3, W3)
print('R1  pure-DC neuron: split cost = %.3e, gaps of table = %.3e' % (cost(Us, Ws, 2), np.abs(gaps(table(U3, W3))).max()))

# ---------------- R2: sign-flip averaging
def flip(U, W, S, d=None, dout=1):
    """S[k-1,t] signs on plane k of slot t; output plane k gets prod_t S[k-1,t]; d[t] input DC signs; dout output DC sign."""
    m, n, p = U.shape
    U2 = fproj(U, [0], p) * (1 if d is None else np.asarray(d)[None, :, None])
    W2 = fproj(W, [0], p) * dout
    for kk, k in enumerate(F_(p)):
        U2 = U2 + fproj(U, [k], p) * S[kk][None, :, None]
        W2 = W2 + fproj(W, [k], p) * np.prod(S[kk])
    return U2, W2

for (p, n, m) in [(3, 2, 3), (5, 2, 3), (5, 3, 2), (7, 2, 2), (3, 4, 2)]:
    F = F_(p)
    for kind in ['gauss', 'heavydc', 'sparse']:
        U, W = rand_net(rng, m, n, p, kind)
        f = table(U, W); Us, Ws = split(U, W); fs = table(Us, Ws)
        b0, bT0 = own_from_table(f)
        acc = np.zeros_like(f); accD = np.zeros_like(f); hs = []; cnt = 0; cntD = 0
        for sg in itertools.product([1, -1], repeat=len(F) * n):
            S = np.array(sg).reshape(len(F), n)
            U2, W2 = flip(U, W, S)
            f2 = table(U2, W2); acc += f2; cnt += 1; hs.append(margin_f(f2))
            rec('R2(1) |cost change| b=2.7', abs(cost(U2, W2, 2.7) - cost(U, W, 2.7)))
            bb, bbT = own_from_table(f2)
            rec('R2(1) |own data change|', max(max(abs(bb[j] - b0[j]) for j in F), max(abs(bbT[k] - bT0[k]) for k in bT0)))
            for dd in itertools.product([1, -1], repeat=n):
                U3, W3 = flip(U, W, S, d=dd)
                accD += table(U3, W3); cntD += 1
        avg = acc / cnt; avgD = accD / cntD
        rec('R2(3) |Sigma-avg - (split + allDC)|', np.abs(avg - fs - alldc_table(f)).max())
        rec("R2(3) Sigma'-avg (DC flips): max|gaps(avg - split)|", np.abs(gaps(avgD - fs)).max())
        rec('R2(4) E h(sigma theta) - h(split) (<=0)', np.mean(hs) - margin_f(fs))
        # output-DC sign flip also fixes all own data (stabiliser larger than Sigma' as defined in R2/R3)
        U4, W4 = flip(U, W, np.ones((len(F), n)), dout=-1)
        bb, bbT = own_from_table(table(U4, W4))
        rec('R3 output-DC flip: |own data change|', max(max(abs(bb[j] - b0[j]) for j in F), max(abs(bbT[k] - bT0[k]) for k in bT0)))
        # per-neuron group
        for i in range(m):
            acc = np.zeros_like(f); cnt = 0
            for sg in itertools.product([1, -1], repeat=len(F) * n):
                a_, b_ = flip(U[i:i + 1], W[i:i + 1], np.array(sg).reshape(len(F), n))
                acc += table(a_, b_); cnt += 1
            usi, wsi = split(U[i:i + 1], W[i:i + 1])
            rec('R2(3) per-neuron |avg - split - allDC|', np.abs(acc / cnt - table(usi, wsi) - alldc_table(table(U[i:i + 1], W[i:i + 1]))).max())
        # R2(5) partial split over K subset of F
        if len(F) >= 2:
            K = [F[0]]
            acc = np.zeros_like(f); cnt = 0
            for sg in itertools.product([1, -1], repeat=n):
                S = np.ones((len(F), n)); S[0] = sg
                U2, W2 = flip(U, W, S); acc += table(U2, W2); cnt += 1
            # partial split network
            Up = np.concatenate([fproj(U, [k], p) for k in K] + [U - fproj(U, K, p)], 0)
            Wp = np.concatenate([fproj(W, [k], p) for k in K] + [W - fproj(W, K, p)], 0)
            rec('R2(5) |partial avg - partial split table|', np.abs(acc / cnt - table(Up, Wp)).max())
            for b in [2.0, 2.5, float(n + 1)]:
                rec('R2(5) cost(partial)-cost (<=0)', cost(Up, Wp, b) - cost(U, W, b))

print()
for k, v in worst.items():
    print('%-70s %+.3e' % (k, v))
