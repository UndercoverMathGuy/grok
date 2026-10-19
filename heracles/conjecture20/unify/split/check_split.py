"""Numerical sanity checks of Lemmas S2 (split), S3 (sign-flip averaging), the mode/character bijection and the
'no clean/sign-flipped content in mixed modes' statement, on random networks (exact small cases).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python check_split.py
"""
import itertools, numpy as np
from lib import *

rng = np.random.default_rng(7)
worst = {}
def rec(key, v):
    worst[key] = max(worst.get(key, 0.0), float(v))

cases = [(3, 2, 3), (5, 2, 4), (5, 3, 3), (7, 2, 3), (7, 3, 2), (11, 2, 3), (5, 4, 2)]
for (p, n, m) in cases:
    F = Fset(p); nu = n + 1
    for trial in range(4):
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        f = table(U, W)
        Us, Ws = split(U, W)
        fs = table(Us, Ws)
        # (1) cost inequality for several b, and the slack formula at b = 2 (= DC energy)
        for b in [2.0, 2.5, 3.0, float(nu)]:
            c, cs = cost(U, W, b), cost(Us, Ws, b)
            rec('cost_split_minus_cost (must be <= 0)', cs - c)
        dc = ((U - np.array([[proj(U[i, t], 1, p) * 0 + U[i, t] - sum(proj(U[i, t], k, p) for k in F)
                               for t in range(n)] for i in range(m)])) ** 2).sum()
        dcE = sum((proj(U[i, t], 0, p) ** 2).sum() for i in range(m) for t in range(n)) + \
              sum((proj(W[i], 0, p) ** 2).sum() for i in range(m))
        rec('|slack_b2 - DC energy|', abs(cost(U, W, 2.0) - cost(Us, Ws, 2.0) - dcE))
        # (2) own data preserved
        b1, bT1, at1 = own_coeffs(U, W); b2, bT2, at2 = own_coeffs(Us, Ws)
        rec('|beta - beta_split|', max(abs(b1[j] - b2[j]) for j in F))
        rec('|betaT - betaT_split|', max(abs(bT1[k] - bT2[k]) for k in bT1))
        # atoms preserved: neuron i's atom at j equals the atom of piece (i, j)
        at_s = {j: np.array([at2[j][i * len(F) + (j - 1)] for i in range(m)]) for j in F}
        rec('|atoms - atoms_split|', max(np.abs(at1[j] - at_s[j]).max() for j in F))
        # (3) mixed part R = f - f_split - (all-DC block) has zero clean and zero sign-flipped Fourier content,
        #     and zero mean gap at every offset
        Ud = np.array([[proj(U[i, t], 0, p) for t in range(n)] for i in range(m)]); Wd = proj(W, 0, p)
        R = f - fs - table(Ud, Wd)
        Rh = np.fft.fftn(R)
        mx = 0
        for j in range(p):
            for T in itertools.product([0, 1], repeat=n):
                k = tuple(((-j) if T[t] else j) % p for t in range(n)) + ((-j) % p,)
                mx = max(mx, abs(Rh[k]))
        rec('max |R_hat| on clean+sign-flipped modes', mx)
        GR = gaps_from_table(R, p, n)
        rec('max |E_x g_R(x,d)|', np.abs(GR.mean(0)).max())
        # (4) Sigma average (global flips, incl. DC flips) of tables
        acc = np.zeros_like(f); cnt = 0
        signs = list(itertools.product([1, -1], repeat=len(F) * n))
        if len(signs) > 4096:
            signs = [tuple(rng.choice([1, -1], size=len(F) * n)) for _ in range(4096)]
            exact = False
        else:
            exact = True
        hs = []
        for sg in signs:
            S = np.array(sg).reshape(len(F), n)
            U2, W2 = flip(U, W, S)
            f2 = table(U2, W2); acc += f2; cnt += 1
            hs.append(margin_table(f2, p, n))
            # cost and own data invariant under sigma
            if cnt <= 8:
                rec('|cost(sigma theta)-cost(theta)|', abs(cost(U2, W2, 2.5) - cost(U, W, 2.5)))
                bb, bbT, _ = own_coeffs(U2, W2)
                rec('|own data(sigma theta)-own data|', max(max(abs(bb[j] - b1[j]) for j in F),
                                                             max(abs(bbT[k] - bT1[k]) for k in bT1)))
        if exact:
            avg = acc / cnt
            rec('|Sigma-avg table - (split + DC block)|', np.abs(avg - fs - table(Ud, Wd)).max())
            hsplit = margin_table(fs, p, n)
            rec('E_sigma h(sigma theta) - h(split) (must be <= 0)', np.mean(hs) - hsplit); rec('min over cases of h(split) - E_sigma h (>=0)', -(hsplit - np.mean(hs)))
        # with DC flips as well, the average is exactly the split table (DC blocks killed)
        if exact and n * (len(F) + 1) <= 12:
            acc = np.zeros_like(f); cnt = 0
            for sg in itertools.product([1, -1], repeat=len(F) * n):
                for dcs in itertools.product([1, -1], repeat=n):
                    U2, W2 = flip(U, W, np.array(sg).reshape(len(F), n), dc=np.array(dcs))
                    acc += table(U2, W2); cnt += 1
            avg = acc / cnt
            # output DC is never flipped; the blocks with output DC and all inputs DC survive only if no input
            # DC flips change them -- they carry prod_t dc_t and average out, so avg = f_split + 0
            rec('|Sigma\'(with DC flips)-avg table - split table| (up to output-DC blocks)',
                np.abs(gaps_from_table(avg - fs, p, n)).max())
        # (5) per-neuron flip average: neuron-wise identity
        for i in range(m):
            acc = np.zeros_like(f); cnt = 0
            for sg in itertools.product([1, -1], repeat=len(F) * n) if len(F) * n <= 10 else []:
                a_, b_ = flip(U[i:i + 1], W[i:i + 1], np.array(sg).reshape(len(F), n))
                acc += table(a_, b_); cnt += 1
            if cnt:
                Usi, Wsi = split(U[i:i + 1], W[i:i + 1])
                rec('|per-neuron Sigma-avg - split(neuron)+DC|',
                    np.abs(acc / cnt - table(Usi, Wsi) - table(Ud[i:i + 1], Wd[i:i + 1])).max())
        # (6) mode/character bijection: phi_l(chi) from formula vs FFT of the field computed from the table
        l = F[-1]
        phi = field_coeffs(U, W, l)
        # field from table: Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-l c}
        idx = np.indices((p,) * n).reshape(n, -1); y = idx.sum(0) % p
        rho = np.exp(2j * np.pi / p)
        Fl = (2 / p) * rho ** (l * y) * (f.reshape(-1, p) @ rho ** (-l * np.arange(p)))
        Fl = Fl.reshape((p,) * n)
        phi2 = np.fft.fftn(Fl) / p ** n   # Phi = sum_chi phi(chi) rho^{chi.x}  =>  phi = fftn(Phi)/p^n
        rec('|phi_l(chi) formula - from table|', np.abs(phi - phi2).max())
        # clean characters: phi_l(0) = conj(beta_l), phi_l(2l 1_T) = conj(beta_{l,T})
        rec('|phi_l(0) - conj(beta_l)|', abs(phi[(0,) * n] - np.conj(b1[l])))
        for r in range(1, n + 1):
            for T in itertools.combinations(range(n), r):
                chi = tuple((2 * l) % p if t in T else 0 for t in range(n))
                rec('|phi_l(2l 1_T) - conj(beta_{l,T})|', abs(phi[chi] - np.conj(bT1[(l, T)])))

# equality cases of the cost inequality: (a) b = 2, no DC  -> equal; (b) single frequency, no DC, any b -> equal;
# (c) DC present -> strict;  (d) b > 2 with two live frequencies -> strict
p, n, m = 7, 2, 3
U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
U0 = U - proj(U, 0, p); W0 = W - proj(W, 0, p)
Us, Ws = split(U0, W0)
print('equality (a) b=2 no DC:   cost - cost_split =', cost(U0, W0, 2.0) - cost(Us, Ws, 2.0))
print('strict   (d) b=3 no DC:   cost - cost_split =', cost(U0, W0, 3.0) - cost(Us, Ws, 3.0))
U1 = proj(U, 2, p); W1 = proj(W, 2, p); Us, Ws = split(U1, W1)
print('equality (b) one freq b=3: cost - cost_split =', cost(U1, W1, 3.0) - cost(Us, Ws, 3.0))
Us, Ws = split(U, W)
print('strict   (c) DC, b=2:      cost - cost_split =', cost(U, W, 2.0) - cost(Us, Ws, 2.0))
print()
for k, v in worst.items():
    print('%-75s %.3e' % (k, v))
