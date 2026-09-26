"""arch_local.py -- second-order test of (PW) at B = I + eps H (the tight point).
Necessary and sufficient (with the probe's 2nd-order term chosen as the multi-affine part), diagonal of H = 0 (worst case):
     max_v [ (Im R)^2 ]_rest  <=  (D-1)/2 * ||H_off||^2 ,     R(v) = sum_{k<k'} H_{kk'} v^{(k,k']} ,
where [.]_rest keeps only the Fourier modes that are NOT of the form v^T or conj(v^T) with T a subset (multi-affine),
and ||H_off||^2 = sum_{k<k'} |H_{kk'}|^2.  We maximise the ratio over H_off (adversary) and v (grid + polish).
Also reports the variant where the probe's 2nd-order multi-affine part is optimised freely (LP over a grid) -- the
true second-order condition:  exists zero-mean multi-affine real trig poly g with  f0 + f_rest(v) - g(v) >= 0 for all v,
where f0 = tau - ||H_off||^2/D^2, tau = ||H_off||^2/D.
usage: arch_local.py n nstart seed
"""
import numpy as np, sys, json
from scipy.optimize import minimize, linprog
from itertools import combinations, product

n = int(sys.argv[1]); nst = int(sys.argv[2]); seed = int(sys.argv[3])
D = n + 1; rng = np.random.default_rng(seed)
intervals = [(k, kp) for k in range(D) for kp in range(k + 1, D)]           # (k,k'] = slots k+1..k' (0-based slots k..k'-1)
def slots(I): return list(range(I[0], I[1]))
# Fourier modes of (Im R)^2: products of interval characters with signs; classify multi-affine or not
def mode(I, J, sI, sJ):
    e = np.zeros(n, dtype=int)
    for s in slots(I): e[s] += sI
    for s in slots(J): e[s] += sJ
    return tuple(e)
def is_ma(e):  # multi-affine: all entries in {0,1} or all in {0,-1}
    e = np.array(e); return (np.all((e == 0) | (e == 1)) or np.all((e == 0) | (e == -1)))

G = {1: 4000, 2: 200, 3: 36, 4: 14}[n]
th = 2 * np.pi * np.arange(G) / G
grid = np.stack(np.meshgrid(*([th] * n), indexing='ij'), -1).reshape(-1, n)

def rest_poly_coeffs(H):
    """(Im R)^2 = (|R|^2 - Re R^2)/2 ; return dict mode->complex coeff of the REST part (as an expansion sum c_e v^e)."""
    co = {}
    def add(e, c):
        co[e] = co.get(e, 0) + c
    for I in intervals:
        for J in intervals:
            hI, hJ = H[I], H[J]
            # |R|^2 = sum hI conj(hJ) v^I conj(v^J)
            add(mode(I, J, 1, -1), 0.5 * hI * np.conj(hJ))
            # -Re R^2 /2 = -(1/4)(R^2 + conj R^2)
            add(mode(I, J, 1, 1), -0.25 * hI * hJ)
            add(mode(I, J, -1, -1), -0.25 * np.conj(hI * hJ))
    return {e: c for e, c in co.items() if not is_ma(e) and any(e)}

def eval_poly(co, ang):  # ang (P,n)
    out = np.zeros(len(ang))
    for e, c in co.items():
        out += (c * np.exp(1j * (ang @ np.array(e)))).real
    return out

def ratio(hvec):
    H = {I: hvec[2 * i] + 1j * hvec[2 * i + 1] for i, I in enumerate(intervals)}
    nrm = sum(abs(h) ** 2 for h in H.values())
    if nrm < 1e-12: return 0.0
    co = rest_poly_coeffs(H)
    vals = eval_poly(co, grid); i = int(vals.argmax())
    r = minimize(lambda a: -eval_poly(co, a[None, :])[0], grid[i], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
    return max(vals[i], -r.fun) / nrm

best = 0; bh = None
for s in range(nst):
    h0 = rng.normal(size=2 * len(intervals))
    r = minimize(lambda h: -ratio(h), h0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-8, fatol=1e-12))
    if -r.fun > best: best = -r.fun; bh = r.x
    print(f"  start {s}: max ratio so far {best:.6f}  (threshold (D-1)/2 = {(D-1)/2})", flush=True)
H = {I: bh[2 * i] + 1j * bh[2 * i + 1] for i, I in enumerate(intervals)}
nrm = sum(abs(h) ** 2 for h in H.values())
print(f"n={n}: sup_H max_v [(Im R)^2]_rest / ||H_off||^2 = {best:.6f}   vs   (D-1)/2 = {(D-1)/2:.3f}   ->  {'PASS' if best <= (D-1)/2 + 1e-9 else 'FAIL (naive 2nd-order probe)'}")
print("   worst H_off (normalised):", {str(I): np.round(h / np.sqrt(nrm), 4) for I, h in H.items()})

# true second-order condition with free multi-affine g (LP): min over g of max_v [ -(f0 + f_rest - g) ]
if best > (D - 1) / 2 - 0.05:
    co = rest_poly_coeffs(H); f_rest = -(2 / D ** 2) * eval_poly(co, grid); f0 = nrm / D - nrm / D ** 2
    subsets = [T for k in range(1, n + 1) for T in combinations(range(n), k)]
    cols = []
    for T in subsets:
        ph = grid[:, list(T)].sum(1); cols += [np.cos(ph), np.sin(ph)]
    A = np.stack(cols, 1)   # g = A @ coeffs (zero-mean multi-affine real trig poly)
    # variables: coeffs (2*len(subsets)), t ;  minimise t  s.t.  -(f0 + f_rest - A c) <= t  ->  A c - t <= f0 + f_rest
    nc = A.shape[1]
    cobj = np.zeros(nc + 1); cobj[-1] = 1
    Aub = np.concatenate([A, -np.ones((len(grid), 1))], 1); bub = f0 + f_rest
    r = linprog(cobj, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * (nc + 1), method='highs')
    print(f"   LP with free multi-affine 2nd-order correction at worst H: min_g max_v violation = {r.fun:+.3e}  ({'PASS' if r.fun <= 1e-9 else 'FAIL'})")
json.dump(dict(n=n, best=float(best), threshold=(D - 1) / 2, worstH={str(I): [float(h.real), float(h.imag)] for I, h in H.items()}),
          open(f'FINAL_RUNS/ARCH/local_n{n}_s{seed}.json', 'w'))
