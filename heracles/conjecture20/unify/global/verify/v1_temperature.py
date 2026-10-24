# R1 (temperature identity) independent check, plus convention sanity checks of vnet.
#  For a FIXED network: d/ds log gamma(theta;s) = -H(w), d2/ds2 = -Var_w(log w)/s, w_i = ||omega_i||^b / sum.
#  We use mpmath (50 digits) for the cost part (h does not depend on s) and central differences with step 1e-12.
import math, numpy as np, mpmath as mp
from vnet import *

mp.mp.dps = 50
rng = np.random.default_rng(20261018)

# --- convention checks: IF block gives E Phi_1 = conj(beta_1), Phi_1 constant, gap = kappa*Re beta
p, n = 7, 2
U, W = if_block(p, n, 1, 1.3, 0.2)
f = table(U, W)
b1 = clean(f)[0]
P1 = Phi(f, 1)
print("conv: E Phi_1 vs conj(beta_1):", np.round(P1.mean(), 12), np.round(np.conj(b1), 12), " Phi_1 spread:", np.ptp(P1.real) + np.ptp(P1.imag))
G = gaps(f)
print("conv: IF gaps / (kappa Re beta):", [round(G[..., d - 1].min() / (kappa(p, 1, d) * b1.real), 10) for d in range(1, p)])
at = atoms(U, W, 1)
print("conv: sum atoms = beta_1:", np.allclose(at.sum(), b1))
# single-neuron budget equality for IF neurons: |beta_i1| = B_p ||omega_i||^nu
print("conv: |beta_i| / (B_p ||omega||^nu):", np.round(np.abs(at) / (B_p(p, n) * neuron_sq(U, W) ** ((n + 1) / 2)), 10))
# gap identity g = Re(Phi_1 (1 - rho^{-d})) for a single feature with arbitrary inputs, output at freq 1 (+DC)
U = rng.normal(size=(4, 2, p)); W = rng.normal(size=(4, 3)) @ np.vstack([freq_basis(p, 0), freq_basis(p, 1)])
f = table(U, W); G = gaps(f); P1 = Phi(f, 1); rho = np.exp(2j * np.pi / p)
print("conv: single-feature gap identity max err:", max(np.abs(G[..., d - 1] - (P1 * (1 - rho ** (-d))).real).max() for d in range(1, p)))

# --- R1
def logs(h, S, s, nu):
    S = [mp.mpf(float(v)) for v in S]
    b = mp.mpf(nu) / s
    return mp.log(h) - s * mp.log(mp.fsum([v ** (b / 2) for v in S]))

worst1 = 0; worst2 = 0
for trial in range(40):
    p = int(rng.choice([3, 5, 7])); n = int(rng.choice([2, 3])); m = int(rng.integers(1, 7)); nu = n + 1
    U = rng.normal(size=(m, n, p)) * rng.uniform(0.1, 3, size=(m, 1, 1)); W = rng.normal(size=(m, p))
    h = margin(table(U, W))
    if h <= 0:
        W = -W; h = margin(table(U, W))
    if h <= 0:
        # add an IF block to force positive margin
        U0, W0 = if_block(p, n, 1, 2.0); U = np.concatenate([U0, 0.2 * U]); W = np.concatenate([W0, 0.2 * W])
        h = margin(table(U, W))
    assert h > 0
    S = neuron_sq(U, W); h = mp.mpf(float(h))
    for s in [mp.mpf('1.0'), mp.mpf('1.37'), mp.mpf(nu) / 2]:
        e = mp.mpf('1e-12')
        f0, fp, fm = logs(h, S, s, nu), logs(h, S, s + e, nu), logs(h, S, s - e, nu)
        d1 = (fp - fm) / (2 * e); d2 = (fp - 2 * f0 + fm) / e ** 2
        b = nu / s
        w = [mp.mpf(float(v)) ** (b / 2) for v in S]; Z = mp.fsum(w); w = [x / Z for x in w]
        H = -mp.fsum([x * mp.log(x) for x in w]); V = mp.fsum([x * mp.log(x) ** 2 for x in w]) - H ** 2
        worst1 = max(worst1, abs(d1 + H)); worst2 = max(worst2, abs(d2 + V / s))
print(f"R1: max |dlog/ds + H| = {mp.nstr(worst1, 5)},  max |d2 + V/s| = {mp.nstr(worst2, 5)}  (40 random nets, p in 3,5,7, n in 2,3)")
