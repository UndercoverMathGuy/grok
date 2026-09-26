"""rev_check.py -- is the same-irrep junk of the trained A5 / S4 nets the invariant of the REVERSED task c = b a?
X = sum_i E1_i x E2_i x W_i in the (rho, rho, rho) block; T: <T, E1 x E2 x W> = tr(W E1^T E2^T) (task c = a b),
R: <R, E1 x E2 x W> = tr(W E2^T E1^T) (task c = b a). Fit X ~ alpha T + beta R and report the residual."""
import sys, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from verify_fourier import irreps_of
from coset_trained import best_net
def chain(d, rev):
    T = np.zeros((d,) * 6)
    for k0 in range(d):
        for k1 in range(d):
            for k2 in range(d):
                if not rev: T[k2, k1, k0, k2, k0, k1] = 1      # W[k0,k1] E1[k2,k1] E2[k0,k2]
                else:       T[k0, k2, k2, k1, k0, k1] = 1      # W[k0,k1] E2[k2,k1] E1[k0,k2]
    return T
for gname, pat, irs in [("a5", "a5_bil_frac1_*_sup-3a-3b", ["3a", "3b"]), ("s4", "s4_bil_frac1_*_sup-std-sign", ["std"]),
                        ("s4", "s4_bil_frac1_*_sup-2-std-sign", ["std", "2"]), ("s5", "s5_bil_frac1_*_wdw20000_sup-std-sign", ["std"])]:
    G = gc.get(gname); rho = irreps_of(G)
    g, Ea, Eb, W = best_net(gname, pat)
    for nm in irs:
        R_ = rho[nm]; d = R_.shape[1]
        A = np.einsum("ix,xab->iba", Ea, R_); B = np.einsum("ix,xab->iba", Eb, R_); C = np.einsum("ix,xab->iba", W, R_)
        X = np.einsum("iab,icd,ief->abcdef", A, B, C)
        T = chain(d, False); Rv = chain(d, True)
        M = np.array([[ (T*T).sum(), (T*Rv).sum()], [(T*Rv).sum(), (Rv*Rv).sum()]]); rhs = np.array([(X*T).sum(), (X*Rv).sum()])
        al, be = np.linalg.solve(M, rhs); res = X - al * T - be * Rv
        print(f"{gname} {pat.split('_')[-1]} {nm} (d {d}): X = {al:.4g} T + {be:.4g} R + residual; |residual|/|X| = {np.linalg.norm(res)/np.linalg.norm(X):.4f}; "
              f"beta/alpha = {be/al:.4f}; <T,R>/|T|^2 = {(T*Rv).sum()/(T*T).sum():.4f}")
