# Referee: independent re-verification of R10 (Legendre counterexample to support-only detection).
# Build phi directly as sum_l v_l cos(2 pi l w/p + th_l) from the Gauss-sum formula, compute D_1 exactly over Z_p.
import numpy as np, mpmath as mp
def legendre(a, p):
    a = int(a) % p
    if a == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1
c = 1-np.sqrt(2/3); tp = np.sqrt(1.5)
for p in [13, 29, 61, 101, 401, 1009, 4001]:
    if p % 4 != 1: continue
    w = np.arange(p); chi = np.array([legendre(x, p) for x in w], float)
    # Fourier: chihat(l) = (1/p) sum_x chi(x) e^{-2 pi i l x/p}
    chihat = np.fft.fft(chi)/p
    F = (p-1)//2; l = np.arange(1, F+1)
    phihat = chihat[l]/(1-np.exp(-2j*np.pi*l/p))
    # real phi(w) = sum_{l in F} 2 Re(phihat(l) e^{2 pi i l w/p}) = sum v_l cos(2 pi l w/p + th_l)
    v = 2*np.abs(phihat); th = np.angle(phihat)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])).sum(0)
    D1 = phi - np.roll(phi, 1)
    err = np.abs(D1 - chi).max()
    kap1 = 1-np.cos(2*np.pi*l/p)
    Kv = (v*kap1).sum(); det = 0.5*(D1.max()-D1.min())
    vpred = 1/(np.sqrt(p)*np.abs(np.sin(np.pi*l/p)))
    # efficiency of a = v/tau'
    K = 1-np.cos(2*np.pi*np.outer(l, l)/p)
    a = v/tp; r = 2/3; s = 1.5
    m = (K@a).min(); na = (a**r).sum()**s
    print(f"p={p}: max|D1-chi|={err:.1e}, max|v - 1/(sqrt p |sin|)|={np.abs(v-vpred).max():.1e}, (1/2)osc={det:.4f}, "
          f"(Kv)_1={Kv:.4f} (2 sqrt p/pi={2*np.sqrt(p)/np.pi:.4f}), det/Kv={det/Kv:.5f} vs pi/(2 sqrt p)={np.pi/(2*np.sqrt(p)):.5f}, "
          f"c*Kv/det={c*Kv/det:.3f}, m(a)/||a||={m/na:.5f}")
