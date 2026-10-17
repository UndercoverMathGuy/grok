import math, sys, numpy as np, torch
sys.path.insert(0, '.')
from oned import eff
from oned2 import run, angmargin
from scipy.optimize import brentq
torch.set_default_dtype(torch.float64)

def lamstar(t, p):
    k = np.arange(p)
    f = lambda l: (k * np.exp(2*l*k - np.max(2*l*k))).sum() / np.exp(2*l*k - np.max(2*l*k)).sum() - t
    return brentq(f, -60, 60)

def init3(m, p, n, edge=1.2, tclip=0.04, dph=None):
    L = (p - 1) * n
    cr = np.linspace(edge, L - edge, m - 1)
    ctr = np.concatenate([[0.0], (cr[:-1] + cr[1:]) / 2, [float(L)]])
    lam = np.array([lamstar(min(max(c / n, tclip), p - 1 - tclip), p) for c in ctr])
    lv = np.zeros(m)
    for i in range(m - 1):
        lv[i + 1] = lv[i] - (lam[i + 1] - lam[i]) * cr[i]
    av = -math.pi * ctr / p
    return np.concatenate([lam, lv, av])

if __name__ == "__main__":
    p, n = int(sys.argv[1]), int(sys.argv[2])
    ms = [int(v) for v in sys.argv[3].split(",")]
    for m in ms:
        best = -1e9
        for edge in [0.6, 1.2, 2.0]:
            for tclip in [0.02, 0.05, 0.1]:
                x0 = init3(m, p, n, edge, tclip)
                e0 = eff(torch.tensor(x0), m, p, n).min().item()
                a, x1 = run(angmargin, x0, m, p, n, (0.03, 0.01, 0.003), 1500)
                r, x2 = run(lambda P, m, p, n: eff(P, m, p, n), x1 if a > 0 else x0, m, p, n, (3e-3, 1e-3, 3e-4, 1e-4), 3000)
                if r > best: best, bx = r, x2
                print("   m=%d edge=%.1f tclip=%.2f  init eff %.3g  ang %.3f  -> eff %.5f" % (m, edge, tclip, e0, a, r), flush=True)
        print("p=%d n=%d m=%d nu=%d  BEST %.5f" % (p, n, m, n + 1, best), flush=True)
        np.save("o3_p%d_n%d_m%d.npy" % (p, n, m), bx)
