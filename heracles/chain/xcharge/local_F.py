# Is the block orbit a strict local maximum of the X-FREE functional F = B_XB / (gamma_sym P^{3/2}) over ALL networks?
# (T <= F always, so if F < 1 on a punctured neighbourhood, every network near the block is in case A: no Hessian of T needed.)
# Directions: generic Gaussian on block neurons (junk + mixing + DC), phase-only, extra small neurons (mixing), shrinking
# one neuron (junk), pure DC.
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
from xblind_bound import bxb
p = 5; Gam, ystar = programme(p, 1.5); rng = np.random.default_rng(0)
B = np.array([[e1, e2, w] for e1, e2, w in block_net(p, ystar)])           # [6, 3, p]
def net_of(A): return [tuple(A[i]) for i in range(A.shape[0])]
kinds = {}
def add(kind, A): kinds.setdefault(kind, []).append(A)
for t in range(40):
    add('generic', rng.normal(size=B.shape))
    ph = np.zeros_like(B)
    for i in range(6):
        for j in range(3):
            v = B[i, j]; Fv = np.fft.fft(v); Fv *= np.exp(1j * rng.normal() * (np.arange(p) != 0) * np.sign(np.arange(p) - p / 2))
            ph[i, j] = np.real(np.fft.ifft(Fv)) - v
    add('phase', ph / (np.linalg.norm(ph) + 1e-12))
    add('extra', np.concatenate([np.zeros_like(B), rng.normal(size=(3, 3, p))], 0))
    sh = np.zeros_like(B); sh[rng.integers(6)] = -B[rng.integers(6)] * 0; i = rng.integers(6); sh[i] = -B[i]
    add('shrink', sh)
    dc = np.zeros_like(B); dc[:, :2, :] = rng.normal(size=(6, 2, 1)); add('dc', dc)
print("eps     " + "  ".join(f"{k:>18s}" for k in kinds))
for eps in (1e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3):
    row = []
    for k, dirs in kinds.items():
        worst = -9
        for d in dirs:
            A = (np.concatenate([B, np.zeros((d.shape[0] - 6, 3, p))], 0) if d.shape[0] > 6 else B.copy()) + eps * d * np.abs(B).max()
            f, tt = bxb(p, net_of(A)); worst = max(worst, f - 1)
        row.append(worst)
    print(f"{eps:<7g} " + "  ".join(f"{w:+18.2e}" for w in row), flush=True)
