# Release a 1-D solution into the full slot-symmetric single-feature class (cone.py) and locally optimise.
import sys, math, numpy as np, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'interference_scripts'))
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from cone import Cone, optimise
torch.set_num_threads(1)
p, n, f = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
x = np.load(f); m = len(x) // 3; lam, lv, av = x[:m], x[m:2*m], x[2*m:]
eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)]); e = np.array([2*a if a < p/2 else 2*a - p for a in range(p)], float)
U = np.array([eps * np.exp(l * e) for l in lam]); Z = np.linalg.norm(U, axis=1)
W = np.exp(lv + 1j * av) * Z ** n
cone = Cone(p, n, math.pi / p)
with torch.no_grad():
    r0 = cone.margins(torch.tensor(U / Z[:, None]), torch.tensor(W.real), torch.tensor(W.imag)).min().item()
print("p=%d n=%d m=%d: 1-D eff evaluated in full class %.5f" % (p, n, m, r0), flush=True)
rng = np.random.default_rng(0)
best = r0
for trial in range(4):
    U0 = U / Z[:, None] + (0.0 if trial == 0 else 0.02 * trial) * rng.standard_normal(U.shape)
    r, P = optimise(cone, U0, W.real, W.imag, stages=(r0 * 1e-2, r0 * 3e-3, r0 * 1e-3, r0 * 3e-4), iters=1500)
    best = max(best, r)
    print("   trial %d (noise %.2f): full-class local optimum %.5f" % (trial, 0.02 * trial, r), flush=True)
print("p=%d n=%d: best %.5f (1-D %.5f)" % (p, n, best, r0))
