import time
import numpy as np
from socp72 import *
s = 1.5; r = 1/s
t0 = time.time(); bf = BoundFam(s); t1 = time.time()
A2 = 2 * np.cos(np.pi / 7)
xi = np.exp(1j * np.array([[0.3, 1.1]]))
a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
F = ((3 ** (1 - s)) ** r + A2 ** r) ** s
for k in range(5):
    t2 = time.time(); v, _ = bf([a1, A2], [bt1, [0, 0, 0]], F); t3 = time.time()
    print(f"build {t1-t0:.2f}s  call {t3-t2:.3f}s  solver {bf.prob.solver_stats.solve_time:.3f}s  val {v:.5f}")
