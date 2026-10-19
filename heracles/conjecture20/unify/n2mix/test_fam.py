import sys, time
import numpy as np
from socp72 import *
s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
r = 1 / s
print("paid families:", len(FAM), " members:", sum(len(v) for v in FAM.values()))
bd = Bound(s); bf = BoundFam(s)
A2 = 2 * np.cos(np.pi / 7)
for ph in [(0.3, 1.1), (0.0, 0.0), (np.pi/2, 0.7), (2.0, -1.0)]:
    xi = np.exp(1j * np.array([ph]))
    a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
    F = ((3 ** (1 - s)) ** r + A2 ** r) ** s
    t0 = time.time()
    v1, _ = bd([a1, A2], [bt1, [0, 0, 0]], F)
    v2, _ = bf([a1, A2], [bt1, [0, 0, 0]], F)
    print(f"phases {ph}: l2 ledger {v1:.5f}   family ledger {v2:.5f}   [{time.time()-t0:.2f}s]")
