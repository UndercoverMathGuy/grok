import numpy as np
from kbound import Fr, Pmax
for s in [1.8552, 1.9, 2.0]:
    r = 1/s
    A = np.linspace(0.3, 1, 351); Ms = np.linspace(0.005, 1, 400)
    R = np.array([[(a - Fr(Pmax(a, m), r))/m for m in Ms] for a in A])
    for m0 in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        sel = Ms >= m0
        sub = R[:, sel]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        print(f"s={s} m>={m0}: sup ratio {sub.max():.4f} at a={A[i]:.3f} m={Ms[sel][j]:.3f}")
    # where is bonus bound positive at small m
    neg = [(a, m) for ii, a in enumerate(A) for jj, m in enumerate(Ms) if m < 0.05 and R[ii, jj] > 0.9]
    print("  small-m bad points (ratio>0.9): a range", (min(p[0] for p in neg), max(p[0] for p in neg)) if neg else None)
