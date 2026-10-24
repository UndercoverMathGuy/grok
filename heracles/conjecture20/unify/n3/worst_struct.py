import numpy as np
from struct_relax2 import minMr_phase, Hmat
from kbound import G
for s in [1.8552, 2.0]:
    r = 1/s
    for (a, m) in [(0.95, 0.48), (0.9757, 0.2747), (0.99, 0.15), (0.97, 0.35), (0.93, 0.6)]:
        g = G(a, m)
        v, ph = minMr_phase(m, g, r)
        H = Hmat(ph, m, g)
        nu = np.linalg.eigvalsh(H)
        print(f"s={s} a={a} m={m} g={g:.4f} minMr={v:.5f} ratio={(a-v)/m:.4f} phases={np.round(ph,4)} spec={np.round(nu,4)}")
