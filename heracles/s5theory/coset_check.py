"""coset_check.py -- is the junk of the D_3 / D_7 ascent optima confined to parity-INCONSISTENT triples (sgn(c) != sgn(a) sgn(b))?
For each 2-dim irrep rho, project every neuron's e, f, w onto the rho-isotypic component, form the rho-part of the logit table,
subtract its level-set (class-function) average, and measure the remaining junk separately on consistent / inconsistent triples."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_ascent import ascent, ceilings
for gname, m in [("d3", 12), ("d7", 32)]:
    G = gc.get(gname); N = G.N; gs = ceilings(G, 3, 2.0)["school"][0]
    val, (E, W) = ascent(G, m, 2.0, 6000, 0)
    sg = G.CHAR[G.irrep_names.index("sign")][G.class_of]
    a, b, c = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    consistent = sg[c] == sg[a] * sg[b]
    D = G.MULT[G.MULT[a, b], G.INV[c]]; cls = G.class_of[D]
    print(f"== {gname}: gamma/school {val / gs:.4f}")
    for r, nm in enumerate(G.irrep_names):
        if nm == "trivial": continue
        P = G.PROJ[r]
        e1 = E[0] @ P.T; e2 = E[1] @ P.T; w = W @ P.T
        z = np.einsum("ia,ib,ic->abc", e1, e2, w)
        clean = np.zeros_like(z)
        for C in range(len(G.class_names)):
            clean[cls == C] = z[cls == C].mean()
        J = z - clean
        tot = np.sqrt((z ** 2).sum()) + 1e-300
        print(f"   {nm:6s}: |junk| / |z_rho| on consistent triples {np.sqrt((J[consistent] ** 2).sum()) / tot:.4f}, "
              f"on inconsistent triples {np.sqrt((J[~consistent] ** 2).sum()) / tot:.4f}")
