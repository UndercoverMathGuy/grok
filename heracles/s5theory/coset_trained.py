"""coset_trained.py -- where does the same-irrep junk of the GX-trained bil nets live? For the best free-run S5 net (and the S4
fixed-support std+sign net, and the A5 3a+3b net) form the logit part carried by a set of irreps, subtract its class-function
average, and split the junk norm between sign-consistent triples (sgn(c) = sgn(a) sgn(b)) and inconsistent ones."""
import sys, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from trained_check import gamma_of
from safetensors.numpy import load_file
GX = HERE.parent / "FINAL_RUNS" / "GX"
def best_net(gname, pat):
    best = None
    for f in glob.glob(str(GX / pat / "seed*" / "final.safetensors")):
        st = load_file(f); Ea, Eb, W = [st[k].astype(np.float64) for k in ("Ea", "Eb", "W")]
        g = gamma_of(gc.get(gname), Ea, Eb, W)
        if best is None or g > best[0]: best = (g, Ea, Eb, W.T)
    return best
for gname, pat, sets in [("s5", "s5_bil_frac1_*_wdw20000", [["std", "std_sign"], ["std"], ["std_sign"], ["5a", "5b"], ["6"]]),
                         ("s4", "s4_bil_frac1_*_sup-std-sign", [["std"]]), ("a5", "a5_bil_frac1_*_sup-3a-3b", [["3a"], ["3b"], ["3a", "3b"]])]:
    G = gc.get(gname); N = G.N
    g, Ea, Eb, W = best_net(gname, pat)
    a, b, c = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    if G.sign is not None:
        sg = G.CHAR[G.sign][G.class_of]; consistent = sg[c] == sg[a] * sg[b]
    else:
        consistent = np.ones((N, N, N), bool)
    cls = G.class_of[G.MULT[G.MULT[a, b], G.INV[c]]]
    print(f"== {gname} {pat}: gamma {g:.3e}; consistent fraction {consistent.mean():.2f}")
    for S in sets:
        P = sum(G.PROJ[G.irrep_names.index(x)] for x in S)
        z = np.einsum("ia,ib,ic->abc", Ea @ P, Eb @ P, W @ P)
        clean = np.zeros_like(z)
        for C in range(len(G.class_names)):
            clean[cls == C] = z[cls == C].mean()
        J = z - clean; tot = np.sqrt((z ** 2).sum()) + 1e-300
        print(f"   {'+'.join(S):12s}: clean |.|/|z| {np.sqrt((clean ** 2).sum()) / tot:.3f}; junk on consistent {np.sqrt((J[consistent] ** 2).sum()) / tot:.3f}, "
              f"on inconsistent {np.sqrt((J[~consistent] ** 2).sum()) / tot:.3f}")
