"""anat_trained.py -- block anatomy (clean amplitude, same-irrep junk fraction) of the best GX-trained nets that beat the
symmetric-class ceiling (A5 fixed-support 3a+3b, S4 fixed-support std+sign)."""
import sys, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_s5 import anatomy
from verify_fourier import irreps_of
from trained_check import gamma_of
from safetensors.numpy import load_file
GX = HERE.parent / "FINAL_RUNS" / "GX"
for gname, sup, ceil in [("a5", "sup-3a-3b", 2.71114340e-05), ("s4", "sup-std-sign", 1.64318231e-04), ("s4", "sup-2-std-sign", 1.64318231e-04)]:
    G = gc.get(gname); rho = irreps_of(G)
    best = None
    for f in glob.glob(str(GX / f"{gname}_bil_frac1_*_{sup}" / "seed*" / "final.safetensors")):
        st = load_file(f); Ea, Eb, W = [st[k].astype(np.float64) for k in ("Ea", "Eb", "W")]
        g = gamma_of(G, Ea, Eb, W)
        if best is None or g > best[0]: best = (g, Ea, Eb, W, f)
    g, Ea, Eb, W, f = best
    an = anatomy(G, np.stack([Ea, Eb]), W.T, rho); Amax = max(v["A"] for v in an.values())
    print(f"{gname} {sup}: gamma {g:.4e} = {g / ceil:.4f} x symmetric ceiling; " + "  ".join(f"{k}: A {v['A']/Amax:+.3f} junk {v['junk']:.3f} mass {v['mass_share']:.3f}" for k, v in an.items()))
