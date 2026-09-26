"""a5_escape.py -- A5, n = 2 (nu 3), q = 2: ascent started at the schoolbook symmetric optimum {3a, 3b} (+ 16 small neurons, noise)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_ascent import ascent, ceilings
from construct_verify import build_net
G = gc.get("a5"); ce = ceilings(G, 3, 2.0); gs = ce["school"][0]
E0, W0 = build_net(G, 2, ce["school"][1]); rng = np.random.default_rng(0); out = []
for eps in (0.0, 0.02, 0.1):
    pad = 16; sc = np.abs(E0).max()
    E = np.concatenate([E0, rng.normal(size=(2, pad, G.N)) * 0.05 * sc], 1); W = np.concatenate([W0, rng.normal(size=(pad, G.N)) * 0.05 * sc], 0)
    E = E + eps * sc * rng.normal(size=E.shape); W = W + eps * sc * rng.normal(size=W.shape)
    val, _ = ascent(G, E.shape[1], 2.0, 4000, 0, init=(E, W), lr=0.002)
    out.append(f"A5 nu 3 q 2: symmetric (schoolbook) optimum {ce['school'][1]} + noise {eps}: best gamma/ceiling after ascent {val / gs:.5f}"); print(out[-1], flush=True)
(HERE / "out" / "a5_escape.log").write_text("\n".join(out) + "\n")
