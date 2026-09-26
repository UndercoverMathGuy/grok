"""junk_abelian.py -- (T) on abelian groups with and without an index-2 subgroup: unrestricted normalized-margin ascent vs the
(H)-class ceiling (ELEGANT pricing nu^{1-r} for complex pairs, 1 for real characters), q = 2, nu = 3 (n = 2) and nu = 4 (n = 3).
Odd order (Z_7, Z_9, Z_15, Z_3xZ_3): no real character. Even order (Z_6, Z_8, Z_10, Z_12, Z_2xZ_4): has real characters."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_nu4 import ascent, ceil
torch.set_num_threads(4)
lines = []
def log(s):
    print(s, flush=True); lines.append(s); (HERE / "out" / "junk_abelian.log").write_text("\n".join(lines) + "\n")
for orders in [(7,), (9,), (3, 3), (6,), (8,), (10,), (12,), (2, 4)]:
    G = gc.groups_mod.abelian(orders, "z" + "x".join(map(str, orders)))
    for n in (2, 3):
        if G.N ** n > 2000: continue
        ce = ceil(G, n + 1, 2.0); gs = ce["school"][0]
        vals = [ascent(G, n, 24, 2.0, 5000 if n == 2 else 6000, sd) / gs for sd in range(4)]
        log(f"Z{orders} nu {n + 1}: (H)-ceiling {gs:.5e} on {ce['school'][1]}; ascent gamma/ceiling best {max(vals):.5f} (seeds {', '.join(f'{v:.4f}' for v in vals)}); G1/ceiling {ce['unres'][0] / gs:.3f}")
