# Sanity check of prog_bb (exact B&B programme) against vertex enumeration (small p) and stored supports (p=31..43, b=2).
import time, json, numpy as np
import hlib as H, hlib2 as H2
rng = np.random.default_rng(5)
for p in [7, 11, 13]:
    F = (p - 1) // 2
    for s in [1.0, 1.2, 1.5]:
        for trial in range(4):
            S = None if trial == 0 else sorted(rng.choice(np.arange(1, F + 1), size=int(rng.integers(2, F + 1)), replace=False).tolist())
            Ge, ye = H.programme_exact(p, s, S)
            t = time.time(); R = H2.prog_bb(p, s, S)
            print(f"p={p} s={s} S={S}: exact-vertex={Ge:.12f} bb={R['G']:.12f} rel diff={abs(R['G']-Ge)/Ge:.1e} nodes={R['nodes']} gap={R['gap']} ({time.time()-t:.2f}s)", flush=True)
sup = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/supports.json'))
for p in [31, 37, 41, 43]:
    t = time.time(); R = H2.prog_bb(p, 1.5)
    print(f"p={p} b=2: bb={R['G']:.12f} supp={R['supp']} | stored {sup[str(p)]['Gamma']:.12f} {sup[str(p)]['S']} nodes={R['nodes']} gap={R['gap']} ({time.time()-t:.1f}s)", flush=True)
for p in [23, 29, 31]:
    t = time.time(); R = H2.prog_bb(p, 1.2)
    G6, _, _ = H.programme(p, 1.2, restarts=600)
    print(f"p={p} b=2.5: bb={R['G']:.12f} K*={len(R['supp'])} supp={R['supp']} | rwLP600={G6:.12f} nodes={R['nodes']} gap={R['gap']} ({time.time()-t:.1f}s)", flush=True)
