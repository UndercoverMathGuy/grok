"""junk_s5.py -- unrestricted normalized-margin ascent on S5 (and A5), n = 2, q = 2: does junk beat the symmetric-class ceiling,
and which irreps does the ascent optimum use? (anatomy: clean amplitude A = d G, block junk fraction, weight mass share per irrep)"""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_ascent import ascent, ceilings
from construct_verify import build_net
from verify_fourier import irreps_of, fhat
torch.set_num_threads(8)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def anatomy(G, E, W, rho):
    N = G.N; m = W.shape[0]; P2 = (E ** 2).sum() + (W ** 2).sum(); out = {}
    for nm, R in rho.items():
        d = R.shape[1]
        if nm == "trivial": continue
        Es = np.einsum("jix,xab->jiba", E, R); Ws = np.einsum("ix,xab->iba", W, R)     # E^ = sum e(x) rho(x)^T
        Gc = sum(np.trace(Ws[i] @ Es[0, i].T @ Es[1, i].T) for i in range(m))
        mass = (d / N) * (((Es ** 2).sum((0, 2, 3))) + (Ws ** 2).sum((1, 2)))
        # block norm^2 without forming X: ||sum_i A_i x B_i x C_i||^2 = sum_{i,k} <A_i,A_k><B_i,B_k><C_i,C_k>
        GA = np.einsum("iab,kab->ik", Es[0], Es[0]); GB = np.einsum("iab,kab->ik", Es[1], Es[1]); GW = np.einsum("iab,kab->ik", Ws, Ws)
        Xn2 = (GA * GB * GW).sum(); junk = np.sqrt(max(Xn2 - Gc ** 2 / d ** 3, 0) / max(Xn2, 1e-300))
        out[nm] = dict(A=float(d * Gc), junk=float(junk), mass_share=float(mass.sum() / P2))
    return out
def main():
    res = {}
    for gname, runs in [("s5", [(80, 0, "school"), (80, 1, "school")])]:
        G = gc.get(gname); ce = ceilings(G, 3, 2.0); gs = ce["school"][0]; rho = irreps_of(G)
        log(f"== {gname}: gamma_school {gs:.6e} on {ce['school'][1]}; gamma_unres {ce['unres'][0]:.6e}")
        for m, sd, init in runs:
            if init == "school":
                E0, W0 = build_net(G, 2, ce["school"][1]); rng = np.random.default_rng(sd); pad = m - W0.shape[0]
                E = np.concatenate([E0, rng.normal(size=(2, pad, G.N)) * 1e-3], 1); W = np.concatenate([W0, rng.normal(size=(pad, G.N)) * 1e-3], 0)
                sc = np.abs(E0).max(); E = E + (0.02 if sd == 0 else 0.1) * sc * rng.normal(size=E.shape); W = W + 0.02 * sc * rng.normal(size=W.shape)
                val, (E, W) = ascent(G, m, 2.0, 3000, sd, init=(E, W), lr=0.003)
            else:
                val, (E, W) = ascent(G, m, 2.0, 5000, sd)
            an = anatomy(G, E, W, rho); Amax = max(v["A"] for v in an.values())
            log(f"   m {m} seed {sd} init {init or 'random'}: gamma/school {val / gs:.5f}; " + "  ".join(f"{k}: A {v['A'] / Amax:+.3f} junk {v['junk']:.2f} mass {v['mass_share']:.2f}" for k, v in an.items()))
            res[f"{gname}_m{m}_s{sd}_{init}"] = dict(ratio=val / gs, anatomy=an)
            (HERE / "out" / "junk_s5.json").write_text(json.dumps(res, indent=1))
            (HERE / "out" / "junk_s5.log").write_text("\n".join(LOG) + "\n")
if __name__ == "__main__":
    main()
