"""trained_check.py -- the GX bil nets (gated chain, n = 2, q = 2, trained by GD + weight decay) against Theorems G1, G1', G2.

For every full-data run (free and fixed-support) of S5, S4, A5, Z_120, Z_11^2: gamma_2 (full-table min margin / ||theta||^3,
recomputed here), the own-share bound G1' (network's own irrep mass shares), the unrestricted bound G1, and the symmetric-class
ceiling G2 (schoolbook kappa; ELEGANT kappa for complex pairs). G1 and G1' must never be exceeded.
"""
import glob, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402
from prog_large import weights, polish, support_search  # noqa: E402

GX = HERE.parent / "FINAL_RUNS" / "GX"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def gamma_of(G, Ea, Eb, W):
    N = G.N
    h = (Ea[:, :, None] * Eb[:, None, :]).reshape(Ea.shape[0], -1)
    Z = h.T @ W.T                                          # rows a*N + b, classes
    y = G.MULT.reshape(-1); zz = Z.copy(); zy = zz[np.arange(len(y)), y]; zz[np.arange(len(y)), y] = -np.inf
    m = float((zy - zz.max(1)).min()); P2 = float((Ea ** 2).sum() + (Eb ** 2).sum() + (W ** 2).sum())
    return m / P2 ** 1.5


def main():
    from safetensors.numpy import load_file
    out = {}
    for gname in ["s5", "s4", "a5", "z120", "z11x11"]:
        G = gc.get(gname); irr, classes = gc.programme_data(G); nu, q, s = 3, 2.0, 1.5
        cov = np.array([p["cov"] for p in irr])
        w_un = weights(irr, nu, q, "real_unres"); w_H = weights(irr, nu, q, "real_H")
        # global ceilings
        if len(irr) <= 12:
            verts = gc.vertices(cov, exact=False)
            cU = w_un ** (-1 / s); cH = w_H ** (-1 / s)
            GU = gc.solve(verts, cU, q / nu)[0]; GH = gc.solve(verts, cH, q / nu)[0]
        else:
            pj = json.loads((HERE / "out" / "prog_large.json").read_text())
            GU = pj[f"{gname}_nu3_q2_real_unres"]["Gamma"]; GH = pj[f"{gname}_nu3_q2_real_H"]["Gamma"]
        gU = (G.N * nu) ** (-nu / 2) * GU; gH = (G.N * nu) ** (-nu / 2) * GH
        rr = [p["r"] for p in irr]
        cells = sorted(glob.glob(str(GX / f"{gname}_bil_frac1_*")))
        log(f"== {gname}: G1 bound {gU:.4e}; G2 ceiling (schoolbook / ELEGANT) {gH:.4e}")
        rows = []
        for cell in cells:
            for sd in sorted(glob.glob(cell + "/seed*/final.safetensors")):
                st = load_file(sd); Ea, Eb, W = [st[k].astype(np.float64) for k in ("Ea", "Eb", "W")]
                gam = gamma_of(G, Ea, Eb, W)
                if gam <= 0:
                    continue
                # own shares: q = 2 -> U_pi = total weight mass on pi (all three weight matrices)
                mass = sum(G.irrep_energy(T).sum(0) for T in (Ea, Eb, W.T))
                U = np.array([mass[r] for r in rr]); beta = U / U.sum()
                V = ((beta ** s * w_un)[:, None] * cov).sum(0).min()
                bown = (G.N * nu) ** (-nu / 2) * V
                tag = Path(cell).name.split("wdw20000")[-1] or "free"
                rows.append(dict(cell=tag, seed=Path(sd).parent.name, gamma=gam, own=bown, G1=gU, G2=gH))
        for tag in sorted(set(r["cell"] for r in rows)):
            g = [r for r in rows if r["cell"] == tag]
            mx = max(g, key=lambda r: r["gamma"])
            log(f"   {tag:40s} n {len(g):2d}  gamma max {mx['gamma']:.3e}  max gamma/G1' {max(r['gamma'] / r['own'] for r in g):.4f}  "
                f"gamma/G1 {mx['gamma'] / gU:.4f}  gamma/G2 {mx['gamma'] / gH:.4f}")
        viol = [r for r in rows if r["gamma"] > r["own"] * (1 + 1e-9) or r["gamma"] > r["G1"] * (1 + 1e-9)]
        log(f"   {len(rows)} nets checked; violations of G1/G1': {len(viol)}")
        out[gname] = rows
    (HERE / "out" / "trained_check.json").write_text(json.dumps(out, indent=1))
    (HERE / "out" / "trained_check.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
