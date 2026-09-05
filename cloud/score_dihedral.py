"""Score the dihedral sweep against notes/prereg_dihedral.md (D1-D5).

    python cloud/score_dihedral.py --root runs_dihedral [--out notes/dihedral_scored.json]

spectra.npz coeffs = sqrt(irrep energy) over the nontrivial irreps in the
order [sign, rho_1, ..., rho_(n-1)/2] (trivial dropped). Amplitude
A_rho = sqrt(E_rho) * d_rho (S5 note convention). Per run: grokked (final
test acc >= 0.99), committee (gap detector, transformer instrument),
K_gap, K_thr (irreps with A >= 0.1 max A), sign share = A_sign / max A_rho,
leader rho index. Per cell: K hist, sign-share median, K_thr of memorised
runs, mean pairwise Jaccard of grokked committees.
"""
import argparse, glob, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import committee_from_coeffs


def read_run(rd):
    z = np.load(rd / "spectra.npz")
    cfg = json.loads((rd / "config.json").read_text())
    n = int(cfg["fn_name"][1:])
    E = z["energy"][-1].astype(float)                    # [sign, rho_1..]
    dims = np.array([1] + [2] * ((n - 1) // 2))
    A = np.sqrt(E) * dims
    Arho = A[1:]
    comm_rho = committee_from_coeffs(np.sqrt(E[1:]))     # gap detector over the rho's only (1-indexed j)
    return dict(run=rd.name, cell=rd.parent.name, acc=float(z["test_acc"][-1]),
                train_acc=float(z["train_acc"][-1]),
                committee=[int(j) for j in comm_rho], K_gap=len(comm_rho),
                K_thr=int((A >= 0.1 * A.max()).sum()),
                K_thr_rho=int((Arho >= 0.1 * Arho.max()).sum()),
                sign_share=float(A[0] / Arho.max()), leader=int(Arho.argmax() + 1),
                top6=[int(j + 1) for j in np.argsort(-Arho)[:6]],
                top6_rel=[float(x) for x in np.sort(Arho)[::-1][:6] / Arho.max()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_dihedral")
    ap.add_argument("--out", default="notes/dihedral_scored.json")
    args = ap.parse_args()
    runs = [read_run(Path(f).parent) for f in sorted(glob.glob(f"{args.root}/*/*/spectra.npz"))]
    cells = {}
    for r in runs: cells.setdefault(r["cell"], []).append(r)
    summ = {}
    for cell, rs in sorted(cells.items()):
        g = [r for r in rs if r["acc"] >= 0.99]; m = [r for r in rs if r["acc"] < 0.99]
        ks = [r["K_gap"] for r in g]
        jac = [len(set(a["committee"]) & set(b["committee"])) / len(set(a["committee"]) | set(b["committee"]))
               for a, b in combinations(g, 2)] if len(g) > 1 else []
        s = dict(n=len(rs), n_grok=len(g), K_hist={k: ks.count(k) for k in sorted(set(ks))},
                 K_thr_grok=[r["K_thr"] for r in g], K_thr_mem=[r["K_thr"] for r in m],
                 sign_share_grok=[round(r["sign_share"], 3) for r in g],
                 sign_share_median=float(np.median([r["sign_share"] for r in g])) if g else None,
                 leaders=[r["leader"] for r in g], mean_jaccard=float(np.mean(jac)) if jac else None,
                 top6_rel_median=np.median([r["top6_rel"] for r in g], 0).round(3).tolist() if g else None)
        summ[cell] = s
        print(f"{cell}: n={s['n']} grok={s['n_grok']} K_gap hist {s['K_hist']} K_thr grok {s['K_thr_grok']} mem {s['K_thr_mem']}")
        print(f"   sign share (grok) {s['sign_share_grok']} median {s['sign_share_median']} | leaders {s['leaders']} | Jaccard {s['mean_jaccard']} | top6 rel {s['top6_rel_median']}")
    g_all = [r for r in runs if r["acc"] >= 0.99]
    if g_all:
        ks = [r["K_gap"] for r in g_all]; mode = max(set(ks), key=ks.count)
        in345 = np.mean([k in (3, 4, 5) for k in ks])
        d1 = "PASS" if (2 <= mode <= 6 and in345 >= 0.7) else "FAIL"
        d2 = "PASS" if np.mean([r["sign_share"] < 0.05 for r in g_all]) >= 0.8 else "FAIL"
        d3 = "Morwani-dense KILLED" if all(r["K_thr"] <= 8 for r in g_all) else "Morwani-dense NOT killed"
        print(f"\nD1 mode {mode}, frac in {{3,4,5}} {in345:.2f}: {d1} | D2 sign<0.05 frac {np.mean([r['sign_share'] < 0.05 for r in g_all]):.2f}: {d2} | D3 {d3} | grokked {len(g_all)}/{len(runs)}")
    Path(args.out).write_text(json.dumps(dict(cells=summ, runs=runs), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
