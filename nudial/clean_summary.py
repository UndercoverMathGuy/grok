"""Clean cohort 2026-09-09: measured count (interface ablation) vs the formula's K*.

Reads notes/ablation_count_clean_{mlp,tf,s5}.json and prints one row per
sweep cell: n, grokked, K_abl histogram (grokked runs), unfolded raw count,
legacy K_thr10, clean fraction, predicted K* by degree.

Usage: uv run python nudial/clean_summary.py
"""
import collections, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KSTAR = {2: 12, 3: 6, 4: 4, 5: 4, 6: 3, 7: 3, 8: 2, 9: 2, 10: 2}      # p=113 equal-share ladder (kstar_table, 09-09)


def hist(rs, f):
    return dict(sorted(collections.Counter(f(r) for r in rs).items(), key=lambda kv: str(kv[0])))


def row(label, c, nu, extra=""):
    g = [r for r in c if r["acc"] >= 0.99]
    ks = KSTAR.get(nu, "?")
    if not g:
        print(f"{label:38s} {len(c):3d} {0:4d} | not grokked (acc {[round(r['acc'], 2) for r in c]})  | K*={ks}")
        return
    med = st.median(r["K_abl"] for r in g)
    print(f"{label:38s} {len(c):3d} {len(g):4d} | K_abl {str(hist(g, lambda r: r['K_abl'])):20s} med {med:<4g} | raw {str(hist(g, lambda r: r['K_abl_raw'])):24s} "
          f"| thr10 {str(hist(g, lambda r: r['legacy']['K_thr10'])):16s} | clean {st.median(r['legacy']['clean_frac'] for r in g):.3f} "
          f"| nu={nu} K*={ks} {extra}")


def main():
    hdr = f"{'cell':38s} {'n':>3s} {'grok':>4s} | measured count (ablation, grokked runs) | unfolded | legacy | clean frac | prediction"
    # ---- MLP
    f = ROOT / "notes" / "ablation_count_clean_mlp.json"
    if f.exists():
        rs = json.load(open(f))
        print("== MLP p=113 (10k epochs, ckpt every 1000) ==\n" + hdr)
        cells = sorted(set(r["cell"] for r in rs))
        for cell in cells:
            c = [r for r in rs if r["cell"] == cell]
            name = Path(cell).name
            fam_nu = name.split("_")[0]; nu = int(fam_nu[-1])
            for wd in sorted(set(r["name"].split("/")[1].split("_")[0] for r in c)):
                cc = [r for r in c if r["name"].split("/")[1].split("_")[0] == wd]
                row(f"{name} {wd}", cc, nu)
    # ---- transformer
    f = ROOT / "notes" / "ablation_count_clean_tf.json"
    if f.exists():
        rs = json.load(open(f))
        print("\n== transformer p=113 (25k epochs, ckpt every 2500) ==\n" + hdr)
        cellof = lambda r: r["name"].split("/")[2].split("@")[0]
        for cell in sorted(set(cellof(r) for r in rs)):
            c = [r for r in rs if cellof(r) == cell]
            nu = 6 + (1 if "L10" in cell else 2 if "L11" in cell else 4 if "L22" in cell else 0)
            row(cell, c, nu, "(transformer nu = 6 measured + linear layers)")
    # ---- S5
    f = ROOT / "notes" / "ablation_count_clean_s5.json"
    if f.exists():
        rs = [r for r in json.load(open(f)) if "counted_names" in r]
        print("\n== S5 (50k epochs) ==  prediction: 1 faithful irrep (+ sign patch)")
        for r in rs:
            print(f"  {r['name'].split('/')[1]:22s} seed {r['name'].split('/')[2][:9]:9s} acc {r['acc']:.4f} m_min {r['m_min']:6.2f} | counted {r['counted_names']} | "
                  f"profile {[(n, round(d, 2)) for n, d in r['profile_names'][:4]]}")
        g = [r for r in rs if r["acc"] >= 0.99]
        print(f"  grokked {len(g)}/{len(rs)}; faithful irreps per run {hist(g, lambda r: len([n for n in r['counted_names'] if n != 'sign']))}; "
              f"sign counted in {sum('sign' in r['counted_names'] for r in g)}/{len(g)}")


if __name__ == "__main__":
    main()
