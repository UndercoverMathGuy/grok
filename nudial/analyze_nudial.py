"""nu-dial analyzer (runs on the pod or locally; numpy only).

Per run: K_gap (committee_from_coeffs on final coeffs — the transformer
cohorts' instrument), K_eps (#frequencies carrying >= eps of final logit
Fourier energy; MAIS instrument, eps=0.005), k_eff (participation ratio
1/sum(share^2)), final test acc, grok epoch (test acc >= 0.99), final
normalised margin. Per cell: medians/modes. Then the pre-registered tests
(README). Writes notes/nudial_summary.json and prints the table.
"""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs_nudial"))
sys.path.insert(0, str(ROOT / "nudial"))
from train_nudial import committee_from_coeffs  # noqa: E402

EPS = 0.005


def per_run(rd):
    z = np.load(rd / "spectra.npz"); cfg = json.loads((rd / "config.json").read_text())
    co, en = z["coeffs"][-1], z["energy"][-1]
    share = en / en.sum()
    ta = z["test_acc"] if cfg.get("frac_train", 0.5) < 1.0 else z["train_acc"]; ga = np.flatnonzero(ta >= 0.99)
    return dict(
        cohort=rd.parents[1].name, cell=rd.parent.name, run=rd.name, family=cfg["family"],
        nu=cfg["nu"], m=cfg["m"], p=cfg["p"], mode=cfg["mode"], wd=cfg["weight_decay"],
        lam=cfg["lam"], seed=cfg["init_seed"], opt=cfg.get("opt", "adamw"), lr=cfg["lr"], frac=cfg.get("frac_train", 0.5), layout=f"{cfg.get('layout_pre',-1)},{cfg.get('layout_post',-1)}",
        init_scale=cfg.get("init_scale", 1.0),
        K_gap=len(committee_from_coeffs(co)), committee=committee_from_coeffs(co),
        K_eps=int((share >= EPS).sum()), k_eff=float(1.0 / (share ** 2).sum()),
        top4_share=float(np.sort(share)[::-1][:4].sum()),
        test_acc=float(ta[-1]), train_acc=float(z["train_acc"][-1]),
        grok_epoch=int(z["epochs"][ga[0]]) if len(ga) else -1,
        margin_norm=float(z["min_margin_norm"][-1]), pnorm2=float(z["pnorm2"][-1]))


def main():
    rows = [per_run(rd) for rd in sorted(RUNS.glob("*/*/*")) if (rd / "spectra.npz").exists()]
    print(f"{len(rows)} runs")
    cells = {}
    for r in rows:
        cells.setdefault((r["cohort"], r["cell"]), []).append(r)
    summary = {"n_runs": len(rows), "cells": {}, "tests": {}}
    print(f"{'cohort':6} {'cell':28} {'n':>4} {'grok':>5} {'K_gap med':>9} {'mode':>5} "
          f"{'K_eps med':>9} {'k_eff med':>9} {'top4':>6} {'margin':>9}")
    for (coh, cell), rr in sorted(cells.items()):
        ok = [r for r in rr if r["mode"] == "margin" or r["test_acc"] >= 0.99]
        use = ok if ok else rr
        kg = [r["K_gap"] for r in use]
        d = dict(n=len(rr), n_grok=len(ok), K_gap_med=float(np.median(kg)),
                 K_gap_mode=Counter(kg).most_common(1)[0][0], K_gap_hist=dict(Counter(kg)),
                 K_eps_med=float(np.median([r["K_eps"] for r in use])),
                 k_eff_med=float(np.median([r["k_eff"] for r in use])),
                 top4_med=float(np.median([r["top4_share"] for r in use])),
                 margin_med=float(np.median([r["margin_norm"] for r in use])),
                 nu=rr[0]["nu"], family=rr[0]["family"], m=rr[0]["m"], p=rr[0]["p"])
        summary["cells"][f"{coh}/{cell}"] = d
        print(f"{coh:6} {cell:28} {d['n']:4d} {d['n_grok']:5d} {d['K_gap_med']:9.1f} "
              f"{d['K_gap_mode']:5d} {d['K_eps_med']:9.1f} {d['k_eff_med']:9.2f} "
              f"{d['top4_med']:6.3f} {d['margin_med']:9.3e}")

    # ---- pre-registered tests (README)
    A = {k: v for k, v in summary["cells"].items() if k.startswith("A/")}
    if A:
        def rho(x, y):
            from scipy.stats import spearmanr
            return float(spearmanr(x, y).correlation)
        try:
            nus = [v["nu"] for v in A.values()]
            summary["tests"]["P-N1_keff_vs_nu_spearman"] = rho(nus, [v["k_eff_med"] for v in A.values()])
            summary["tests"]["P-N1_Kgap_vs_nu_spearman"] = rho(nus, [v["K_gap_med"] for v in A.values()])
        except ImportError:
            pass
        nu4 = [r for r in rows if r["cohort"] == "A" and r["nu"] == 4 and r["test_acc"] >= 0.99]
        if nu4:
            summary["tests"]["P-N2_frac_K4_at_nu4"] = float(np.mean([r["K_gap"] == 4 for r in nu4]))
            summary["tests"]["P-N2_frac_K3to5_at_nu4"] = float(np.mean([3 <= r["K_gap"] <= 5 for r in nu4]))
        nu2 = [r for r in rows if r["cohort"] == "A" and r["nu"] == 2 and r["test_acc"] >= 0.99]
        if nu2:
            half = (nu2[0]["p"] - 1) // 2
            summary["tests"]["P-N3_frac_dense_at_nu2"] = float(np.mean([r["K_eps"] >= 0.8 * half for r in nu2]))
        # width-flatness at nu>=3 vs width growth at nu=2 (k_eff vs m within family/nu)
        for fam, nu in {(v["family"], v["nu"]) for v in A.values()}:
            vv = sorted([v for v in A.values() if v["family"] == fam and v["nu"] == nu], key=lambda v: v["m"])
            if len(vv) >= 2:
                summary["tests"][f"P-N4_keff_ratio_maxwidth_over_minwidth_{fam}{nu}"] = \
                    vv[-1]["k_eff_med"] / max(vv[0]["k_eff_med"], 1e-9)
    B = {k: v for k, v in summary["cells"].items() if k.startswith("B/")}
    if B:
        summary["tests"]["P-N5_B_Keps_med_per_lam"] = {
            k: v["K_eps_med"] for k, v in B.items()}
    F = [r for r in rows if r["cohort"] == "F"]
    if F:
        summary["tests"]["P-N8_F_keff_med_by_family_scale"] = {
            f"{fam}_sc{sc:g}": float(np.median([r["k_eff"] for r in F if r["family"] == fam and r["init_scale"] == sc and r["test_acc"] >= 0.99] or [np.nan]))
            for fam in ("relu", "quad") for sc in sorted({r["init_scale"] for r in F})}
        summary["tests"]["P-N8_F_grokfrac_by_family_scale"] = {
            f"{fam}_sc{sc:g}": float(np.mean([r["test_acc"] >= 0.99 for r in F if r["family"] == fam and r["init_scale"] == sc] or [np.nan]))
            for fam in ("relu", "quad") for sc in sorted({r["init_scale"] for r in F})}
    C = {k: v for k, v in summary["cells"].items() if k.startswith(("C/", "D/"))}
    if C:
        summary["tests"]["P-N6_C_Kgap_med_per_cell"] = {k: v["K_gap_med"] for k, v in C.items()}
        summary["tests"]["P-N6_C_Keps_med_per_cell"] = {k: v["K_eps_med"] for k, v in C.items()}
    print(json.dumps(summary["tests"], indent=1))
    out = ROOT / "notes" / "nudial_summary.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
