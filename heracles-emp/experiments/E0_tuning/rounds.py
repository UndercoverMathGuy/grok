"""Declared E0 tuning rounds. A round is added here (and described in LOG.md) BEFORE it is launched and is
never edited afterwards; later rounds are appended. Objective of every round: convergence quality only."""
import math

BASE = {"optimizer": "gd", "dtype": "float32", "device": "cpu", "measure_ablation": False, "init_scheme": "normal"}

ROUNDS = {}

# R1 (probe): learn how the stable/fast learning rate, momentum and init scale behave at the smallest cells.
ROUNDS["R1"] = {
    "base": {**BASE, "steps": 20000, "log_every": 1000},
    "batch": 10,
    "cells": [
        {"name": "prod_nu3_p11_q2", "shape": {"model": "product", "n": 2, "p": 11, "q": 2.0, "width": 30},
         "grid": {"lr": [0.5, 1.0, 2.0, 4.0, 8.0], "momentum": [0.0, 0.9], "init_scale": [1.0, math.sqrt(11)],
                  "lam": [1e-3, 1e-4], "seed": [0]}},
        {"name": "prod_nu3_p11_q3", "shape": {"model": "product", "n": 2, "p": 11, "q": 3.0, "width": 30},
         "grid": {"lr": [0.5, 1.0, 2.0, 4.0, 8.0], "momentum": [0.0, 0.9], "init_scale": [1.0, math.sqrt(11)],
                  "lam": [1e-3, 1e-4], "seed": [0]}},
        {"name": "prod_nu4_p7_q2", "shape": {"model": "product", "n": 3, "p": 7, "q": 2.0, "width": 24},
         "grid": {"lr": [0.5, 1.0, 2.0, 4.0, 8.0], "momentum": [0.0, 0.9], "init_scale": [1.0, math.sqrt(7)],
                  "lam": [1e-3, 1e-4], "seed": [0]}},
        {"name": "prod_nu5_p7_q2", "shape": {"model": "product", "n": 4, "p": 7, "q": 2.0, "width": 30},
         "grid": {"lr": [0.5, 1.0, 2.0, 4.0, 8.0], "momentum": [0.0, 0.9], "init_scale": [1.0, math.sqrt(7)],
                  "lam": [1e-3, 1e-4], "seed": [0]}},
        {"name": "pow_nu3_p11_q3", "shape": {"model": "power", "n": 2, "p": 11, "q": 3.0, "width": 30},
         "grid": {"lr": [0.5, 1.0, 2.0, 4.0, 8.0], "momentum": [0.0, 0.9], "init_scale": [1.0, math.sqrt(11)],
                  "lam": [1e-3, 1e-4], "seed": [0]}},
    ],
}

# R2 (CPU target cells): momentum 0.9 (R1: needed for lam <= 1e-4 within budget), init entries ~ N(0,1)
# (init_scale = sqrt(p); R1: escapes the origin basin at nu=4,5), lam in {1e-4, 1e-5}, lr grid per nu (R1: nu=5
# diverges at lr >= 1 with lam = 1e-4), q in {2, (2+nu)/2, nu}, 40000 steps, seed 0.
def _r2_cells():
    cells = []
    lrs = {("product", 3): [0.5, 1.0, 2.0, 4.0], ("product", 4): [0.25, 0.5, 1.0, 2.0],
           ("product", 5): [0.05, 0.1, 0.2, 0.4], ("power", 3): [0.25, 0.5, 1.0, 2.0]}
    plan = [("product", 3, [11, 17]), ("product", 4, [7, 11]), ("product", 5, [7]), ("power", 3, [11])]
    for model, nu, ps in plan:
        n = nu - 1
        for p in ps:
            h = (p - 1) // 2
            for q in (2.0, (2.0 + nu) / 2, float(nu)):
                tag = f"{'prod' if model == 'product' else 'pow'}_nu{nu}_p{p}_q{q:g}"
                cells.append({"name": tag, "shape": {"model": model, "n": n, "p": p, "q": q, "width": 2 * nu * h},
                              "grid": {"lr": lrs[(model, nu)], "momentum": [0.9], "init_scale": [math.sqrt(p)],
                                       "lam": [1e-4, 1e-5], "seed": [0]}})
    return cells


ROUNDS["R2"] = {"base": {**BASE, "steps": 40000, "log_every": 1000}, "batch": 8, "cells": _r2_cells()}


# R3 (GPU, declared 2026-09-25 17:38 IST by the empirical lead; replaces R2, which was never launched).
# Structure of every recipe: full-batch GD, heavy-ball momentum 0.9 (R1: needed for lam <= 1e-4 within budget),
# init entries ~ N(0, 1) (init_scale = sqrt(p); R1: escapes the origin basin at nu = 4, 5), width 2 nu h,
# constant lr = c * p (declared scaling: the mean-CE gradient per coordinate scales like 1/p with O(1)
# coordinates), constant lambda. Grid per (model, nu): c (factor-2 grid, centred on R1's stable range at
# p = 7..11) x lambda in {1e-4, 1e-5} x q in {2, nu} (the two ends of the allowed range; power: q = nu only)
# at a small and a large p. Steps: 30000 (nu = 3, 4), 20000 (nu = 5).
# Declared selection rule (convergence quality only, NORMS C14b; applied mechanically by select_R3.py):
#   for each (model, nu), candidate (c, lambda): (1) most runs with status 'ok' over the cell's p x q x seeds;
#   (2) ties -> the smaller lambda (closer to the lambda -> 0 limit of the theorems; fixed a priori);
#   (3) ties -> the lower median final grad_rel.
#   The frozen recipe uses the selected (c, lambda), the same steps, and applies lr = c * p at every p.
def _r3_cells():
    cells = []
    cgrid = {("product", 3): [0.05, 0.1, 0.2, 0.4], ("product", 4): [0.035, 0.07, 0.14, 0.28],
             ("product", 5): [0.0175, 0.035, 0.07, 0.14], ("power", 3): [0.05, 0.1, 0.2, 0.4]}
    plan = [("product", 3, [23, 113], [0]), ("power", 3, [23, 113], [0, 1]), ("product", 4, [23, 43], [0]),
            ("product", 5, [13, 23], [0])]
    for model, nu, ps, seeds in plan:
        n = nu - 1
        steps = 20000 if nu == 5 else 30000
        qs = [float(nu)] if model == "power" else [2.0, float(nu)]
        for p in ps:
            h = (p - 1) // 2
            runs = []
            for c in cgrid[(model, nu)]:
                for lam in (1e-4, 1e-5):
                    for q in qs:
                        for seed in seeds:
                            runs.append({"lr": c * p, "lr_c": c, "lam": lam, "q": q, "seed": seed})
            tag = f"{'prod' if model == 'product' else 'pow'}_nu{nu}_p{p}"
            cells.append({"name": tag, "shape": {"model": model, "n": n, "p": p, "width": 2 * nu * h, "steps": steps},
                          "runs": runs})
    return cells


ROUNDS["R3"] = {"base": {**BASE, "device": "cuda", "momentum": 0.9, "log_every": 1000}, "batch": 16,
                "cells": _r3_cells(), "init_scale_sqrt_p": True}


# R4 (GPU, declared 2026-09-25 17:42 IST, after reading R3 convergence diagnostics only).
# R3 finding (optimisation only): a fixed lambda does not transfer across (nu, q, p). At q = nu, lambda = 1e-4
# (nu = 4) and 1e-5 (nu = 5) collapse to the origin (margin 0, grad_rel ~ 1): the penalty lambda ||theta||^q
# outweighs the CE gain. Dimensional analysis of mean CE(h) + lambda ||theta||_{2,q}^q with h = gamma ||theta||^nu:
# with u = gamma ||theta||^nu the objective is CE(u) + lambda gamma^{-q/nu} u^{q/nu}, so the dimensionless strength is
# lambda * gamma^{-q/nu}. R4 therefore parametrises lambda = lt * B_p^{q/nu}, B_p = 2 (2 p nu)^{-nu/2} (the
# single-neuron constant: depends on (p, nu) only, not on any programme prediction), and tunes lt.
# Grid: lt in {1e-2, 3e-3} x c (lr = c p; the R3 values that did not diverge) x q in {2, nu}; steps 40000; seed 0
# (power: seeds 0, 1). Selection rule: identical to R3 (select_R3.py), with lt in place of lambda.
def _r4_cells():
    import math as _m
    cells = []
    cgrid = {("product", 3): [0.2, 0.4], ("product", 4): [0.07, 0.14, 0.28],
             ("product", 5): [0.0175, 0.035], ("power", 3): [0.05, 0.1]}
    plan = [("product", 3, [23, 113], [0]), ("power", 3, [23, 113], [0, 1]), ("product", 4, [23, 43], [0]),
            ("product", 5, [13, 17, 23], [0])]
    for model, nu, ps, seeds in plan:
        n = nu - 1
        qs = [float(nu)] if model == "power" else [2.0, float(nu)]
        for p in ps:
            h = (p - 1) // 2
            Bp = 2.0 * (2.0 * p * nu) ** (-nu / 2.0)
            runs = []
            for c in cgrid[(model, nu)]:
                for lt in (1e-2, 3e-3):
                    for q in qs:
                        for seed in seeds:
                            runs.append({"lr": c * p, "lr_c": c, "lam_tilde": lt, "lam": lt * Bp ** (q / nu), "q": q,
                                         "seed": seed})
            tag = f"{'prod' if model == 'product' else 'pow'}_nu{nu}_p{p}"
            cells.append({"name": tag, "shape": {"model": model, "n": n, "p": p, "width": 2 * nu * h, "steps": 40000},
                          "runs": runs})
    return cells


ROUNDS["R4"] = {"base": {**BASE, "device": "cuda", "momentum": 0.9, "log_every": 1000}, "batch": 16,
                "cells": _r4_cells(), "init_scale_sqrt_p": True}
