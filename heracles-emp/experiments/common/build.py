"""Build the evaluation run grids from the FROZEN recipe (core/RECIPES.md, experiments/common/RECIPE.json).

Usage: python experiments/common/build.py EXP   -> writes experiments/EXP/configs/grid.json (all runs, batched)
Every run config = frozen recipe for (model, nu) + (p, q, seed[, init_relabel_u]); lr = c p, lambda = lt B_p^{q/nu},
width = 2 nu h, init entries N(0, 1) (init_scale = sqrt p), full-batch GD with heavy-ball momentum, float32 on CUDA
with TF32 off (asserted by core.train), measure_ablation on.
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RECIPE = json.load(open(os.path.join(HERE, "RECIPE.json")))


def Bp(p, nu):
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def config(model, nu, p, q, seed, **extra):
    r = RECIPE["recipes"][f"{model}_nu{nu}"]
    h = (p - 1) // 2
    cfg = {"model": model, "n": nu - 1, "p": p, "q": float(q), "width": 2 * nu * h, "steps": (r.get("steps_p59", r["steps"]) if p == 59 else r["steps"]),
           "optimizer": "gd", "momentum": r["momentum"], "lr": r["c"] * p, "lr_c": r["c"],
           "lam": r["lt"] * Bp(p, nu) ** (float(q) / nu), "lam_tilde": r["lt"], "init_scheme": "normal",
           "init_scale": math.sqrt(p), "seed": int(seed), "dtype": "float32", "device": "cuda",
           "log_every": r["log_every"], "measure_ablation": True, "recipe_id": RECIPE["recipe_id"]}
    cfg.update(extra)
    return cfg


def q_of(nu, s):
    return nu / s


GRIDS = {}
# E1 / E1b: product networks, s grid shared across nu, several p per nu, seeds declared in advance
E1_S = {3: [1.25, 1.5], 4: [1.25, 1.5, 2.0], 5: [1.25, 1.5, 2.0, 2.5]}
E1_P = {3: [23, 31, 43, 59, 83, 113], 4: [17, 23, 31, 43, 59], 5: [11, 13, 17]}
E1_SEEDS = {3: list(range(6)), 4: list(range(4)), 5: list(range(4))}


def grid_E1():
    cells = []
    for nu in (3, 4, 5):
        for p in E1_P[nu]:
            runs = []
            for s in E1_S[nu]:
                for seed in E1_SEEDS[nu]:
                    runs.append((f"E1_nu{nu}_p{p}_s{s:g}_seed{seed}", config("product", nu, p, q_of(nu, s), seed, s_target=s)))
            cells.append((f"nu{nu}_p{p}", runs, 6 if (nu == 4 and p == 59) else 16))
    return cells


def grid_E3():
    cells = []
    plan = [("product", 3, [23, 59, 113]), ("product", 4, [23, 43, 59]), ("product", 5, [11, 13, 17]), ("power", 3, [23, 59, 113])]
    for model, nu, ps in plan:
        for p in ps:
            runs = [(f"E3_{model}_nu{nu}_p{p}_seed{seed}", config(model, nu, p, float(nu), seed, s_target=1.0)) for seed in range(4)]
            cells.append((f"{model}_nu{nu}_p{p}", runs, 16))
    return cells


def grid_E7(qs, p=37, nu=4, seeds=(0, 1)):
    runs = [(f"E7_nu{nu}_p{p}_q{q:.4f}_seed{seed}", config("product", nu, p, q, seed, s_target=nu / q)) for q in qs for seed in seeds]
    return [(f"nu{nu}_p{p}", runs, 16)]


def grid_E6():
    cells = []
    paired = [(f"E6_pair_seed{seed}_u{u}", config("product", 3, 23, 2.0, seed, init_relabel_u=u, s_target=1.5))
              for seed in range(6) for u in (1, 2, 5)]
    cells.append(("pairs", paired, 18))
    marg = [(f"E6_marg_seed{seed}", config("product", 3, 23, 2.0, seed, s_target=1.5)) for seed in range(100, 132)]
    cells.append(("marginals", marg, 16))
    return cells


def grid_E8():
    cells = []
    for p in [11, 13, 17, 19, 29, 37, 47, 71]:
        runs = [(f"E8_nu3_p{p}_s{s:g}_seed{seed}", config("product", 3, p, q_of(3, s), seed, s_target=s))
                for s in (1.25, 1.5) for seed in range(3)]
        cells.append((f"nu3_p{p}", runs, 16))
    return cells


def write(exp, cells):
    d = os.path.join(ROOT, "experiments", exp, "configs"); os.makedirs(d, exist_ok=True)
    batches = []
    for name, runs, bs in cells:
        for k in range(0, len(runs), bs):
            batches.append({"batch_id": f"{exp}_{name}_b{k // bs:02d}",
                            "runs": [{"run_id": rid, "config": cfg} for rid, cfg in runs[k:k + bs]]})
    json.dump({"batches": batches, "recipe_id": RECIPE["recipe_id"]}, open(os.path.join(d, "grid.json"), "w"), indent=1)
    print(exp, len(batches), "batches", sum(len(b["runs"]) for b in batches), "runs")
    return batches


if __name__ == "__main__":
    exp = sys.argv[1]
    if exp.startswith("E1"):
        write("E1", grid_E1())
    elif exp.startswith("E3"):
        write("E3", grid_E3())
    elif exp.startswith("E6"):
        write("E6", grid_E6())
    elif exp.startswith("E8"):
        write("E8", grid_E8())
    elif exp.startswith("E7"):
        qs = [float(x) for x in json.load(open(os.path.join(ROOT, "experiments", "E7", "q_grid.json")))["q"]]
        write("E7", grid_E7(qs))
