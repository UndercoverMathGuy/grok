"""E0 tuning driver (NORMS C14): expands a declared round of tuning runs into runner batches.

Usage (from heracles-emp/):  uv run python experiments/E0_tuning/tune.py ROUND [--parallel K] [--dry]

Rounds are declared in rounds.py BEFORE they are launched (and described in LOG.md). Every run goes
through core.runner and is listed in experiments/E0_tuning/RUNS.csv. The tuning objective is convergence
quality only; nothing here computes agreement with the programme's predictions.
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from core import runner  # noqa: E402
import rounds  # noqa: E402


def expand(round_name: str):
    spec = rounds.ROUNDS[round_name]
    batches = []
    for cell in spec["cells"]:
        base = dict(spec.get("base", {}))
        base.update(cell["shape"])
        grid = cell["grid"]
        keys = sorted(grid)
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
        runs = []
        for i, c in enumerate(combos):
            cfg = dict(base)
            cfg.update(c)
            rid = f"{round_name}_{cell['name']}_{i:03d}"
            runs.append({"run_id": rid, "config": cfg})
        bsz = cell.get("batch", spec.get("batch", 8))
        for k in range(0, len(runs), bsz):
            batches.append({"batch_id": f"{round_name}_{cell['name']}_b{k // bsz:02d}", "runs": runs[k:k + bsz]})
    return batches


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("round")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    batches = expand(a.round)
    n = sum(len(b["runs"]) for b in batches)
    print(f"round {a.round}: {len(batches)} batches, {n} runs")
    os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
    with open(os.path.join(HERE, "configs", f"round_{a.round}.json"), "w") as f:
        json.dump({"round": a.round, "batches": batches}, f, indent=1)
    if a.dry:
        return
    rows = runner.run_batches(HERE, batches, parallel=a.parallel, timeout=a.timeout)
    print(json.dumps([{k: r[k] for k in ("run_id", "status", "wall_s")} for r in rows], indent=0))


if __name__ == "__main__":
    main()
