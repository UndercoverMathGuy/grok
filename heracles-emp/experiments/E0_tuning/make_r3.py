"""Expand the declared R3 round (rounds.py) into one BATCHES json per cell (GPU pods, core.remote)."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rounds
RN = sys.argv[1] if len(sys.argv) > 1 else "R3"
spec = rounds.ROUNDS[RN]
os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
for cell in spec["cells"]:
    runs = []
    for i, r in enumerate(cell["runs"]):
        cfg = dict(spec["base"]); cfg.update(cell["shape"]); cfg.update(r)
        cfg["init_scale"] = math.sqrt(cfg["p"])
        runs.append({"run_id": f"{RN}_{cell['name']}_{i:03d}", "config": cfg})
    batches = [{"batch_id": f"{RN}_{cell['name']}_b{k // spec['batch']:02d}", "runs": runs[k:k + spec["batch"]]}
               for k in range(0, len(runs), spec["batch"])]
    fn = os.path.join(HERE, "configs", f"{RN}_{cell['name']}.json")
    json.dump({"batches": batches}, open(fn, "w"), indent=1)
    print(fn, len(runs))
