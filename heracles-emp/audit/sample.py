"""AUDIT sampling: 12 evaluation runs, numpy seed 20260926, stratified across experiments.
Population per experiment = run ids whose LAST RUNS.csv row has status ok or not_converged (scored runs).
Allocation: E1 4, E3 2, E6 2, E7 2, E8 2. Retrain subset (6): the cheapest run of each experiment by the
config-only cost proxy p^n * p * width * steps, plus the next cheapest of the remainder.
Reads only RUNS.csv and config.json (no metric values)."""
import csv, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOC = {"E1": 4, "E3": 2, "E6": 2, "E7": 2, "E8": 2}
rng = np.random.default_rng(20260926)
out = []
for e, k in ALLOC.items():
    last = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "experiments", e, "RUNS.csv"))):
        last[r["run_id"]] = r["status"]
    pop = sorted(i for i, s in last.items() if s in ("ok", "not_converged"))
    pick = rng.choice(len(pop), size=k, replace=False)
    for j in sorted(pick):
        rid = pop[j]
        c = json.load(open(os.path.join(ROOT, "experiments", e, "runs", rid, "config.json")))
        cost = c["p"] ** c["n"] * c["p"] * c["width"] * c["steps"]
        out.append({"exp": e, "run_id": rid, "status": last[rid], "model": c["model"], "nu": c["nu"], "p": c["p"],
                    "q": c["q"], "width": c["width"], "steps": c["steps"], "cost": cost, "pop": len(pop)})
ret = []
for e in ALLOC:
    rows = sorted([o for o in out if o["exp"] == e], key=lambda o: o["cost"])
    ret.append(rows[0]["run_id"])
rest = sorted([o for o in out if o["run_id"] not in ret], key=lambda o: o["cost"])
ret.append(rest[0]["run_id"])
for o in out:
    o["retrain"] = o["run_id"] in ret
json.dump(out, open(os.path.join(ROOT, "audit", "sample.json"), "w"), indent=1)
for o in out:
    print(o)
