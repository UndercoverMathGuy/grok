"""Assign evaluation batches to pods and launch one detached, chained queue per pod (core.remote queue per step).
Usage: python experiments/common/launch.py PLAN.json [--dry]
PLAN.json = {"pods": {"P1": {"host": .., "port": .., "steps": [["E1", ["E1_nu4_p59_b00", ...]], ["E6", [...]]]}}}"""
import json, os, shlex, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from core import remote as R

plan = json.load(open(sys.argv[1])); dry = "--dry" in sys.argv
grids = {}
used = set()
for pod, spec in plan["pods"].items():
    cmds = []
    for k, (exp, bids) in enumerate(spec["steps"]):
        if exp not in grids:
            grids[exp] = {b["batch_id"]: b for b in json.load(open(os.path.join(ROOT, "experiments", exp, "configs", "grid.json")))["batches"]}
            rr = os.path.join(ROOT, "experiments", exp, "configs", "grid_reruns.json")
            if os.path.exists(rr):  # reruns under new run ids (NORMS B9), e.g. after a queue was stopped for rebalancing
                grids[exp].update({b["batch_id"]: b for b in json.load(open(rr))["batches"]})
        bs = [grids[exp][b] for b in bids]
        for b in bids:
            assert (exp, b) not in used, (exp, b); used.add((exp, b))
        tag = f"{pod}s{k}"
        fn = os.path.join(ROOT, "experiments", exp, "configs", f"batches_{tag}.json")
        json.dump({"batches": bs}, open(fn, "w"), indent=1)
        rel = os.path.relpath(fn, ROOT)
        if not dry:
            R._ssh(spec["host"], spec["port"], f"mkdir -p {R.REMOTE_ROOT}/experiments/{exp}/configs {R.REMOTE_ROOT}/experiments/{exp}/logs {R.REMOTE_ROOT}/experiments/{exp}/runs")
            R._scp_up(spec["host"], spec["port"], fn, f"{R.REMOTE_ROOT}/{rel}")
        cmds.append(f".venv/bin/python -m core.remote queue experiments/{exp} {rel} --tag {tag} > experiments/{exp}/logs/queue_{tag}.log 2>&1")
    chain = " ; ".join(cmds)
    print(pod, len(spec["steps"]), "steps:", [(e, len(b)) for e, b in spec["steps"]])
    if not dry:
        cmd = f"cd {R.REMOTE_ROOT} && (setsid nohup bash -c {shlex.quote(chain)} > chain_{pod}.log 2>&1 < /dev/null &) ; echo launched {pod}"
        print(R._ssh(spec["host"], spec["port"], cmd).stdout.strip())
# every batch of every grid assigned exactly once
if not plan.get("partial"):
    for exp in grids:
        missing = [b for b in grids[exp] if (exp, b) not in used]
        if missing:
            print("UNASSIGNED", exp, missing)
