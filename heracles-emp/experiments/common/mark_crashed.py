"""Pod-side: append 'crashed' rows for the given run ids of a batch (NORMS B9), with a reason.
Usage: python experiments/common/mark_crashed.py EXP BATCH_ID REASON RUN_ID [RUN_ID ...]"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from core import runner
exp, bid, reason = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [{"run_id": r, "batch_id": bid, "status": "crashed", "note": reason, "run_dir": os.path.join(ROOT, exp, "runs", r)} for r in sys.argv[4:]]
runner.append_runs_csv(os.path.join(ROOT, exp), rows)
print("marked", len(rows))
