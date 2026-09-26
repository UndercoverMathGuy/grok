"""Runner (core/runner.py): machine-wide slot limit, RUNS.csv bookkeeping, skip-completed, timeouts."""
import csv
import os
import sys
import threading
import time

from core import runner

BASE = dict(model="product", n=2, p=5, q=2.0, width=6, lam=1e-3, lr=1.0, steps=50, seed=1, log_every=25,
            init_scale=1.0, measure_ablation=False)


def test_slot_limit_ten_dummy_jobs(tmp_path):
    stamp = tmp_path / "stamps"
    stamp.mkdir()
    code = ("import os,sys,time; d=sys.argv[1]; i=sys.argv[2]; "
            "open(os.path.join(d,i+'.s'),'w').write(repr(time.time())); "
            "assert os.environ['OMP_NUM_THREADS']=='1'; time.sleep(2.0); "
            "open(os.path.join(d,i+'.e'),'w').write(repr(time.time()))")
    results = []

    def go(i):
        results.append(runner.run_command([sys.executable, "-c", code, str(stamp), str(i)], str(tmp_path / f"{i}.log")))

    th = [threading.Thread(target=go, args=(i,)) for i in range(10)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert all(r["returncode"] == 0 for r in results), results
    iv = [(float((stamp / f"{i}.s").read_text()), float((stamp / f"{i}.e").read_text())) for i in range(10)]
    events = sorted([(s, 1) for s, e in iv] + [(e, -1) for s, e in iv], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    print(f"max concurrent dummy jobs: {mx}")
    assert mx <= runner.N_SLOTS
    assert mx >= 2  # the jobs did run in parallel
    assert len({r["slot"] for r in results}) <= runner.N_SLOTS


def test_timeout_is_recorded(tmp_path):
    r = runner.run_command([sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path / "t.log"), timeout=1.0)
    assert r["timed_out"] and r["wall_s"] < 15


def test_runs_csv_skip_completed_and_crashed(tmp_path):
    exp = str(tmp_path / "EXP")
    os.makedirs(exp)
    batch = {"batch_id": "b0", "runs": [{"run_id": "r0", "config": dict(BASE)},
                                        {"run_id": "r1", "config": dict(BASE, seed=2)}]}
    rows = runner.run_batch(exp, batch)
    assert {r["status"] for r in rows} <= {"ok", "not_converged"}
    last = runner.read_runs_csv(exp)
    assert set(last) == {"r0", "r1"}
    with open(os.path.join(exp, "RUNS.csv")) as f:
        n_rows = len(list(csv.DictReader(f)))
    assert n_rows == 4  # launched + final for each run
    # rerun: completed runs are skipped, nothing appended
    assert runner.run_batch(exp, batch) == []
    with open(os.path.join(exp, "RUNS.csv")) as f:
        assert len(list(csv.DictReader(f))) == 4
    # an incomplete run dir is never overwritten; it is recorded as crashed
    os.makedirs(os.path.join(exp, "runs", "r2"))
    open(os.path.join(exp, "runs", "r2", "log.csv"), "w").write("partial")
    runner.run_batch(exp, {"batch_id": "b1", "runs": [{"run_id": "r2", "config": dict(BASE)}]})
    assert runner.read_runs_csv(exp)["r2"]["status"] == "crashed"
    assert open(os.path.join(exp, "runs", "r2", "log.csv")).read() == "partial"
