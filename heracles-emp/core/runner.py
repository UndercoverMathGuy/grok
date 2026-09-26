"""Local CPU job runner with a machine-wide slot lock (NORMS F20/F24).

- Slots: fcntl.flock on heracles-emp/.slots/slot0..slot5. A job runs only while it holds one slot, so at
  most 6 jobs run at once across ALL processes and agents that use this runner. The locked file
  descriptor is passed to the child process, so the lock is held until the child itself exits (even if
  the launching process dies), and is released by the OS if the child crashes.
- Every job is its own process with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, HERACLES_THREADS=1 (the trainer
  then calls torch.set_num_threads(1)).
- Status is APPENDED to the experiment's RUNS.csv under an fcntl lock (append-only history: the last
  row of a run_id is its current status). Completed runs (metrics.json present) are skipped. A run
  directory that exists without metrics.json is never overwritten: it is recorded as 'crashed' and
  must be rerun under a new run id (NORMS B9).
- Timeouts: a job exceeding `timeout` seconds is killed (its whole process group) and its runs are
  recorded 'crashed' with note 'timeout'.

CLI:
    python -m core.runner EXP_DIR BATCHES.json [--parallel K] [--timeout SECONDS]
BATCHES.json = {"batches": [{"batch_id": str, "runs": [{"run_id": str, "config": {...}}, ...]}, ...]}
Run directories are EXP_DIR/runs/<run_id>; batch specs are saved to EXP_DIR/configs/<batch_id>.json and
job stdout/stderr to EXP_DIR/logs/<batch_id>.log.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT_DIR = os.path.join(ROOT, ".slots")
N_SLOTS = 6
RUNS_FIELDS = ["time", "run_id", "batch_id", "status", "note", "run_dir", "model", "n", "p", "q", "width", "lam",
               "lr", "steps", "seed", "wall_s", "host"]
VALID_STATUS = ("launched", "ok", "not_converged", "diverged", "crashed", "invalid", "skipped_completed")


# ----------------------------------------------------------------------------------------------
# slots
# ----------------------------------------------------------------------------------------------

def acquire_slot(poll: float = 0.25, wait_timeout: float | None = None):
    """Block until one of the N_SLOTS machine-wide slots is free; return (fd, slot_index)."""
    os.makedirs(SLOT_DIR, exist_ok=True)
    t0 = time.time()
    while True:
        for i in range(N_SLOTS):
            path = os.path.join(SLOT_DIR, f"slot{i}")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} host={socket.gethostname()} t={time.time():.0f}\n".encode())
            return fd, i
        if wait_timeout is not None and time.time() - t0 > wait_timeout:
            raise TimeoutError("no free slot")
        time.sleep(poll)


def release_slot(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def job_env() -> dict:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "HERACLES_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_command(cmd: list[str], log_path: str, timeout: float | None = None) -> dict:
    """Run one command as its own process while holding a machine-wide slot. Returns
    {'returncode', 'timed_out', 'slot', 'wall_s'}; the slot is held by the child until it exits."""
    fd, slot = acquire_slot()
    t0 = time.time()
    timed_out = False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} slot={slot} cmd={' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=job_env(), cwd=ROOT,
                                    pass_fds=(fd,), start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
    finally:
        release_slot(fd)
    return {"returncode": rc, "timed_out": timed_out, "slot": slot, "wall_s": time.time() - t0}


# ----------------------------------------------------------------------------------------------
# RUNS.csv
# ----------------------------------------------------------------------------------------------

def append_runs_csv(exp_dir: str, rows: list[dict]):
    """Append status rows to EXP_DIR/RUNS.csv under an exclusive fcntl lock."""
    path = os.path.join(exp_dir, "RUNS.csv")
    lock = path + ".lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUNS_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                assert r["status"] in VALID_STATUS, r["status"]
                w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), **r})
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_runs_csv(exp_dir: str) -> dict:
    """{run_id: last row} from EXP_DIR/RUNS.csv (append-only history; the last row wins)."""
    path = os.path.join(exp_dir, "RUNS.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["run_id"]] = r
    return out


def _row(run: dict, batch_id: str, status: str, note: str = "", wall: float | None = None) -> dict:
    c = run["config"]
    return {"run_id": run["run_id"], "batch_id": batch_id, "status": status, "note": note,
            "run_dir": run["out_dir"], "model": c.get("model"), "n": c.get("n"), "p": c.get("p"), "q": c.get("q"),
            "width": c.get("width"), "lam": c.get("lam"), "lr": c.get("lr"), "steps": c.get("steps"),
            "seed": c.get("seed"), "wall_s": "" if wall is None else f"{wall:.1f}"}


# ----------------------------------------------------------------------------------------------
# training batches
# ----------------------------------------------------------------------------------------------

def run_batch(exp_dir: str, batch: dict, timeout: float | None = None) -> list[dict]:
    """Run one training batch (skipping completed runs) and record every run's status."""
    bid = batch["batch_id"]
    todo = []
    for r in batch["runs"]:
        r = dict(r)
        r["out_dir"] = os.path.join(exp_dir, "runs", r["run_id"])
        if os.path.exists(os.path.join(r["out_dir"], "metrics.json")):
            continue  # completed: skip silently (already listed in RUNS.csv)
        if os.path.exists(r["out_dir"]) and os.listdir(r["out_dir"]):
            append_runs_csv(exp_dir, [_row(r, bid, "crashed", "incomplete run dir found; rerun under a new id")])
            continue
        todo.append(r)
    if not todo:
        return []
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)
    spec_path = os.path.join(exp_dir, "configs", f"{bid}.json")
    spec = {"batch_id": bid, "runs": todo}
    if os.path.exists(spec_path):
        spec_path = os.path.join(exp_dir, "configs", f"{bid}.{int(time.time())}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=1)
    append_runs_csv(exp_dir, [_row(r, bid, "launched") for r in todo])
    res = run_command([sys.executable, "-m", "core.train", spec_path],
                      os.path.join(exp_dir, "logs", f"{bid}.log"), timeout=timeout)
    rows = []
    for r in todo:
        mp = os.path.join(r["out_dir"], "metrics.json")
        if res["timed_out"]:
            rows.append(_row(r, bid, "crashed", "timeout (killed by runner)", res["wall_s"]))
        elif os.path.exists(mp):
            with open(mp) as f:
                st = json.load(f)["status"]
            rows.append(_row(r, bid, st, f"rc={res['returncode']} slot={res['slot']}", res["wall_s"]))
        else:
            rows.append(_row(r, bid, "crashed", f"rc={res['returncode']} no metrics.json", res["wall_s"]))
    append_runs_csv(exp_dir, rows)
    return rows


def run_batches(exp_dir: str, batches: list[dict], parallel: int = N_SLOTS, timeout: float | None = None):
    """Run many batches with up to `parallel` local workers (each still needs a machine-wide slot)."""
    queue = list(batches)
    lock = threading.Lock()
    results = []

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                b = queue.pop(0)
            rows = run_batch(exp_dir, b, timeout=timeout)
            with lock:
                results.extend(rows)

    th = [threading.Thread(target=worker) for _ in range(max(1, min(parallel, N_SLOTS)))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return results


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("batches_json")
    ap.add_argument("--parallel", type=int, default=N_SLOTS)
    ap.add_argument("--timeout", type=float, default=7200.0)
    a = ap.parse_args(argv)
    with open(a.batches_json) as f:
        spec = json.load(f)
    rows = run_batches(os.path.abspath(a.exp_dir), spec["batches"], a.parallel, a.timeout)
    print(json.dumps(rows, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
