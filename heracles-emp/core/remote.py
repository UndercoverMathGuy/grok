"""Run training batches on a GPU pod that the ORCHESTRATOR created (NORMS F21: this module never creates
or deletes pods). Direct ssh port mapping only; plain scp for uploads and downloads.

Local side (subcommands of `python -m core.remote`):
    sync   HOST PORT                 copy pyproject.toml, uv.lock, .python-version, core/, tests/ to the pod and
                                     run `uv sync --frozen` there (installs the pinned packages)
    launch HOST PORT EXP BATCHES.json [--tag T]
                                     copy the batch list to the pod and start `core.remote queue` detached
                                     (setsid nohup); returns immediately
    poll   HOST PORT EXP [--tag T]   print the pod-side queue status (done / total, last log lines)
    pull   HOST PORT EXP [--tag T]   scp the finished run directories back into EXP/runs/, verify every file
                                     against the pod-side sha256 manifest, and append the pod's RUNS rows to
                                     the local EXP/RUNS.csv (note 'remote:<tag>')
Pod side:
    queue  EXP BATCHES.json          run each batch with core.train on CUDA, sequentially, appending status to
                                     EXP/RUNS.csv, then write EXP/manifest_<tag>.json (sha256 of every run file)
Key: ~/.runpod/ssh/runpodctl-ssh-key (override with HERACLES_SSH_KEY). Remote root: /workspace/heracles-emp.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_ROOT = "/workspace/heracles-emp"
KEY = os.environ.get("HERACLES_SSH_KEY", os.path.expanduser("~/.runpod/ssh/runpodctl-ssh-key"))
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30",
            "-o", "LogLevel=ERROR"]


def _ssh(host: str, port: int, cmd: str, check: bool = True, capture: bool = True):
    full = ["ssh", "-i", KEY, "-p", str(port), *SSH_OPTS, f"root@{host}", cmd]
    r = subprocess.run(full, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd}\n{r.stdout}\n{r.stderr}")
    return r


def _scp_up(host: str, port: int, local: str, remote: str, recursive: bool = False):
    cmd = ["scp", "-i", KEY, "-P", str(port), *SSH_OPTS] + (["-r"] if recursive else []) + [local, f"root@{host}:{remote}"]
    subprocess.run(cmd, check=True)


def _scp_down(host: str, port: int, remote: str, local: str, recursive: bool = False):
    cmd = ["scp", "-i", KEY, "-P", str(port), *SSH_OPTS] + (["-r"] if recursive else []) + [f"root@{host}:{remote}", local]
    subprocess.run(cmd, check=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# local side
# ----------------------------------------------------------------------------------------------

def sync(host: str, port: int):
    """Upload the package code and install the pinned environment on the pod."""
    _ssh(host, port, f"mkdir -p {REMOTE_ROOT}/core/data {REMOTE_ROOT}/tests {REMOTE_ROOT}/experiments")
    for fn in ("pyproject.toml", "uv.lock", ".python-version", "NORMS.md"):
        _scp_up(host, port, os.path.join(ROOT, fn), f"{REMOTE_ROOT}/{fn}")
    for fn in sorted(os.listdir(os.path.join(ROOT, "core"))):
        if fn.endswith(".py") or fn.endswith(".md"):
            _scp_up(host, port, os.path.join(ROOT, "core", fn), f"{REMOTE_ROOT}/core/{fn}")
    for fn in sorted(os.listdir(os.path.join(ROOT, "tests"))):
        if fn.endswith(".py"):
            _scp_up(host, port, os.path.join(ROOT, "tests", fn), f"{REMOTE_ROOT}/tests/{fn}")
    _ssh(host, port, "command -v uv >/dev/null 2>&1 || pip install -q uv", check=True)
    r = _ssh(host, port, f"cd {REMOTE_ROOT} && uv sync --frozen 2>&1 | tail -5 && "
                         f".venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), "
                         f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'")
    print(r.stdout)
    # verify the code on the pod is byte-identical to the local code
    local = {f"core/{fn}": sha256_file(os.path.join(ROOT, "core", fn))
             for fn in os.listdir(os.path.join(ROOT, "core")) if fn.endswith(".py")}
    r = _ssh(host, port, f"cd {REMOTE_ROOT} && sha256sum core/*.py")
    remote = {line.split()[1]: line.split()[0] for line in r.stdout.strip().splitlines()}
    bad = {k for k in local if remote.get(k) != local[k]}
    if bad:
        raise RuntimeError(f"code mismatch after sync: {sorted(bad)}")
    return {"synced": sorted(local), "torch": r.stdout}


def launch(host: str, port: int, exp: str, batches_json: str, tag: str):
    exp_rel = os.path.relpath(os.path.abspath(exp), ROOT)
    _ssh(host, port, f"mkdir -p {REMOTE_ROOT}/{exp_rel}/runs {REMOTE_ROOT}/{exp_rel}/logs")
    _scp_up(host, port, batches_json, f"{REMOTE_ROOT}/{exp_rel}/batches_{tag}.json")
    # the subshell ( ... & ) detaches the job from the ssh channel, so this call returns immediately
    cmd = (f"cd {REMOTE_ROOT} && (setsid nohup .venv/bin/python -m core.remote queue {shlex.quote(exp_rel)} "
           f"{shlex.quote(exp_rel)}/batches_{tag}.json --tag {tag} > {shlex.quote(exp_rel)}/logs/queue_{tag}.log 2>&1 "
           f"< /dev/null &) ; echo launched")
    print(_ssh(host, port, cmd).stdout.strip())


def poll(host: str, port: int, exp: str, tag: str):
    exp_rel = os.path.relpath(os.path.abspath(exp), ROOT)
    r = _ssh(host, port, f"cd {REMOTE_ROOT}/{exp_rel} && cat queue_status_{tag}.json 2>/dev/null; echo; "
                         f"tail -3 logs/queue_{tag}.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used "
                         f"--format=csv,noheader 2>/dev/null", check=False)
    print(r.stdout)
    return r.stdout


def pull(host: str, port: int, exp: str, tag: str):
    """Download finished runs, verify checksums against the pod manifest, merge RUNS rows."""
    from . import runner

    exp = os.path.abspath(exp)
    exp_rel = os.path.relpath(exp, ROOT)
    man_local = os.path.join(exp, f"manifest_{tag}.json")
    _scp_down(host, port, f"{REMOTE_ROOT}/{exp_rel}/manifest_{tag}.json", man_local)
    with open(man_local) as f:
        manifest = json.load(f)
    os.makedirs(os.path.join(exp, "runs"), exist_ok=True)
    bad = []
    for run_id, files in manifest["runs"].items():
        dst = os.path.join(exp, "runs", run_id)
        if os.path.exists(dst):
            # already pulled: verify only
            pass
        else:
            _scp_down(host, port, f"{REMOTE_ROOT}/{exp_rel}/runs/{run_id}", os.path.join(exp, "runs"), recursive=True)
        for fn, sha in files.items():
            if sha256_file(os.path.join(dst, fn)) != sha:
                bad.append(f"{run_id}/{fn}")
    if bad:
        raise RuntimeError(f"checksum mismatch: {bad}")
    rows_path = os.path.join(exp, f"RUNS_remote_{tag}.csv")
    _scp_down(host, port, f"{REMOTE_ROOT}/{exp_rel}/RUNS.csv", rows_path)
    import csv
    with open(rows_path) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] != "launched"]
    for r in rows:
        r["note"] = f"remote:{tag} {r.get('note', '')}"
        r["run_dir"] = os.path.join(exp, "runs", r["run_id"])
        r.pop("time", None)
        r.pop("host", None)
    runner.append_runs_csv(exp, rows)
    print(f"pulled {len(manifest['runs'])} runs, all checksums OK")
    return {"n_runs": len(manifest["runs"]), "bad": bad}


# ----------------------------------------------------------------------------------------------
# pod side
# ----------------------------------------------------------------------------------------------

def queue(exp: str, batches_json: str, tag: str):
    from . import runner, train

    exp = os.path.abspath(exp)
    with open(batches_json) as f:
        batches = json.load(f)["batches"]
    status_path = os.path.join(exp, f"queue_status_{tag}.json")
    done = 0
    for b in batches:
        bid = b["batch_id"]
        todo = []
        for r in b["runs"]:
            r = dict(r)
            r["out_dir"] = os.path.join(exp, "runs", r["run_id"])
            if os.path.exists(os.path.join(r["out_dir"], "metrics.json")):
                continue
            if os.path.exists(r["out_dir"]) and os.listdir(r["out_dir"]):
                runner.append_runs_csv(exp, [runner._row(r, bid, "crashed", "incomplete run dir; rerun under new id")])
                continue
            todo.append(r)
        if todo:
            os.makedirs(os.path.join(exp, "configs"), exist_ok=True)
            with open(os.path.join(exp, "configs", f"{bid}.json"), "w") as f:
                json.dump({"batch_id": bid, "runs": todo}, f, indent=1)
            runner.append_runs_csv(exp, [runner._row(r, bid, "launched") for r in todo])
            t0 = time.time()
            try:
                summ = train.train_batch(todo, verbose=False)
                rows = [runner._row(r, bid, s["status"], "cuda", time.time() - t0) for r, s in zip(todo, summ)]
            except Exception as e:  # recorded, never dropped
                rows = [runner._row(r, bid, "crashed", f"exception: {e!r}"[:300], time.time() - t0) for r in todo]
            runner.append_runs_csv(exp, rows)
        done += 1
        with open(status_path, "w") as f:
            json.dump({"done_batches": done, "total_batches": len(batches), "time": time.strftime("%H:%M:%S")}, f)
    manifest = {"tag": tag, "runs": {}}
    for b in batches:
        for r in b["runs"]:
            d = os.path.join(exp, "runs", r["run_id"])
            if os.path.isdir(d):
                manifest["runs"][r["run_id"]] = {fn: sha256_file(os.path.join(d, fn)) for fn in sorted(os.listdir(d))}
    with open(os.path.join(exp, f"manifest_{tag}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(status_path, "w") as f:
        json.dump({"done_batches": done, "total_batches": len(batches), "finished": True,
                   "time": time.strftime("%H:%M:%S")}, f)


def main(argv=None):
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("host"); s.add_argument("port", type=int)
    s = sub.add_parser("launch"); s.add_argument("host"); s.add_argument("port", type=int); s.add_argument("exp")
    s.add_argument("batches_json"); s.add_argument("--tag", default=time.strftime("%m%d%H%M"))
    s = sub.add_parser("poll"); s.add_argument("host"); s.add_argument("port", type=int); s.add_argument("exp")
    s.add_argument("--tag", required=True)
    s = sub.add_parser("pull"); s.add_argument("host"); s.add_argument("port", type=int); s.add_argument("exp")
    s.add_argument("--tag", required=True)
    s = sub.add_parser("queue"); s.add_argument("exp"); s.add_argument("batches_json"); s.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sync":
        sync(a.host, a.port)
    elif a.cmd == "launch":
        launch(a.host, a.port, a.exp, a.batches_json, a.tag)
    elif a.cmd == "poll":
        poll(a.host, a.port, a.exp, a.tag)
    elif a.cmd == "pull":
        pull(a.host, a.port, a.exp, a.tag)
    elif a.cmd == "queue":
        queue(a.exp, a.batches_json, a.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
