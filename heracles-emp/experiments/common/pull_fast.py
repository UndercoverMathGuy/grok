"""Pull every run directory of EXP from one pod with ONE plain `scp -r`, verify every file by sha256 against (a) a
manifest computed on the pod at pull time and (b) the queue manifests (manifest_<tag>.json) where they cover the file,
then move verified run dirs into experiments/EXP/runs/ and merge the pod's RUNS.csv rows once (note 'remote:<pod>').
Usage: python experiments/common/pull_fast.py POD HOST PORT EXP"""
import csv, glob, hashlib, json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from core import remote as R, runner

pod, host, port, exp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rex = f"{R.REMOTE_ROOT}/experiments/{exp}"
loc = os.path.join(ROOT, "experiments", exp)
stage = os.path.join(loc, f".pull_{pod}")
os.makedirs(stage, exist_ok=True)
busy = R._ssh(host, port, f"pgrep -f '^.venv/bin/python -m core.remote queue experiments/{exp} ' || true").stdout.strip()
if busy:
    raise SystemExit(f"{pod} {exp}: a queue for this experiment is still running on the pod (pids {busy}); not pulling")
man = R._ssh(host, port, f"cd {rex}/runs && find . -type f -print0 | xargs -0 sha256sum").stdout
pod_sha = {}
for line in man.strip().splitlines():
    h, f = line.split(None, 1); pod_sha[f[2:] if f.startswith("./") else f] = h
subprocess.run(["scp", "-q", "-i", R.KEY, "-P", str(port), *R.SSH_OPTS, "-r", f"root@{host}:{rex}/runs", stage], check=True)
subprocess.run(["scp", "-q", "-i", R.KEY, "-P", str(port), *R.SSH_OPTS, f"root@{host}:{rex}/RUNS.csv", os.path.join(stage, "RUNS_pod.csv")], check=True)
subprocess.run(["bash", "-c", f"scp -q -i {R.KEY} -P {port} {' '.join(R.SSH_OPTS)} 'root@{host}:{rex}/manifest_*.json' {stage}/ 2>/dev/null || true"])
bad = []
for f, h in pod_sha.items():
    if R.sha256_file(os.path.join(stage, "runs", f)) != h:
        bad.append(f)
qm = {}
for mf in glob.glob(os.path.join(stage, "manifest_*.json")):
    for rid, files in json.load(open(mf))["runs"].items():
        for fn, h in files.items():
            qm[f"{rid}/{fn}"] = h
for f, h in qm.items():
    if f in pod_sha and pod_sha[f] != h:
        bad.append("queue-manifest:" + f)
if bad:
    raise SystemExit(f"CHECKSUM MISMATCH {pod} {exp}: {bad[:10]}")
os.makedirs(os.path.join(loc, "runs"), exist_ok=True)
moved = skipped = 0
for d in sorted(os.listdir(os.path.join(stage, "runs"))):
    dst = os.path.join(loc, "runs", d)
    if os.path.exists(dst):
        skipped += 1; continue
    shutil.move(os.path.join(stage, "runs", d), dst); moved += 1
rows = [r for r in csv.DictReader(open(os.path.join(stage, "RUNS_pod.csv"))) ]
for r in rows:
    r["note"] = f"remote:{pod} {r.get('note', '')}".strip(); r["run_dir"] = os.path.join(loc, "runs", r["run_id"])
    r.pop("time", None); r.pop("host", None)
done_file = os.path.join(loc, f"pulled_{pod}.json")
prev = json.load(open(done_file)) if os.path.exists(done_file) else {"n_rows": 0}
new_rows = rows[prev["n_rows"]:]  # the pod RUNS.csv is append-only: merge only rows not merged before
runner.append_runs_csv(loc, new_rows)
json.dump({"n_rows": len(rows), "files_verified": len(pod_sha), "queue_manifest_files_checked": len(qm)}, open(done_file, "w"))
shutil.rmtree(stage)
print(f"{pod} {exp}: moved {moved} run dirs (skipped {skipped} present), verified {len(pod_sha)} files (+{len(qm)} vs queue manifests), merged {len(new_rows)} RUNS rows")
