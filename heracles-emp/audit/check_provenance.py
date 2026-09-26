"""AUDIT checks 1 (preregistration / recipe) and 2 (completeness / provenance). Own code; reads files only."""
import csv, json, os, re, glob, hashlib, math, datetime as dt, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "experiments")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
out = {}

def utc(s):  # '2026-09-25 12:48:28 UTC'
    return dt.datetime.strptime(s.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

def stamp(e):
    t = open(os.path.join(EX, e, "PREREG.md")).read()
    m = re.search(r"Stamp \(`date`\): \w+ (\w+ \d+ [\d:]+) IST (\d{4})", t)
    return dt.datetime.strptime(m.group(1) + " " + m.group(2), "%b %d %H:%M:%S %Y").replace(tzinfo=IST)

def last_rows(e):
    last = {}
    for r in csv.DictReader(open(os.path.join(EX, e, "RUNS.csv"))):
        last[r["run_id"]] = r
    return last

# ---------------- check 1: prereg timing
pre = {}
for e in ["E1", "E3", "E4", "E5", "E6", "E7", "E8"]:
    st = stamp(e)
    mt = dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(EX, e, "PREREG.md")), IST)
    src = "E1" if e in ("E4", "E5") else e
    starts, ends = [], []
    for c in glob.glob(os.path.join(EX, src, "runs", "*", "config.json")):
        c = json.load(open(c))
        if "start_time" in c: starts.append(utc(c["start_time"]))
        if "end_time" in c: ends.append(utc(c["end_time"]))
    pre[e] = {"stamp_ist": st.isoformat(), "prereg_mtime_ist": mt.isoformat(),
              "earliest_start_ist": min(starts).astimezone(IST).isoformat(),
              "earliest_end_ist": min(ends).astimezone(IST).isoformat(),
              "stamp_before_first_start": st < min(starts), "stamp_before_first_end": st < min(ends),
              "mtime_after_first_start": mt > min(starts)}
out["prereg"] = pre

# ---------------- check 1: recipe
R = json.load(open(os.path.join(EX, "common", "RECIPE.json")))
def expect(c):
    key = f"{c['model']}_nu{c['nu']}"
    rec = R["recipes"][key]
    p, nu, q = c["p"], c["nu"], c["q"]
    h = (p - 1) // 2
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    steps = rec["steps"]
    if key == "product_nu4" and p == 59: steps = rec["steps_p59"]
    return {"optimizer": "gd", "momentum": rec["momentum"], "nesterov": False, "init_scheme": "normal",
            "init_scale": math.sqrt(p), "width": 2 * nu * h, "lr": rec["c"] * p,
            "lam": rec["lt"] * Bp ** (q / nu), "steps": steps, "dtype": "float32",
            "lr_schedule": {"kind": "constant"}, "lam_schedule": {"kind": "constant"},
            "recipe_id": R["recipe_id"], "batch_size": None, "device": "cuda", "log_every": 1000}
viol = collections.defaultdict(list); nrun = collections.Counter(); models = collections.Counter()
flags_bad = []; q_bad = []
for e in ["E1", "E3", "E6", "E7", "E8"]:
    for f in sorted(glob.glob(os.path.join(EX, e, "runs", "*", "config.json"))):
        c = json.load(open(f)); rid = c["run_id"]; nrun[e] += 1
        models[(e, c["model"], c["nu"], c["optimizer"])] += 1
        try:
            ex = expect(c)
        except KeyError as k:
            viol[e].append((rid, "no recipe", str(k))); continue
        for k, v in ex.items():
            got = c.get(k)
            if isinstance(v, float):
                if got is None or abs(got - v) > 1e-9 * max(1, abs(v)): viol[e].append((rid, k, got, v))
            elif got != v:
                viol[e].append((rid, k, got, v))
        pf = c.get("precision_flags", {})
        if not (pf.get("torch.backends.cuda.matmul.allow_tf32") is False and pf.get("torch.backends.cudnn.allow_tf32") is False
                and pf.get("float32_matmul_precision") == "highest"):
            flags_bad.append(rid)
        if not (2 - 1e-12 <= c["q"] <= c["nu"] + 1e-12) or abs(c["s"] - c["nu"] / c["q"]) > 1e-9 or c["nu"] != c["n"] + 1:
            q_bad.append(rid)
        if c.get("init_relabel_u", 1) != 1 and e != "E6":
            viol[e].append((rid, "init_relabel_u", c["init_relabel_u"]))
out["recipe_violations"] = {e: v[:20] for e, v in viol.items()}
out["recipe_violation_counts"] = {e: len(v) for e, v in viol.items()}
out["n_run_dirs"] = dict(nrun)
out["model_optimizer_table"] = {str(k): v for k, v in models.items()}
out["tf32_flag_missing_or_wrong"] = flags_bad
out["q_range_or_s_bad"] = q_bad

# ---------------- check 2: completeness + status agreement
comp = {}
for e in ["E1", "E3", "E6", "E7", "E8"]:
    last = last_rows(e)
    dirs = set(os.listdir(os.path.join(EX, e, "runs")))
    mism = []; files = collections.Counter(); ro = []
    for rid in sorted(dirs):
        d = os.path.join(EX, e, "runs", rid)
        fl = tuple(sorted(os.listdir(d))); files[fl] += 1
        st = last.get(rid, {}).get("status")
        mj = os.path.join(d, "metrics.json")
        if os.path.exists(mj):
            m = json.load(open(mj))
            ch = m["convergence"]["checks"]
            if not ch.get("finite", True): derived = "diverged"
            elif all(ch.values()): derived = "ok"
            else: derived = "not_converged"
            if m.get("status") != st or derived != st:
                mism.append((rid, st, m.get("status"), derived, {k: v for k, v in ch.items() if not v}))
            for fn in ("metrics.json", "final.pt"):
                if os.path.exists(os.path.join(d, fn)) and os.access(os.path.join(d, fn), os.W_OK):
                    ro.append(f"{rid}/{fn}")
        else:
            if st not in ("crashed", "invalid"):
                mism.append((rid, st, None, "no metrics.json", {}))
    comp[e] = {"ids_without_dir": sorted(set(last) - dirs), "dirs_without_id": sorted(dirs - set(last)),
               "status_mismatch": mism, "file_sets": {str(k): v for k, v in files.items()},
               "status_counts": dict(collections.Counter(r["status"] for r in last.values())),
               "writable_final_files": len(ro)}
out["completeness"] = comp

# rerun configs identical to originals (lost P4 + E3 stopped batches)
rr = []
for e in ["E1", "E3"]:
    g = {r["run_id"]: r["config"] for b in json.load(open(os.path.join(EX, e, "configs", "grid.json")))["batches"] for r in b["runs"]}
    fn = os.path.join(EX, e, "configs", "grid_reruns.json")
    if os.path.exists(fn):
        for b in json.load(open(fn))["batches"]:
            for r in b["runs"]:
                orig = r["run_id"][:-3]
                d = {k: (r["config"].get(k), g[orig].get(k)) for k in set(r["config"]) | set(g[orig]) if r["config"].get(k) != g[orig].get(k)}
                rr.append((e, r["run_id"], d))
out["rerun_config_diffs"] = [x for x in rr if x[2]]
out["n_reruns_checked"] = len(rr)

# lost P4 dir: never referenced by analysis code
refs = []
for f in glob.glob(os.path.join(EX, "*", "analysis", "*.py")) + glob.glob(os.path.join(EX, "common", "*.py")):
    t = open(f).read()
    if "lost_P4" in t: refs.append(os.path.relpath(f, ROOT) + ":lost_P4")
    if "E0_tuning/runs" in t or "E0_tuning', 'runs" in t or '"E0_tuning", "runs"' in t: refs.append(os.path.relpath(f, ROOT) + ":E0_tuning/runs")
out["analysis_refs_to_lost_or_tuning"] = refs
lost = []
for dp, dn, fn in os.walk(os.path.join(EX, "E1", "lost_P4_unverified")):
    for f in fn: lost.append(os.path.relpath(os.path.join(dp, f), EX))
out["lost_P4_files"] = len(lost)
json.dump(out, open(os.path.join(ROOT, "audit", "provenance.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:12000])
