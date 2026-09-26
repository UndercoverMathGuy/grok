"""Builds the anonymized supplementary material for the paper.

Writes OUT (default ../heracles/paper/iclr2027_supplementary) containing
  results.json      every number quoted in the paper, each with the file it was read from or the code that computed it,
                    plus per-run and per-cell tables;
  experiments/      per experiment: PREREG.md, REPORT*.md, AMENDMENTS.md, predictions.json, RUNS.csv, configs/,
                    and runs/<run_id>/{config.json, metrics.json, log.csv, final.pt} (final.pt = trained weights);
  code/             core/, experiments/common/*.py, experiments/*/analysis/*.py, paper_figs/, supplement/,
                    pyproject.toml, uv.lock, NORMS.md;
  audit/            AUDIT.md and the retraining comparison;
  README.md.
Text files are scrubbed of local paths and host names; final.pt files are checked to contain neither.
LOG.md and COMPUTE.md files are not included (they carry local timestamps and account details).
Usage: uv run python supplement/build_supplement.py [OUT]
"""
import csv, glob, json, os, re, shutil, sys, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))
import numpy as np
from core import programme as PG
import alib

OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(ROOT, "..", "heracles", "paper", "iclr2027_supplementary"))
EXPS = ["E0_tuning", "E1", "E3", "E4", "E6", "E7", "E8"]
SCRUB = [(re.compile(r"/Users/[A-Za-z0-9_.-]+"), "/home/anon"), (re.compile(r"[A-Za-z0-9-]*MacBook[A-Za-z0-9.-]*"), "local-machine"),
         (re.compile(r"ruhaan[a-z]*", re.I), "anon"), (re.compile(r"rajadhyaksha", re.I), "anon"), (re.compile(r"sabeera[a-z]*", re.I), "anon")]
BAD = [b"/Users/", b"MacBook", b"ruhaan", b"Ruhaan", b"Sabeera", b"rajadhyaksha"]
TEXT_EXT = {".py", ".md", ".json", ".csv", ".sh", ".toml", ".txt", ".lock", ".log", ".cfg", ".yaml", ".yml"}


def rel(p):
    return os.path.relpath(p, ROOT)


def load_json(p):
    return json.load(open(os.path.join(ROOT, p)))


def csv_rows(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p))))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ============================================================ results.json
def theory_checks():
    out = {}
    # dense window: largest s at which the programme optimum uses all (p-1)/2 frequencies (vertex cache, exact)
    dw = {}
    for p in (7, 11, 13, 17, 19, 23):
        reps, _, _ = PG.load_vertices(p)
        h = (p - 1) // 2; full = (reps > 1e-12).sum(1) == h
        def dense_opt(s):
            c = PG.cost(reps, s); val = PG.coverage_min(reps, p) / c ** s
            return val[full].max() >= val.max() * (1 - 1e-12)
        lo, hi = 1.0 + 1e-9, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dense_opt(mid) else (lo, mid)
        dw[str(p)] = round(lo, 4)
    out["dense_window_s_max"] = {"value": dw, "paper": "Sec. 4, 'How the count grows with p' (1.548, 1.345, 1.285, 1.217, 1.194, 1.160)",
                                 "computed_by": "supplement/build_supplement.py theory_checks (core.programme vertex cache, exact)"}
    # nu-block identity (Prop. 9)
    errs = {}
    for n in (1, 2, 3, 4):
        nu = n + 1; p = 11; j = 3; lam = 0.7; al = np.pi * np.arange(nu) / nu; err = 0.0
        for x in itertools.product(range(p), repeat=n):
            for c in range(p):
                tot = sum(np.prod([lam * np.cos(2 * np.pi * j * x[t] / p + al[l]) for t in range(n)]) * lam * np.cos(2 * np.pi * j * c / p + n * al[l]) for l in range(nu))
                err = max(err, abs(tot - nu * lam ** nu * 2.0 ** (-n) * np.cos(2 * np.pi * j * (sum(x) - c) / p)))
        errs[f"n={n}"] = err
    out["nu_block_identity_max_abs_error_p11_j3"] = {"value": errs, "paper": "Prop. 9", "computed_by": "supplement/build_supplement.py"}
    p = 97; Bp = 2 * (2 * p * 3) ** (-1.5)
    out["morwani_constant_p97"] = {"value": {"2Bp_p_over_p-1": 2 * Bp * p / (p - 1), "sqrt(2/27)/(p^0.5 (p-1))": np.sqrt(2 / 27) / (p ** 0.5 * (p - 1))},
                                   "paper": "Appendix A.4 (Corollary 14)", "computed_by": "supplement/build_supplement.py"}
    out["class_value_factor_nu^((nu-b)/b)_b2"] = {"value": {str(nu): nu ** ((nu - 2) / 2) for nu in (3, 4, 5)}, "paper": "Sec. 4 (sqrt3, 4, 5^1.5)"}
    # Table 1: allocations from the earlier support search, re-evaluated here; scaling bounds; dense ratios
    T = load_json("experiments/common/table1_allocations.json")
    rows = []
    for c in T["cells"]:
        p, s = c["p"], c["s"]; h = (p - 1) // 2
        y = np.zeros(h); y[np.array(c["S"]) - 1] = np.array(c["b"]) ** s
        G = PG.programme_value(y, p, s)
        a = 3 + s / 2; lnp = np.log(p)
        rows.append({"p": p, "nu": c["nu"], "s": s, "K_star": c["K_star_free"], "Gamma_best_found": G, "Gamma_reported": c["val_reported"],
                     "support": c["S"], "budget_shares": c["b"],
                     "scaling_bounds": {"lower": 0.5 * np.ceil(8 * lnp) ** (1 - s), "upper": 1.5 * (2 * a) ** s * lnp ** (1 - s)},
                     "dense_network_over_optimum": (p / 2) * h ** (-s) / G, "dense_over_equal_share_optimum": c["dense_ratio_eq_reported"]})
    out["table1"] = {"value": rows, "paper": "Table 1 and 'How many features' (0.49-0.60, 0.15-0.23, 0.04-0.08)",
                     "source": "experiments/common/table1_allocations.json (earlier support search; best found, not certified), re-evaluated with core.programme.programme_value"}
    return out


def experiments_numbers():
    E = {}
    A1 = {e: [dict((k, num(v)) for k, v in r.items()) for r in csv_rows(f"experiments/{e}/analysis/A1_per_run.csv")] for e in ("E1", "E6", "E7", "E8")}
    E3 = [dict((k, num(v)) for k, v in r.items()) for r in csv_rows("experiments/E3/analysis/per_run.csv")]
    A2 = load_json("experiments/common/A2_summary.json"); A3 = load_json("experiments/common/A3_bestknown.json")
    ok = lambda R: [r for r in R if r["status"] == "ok"]
    # 5.1 clean energy
    allok = ok(A1["E1"]) + ok(A1["E6"]) + ok(A1["E7"]) + ok(A1["E8"])
    ce = np.array([r["energy_clean"] for r in allok])
    E["clean_energy"] = {"value": {"median": float(np.median(ce)), "min": float(ce.min()), "n_runs": len(ce),
                                   "by_nu_s": A2["ok"]["T4_clean_vs_s"]}, "paper": "Sec. 5.1; Appendix E", "source": "experiments/*/analysis/A1_per_run.csv (energy_clean); experiments/common/A2_summary.json (T4)"}
    E["allocation_quality"] = {"value": {"E1_median": A3["E1_eff_median"], "E1_frac_ge_0.95": A3["E1_eff_frac_ge_0.95"],
                                         "null_random_allocation_median": A3["E1_null_random_allocation_median"], "null_init_median": A3["E1_null_random_init_median"],
                                         "E1_median_ci_over_seeds_before_A3": load_json("experiments/E1/analysis/A1_summary.json")["ok"]["efficiency"]["median_ci_over_seeds"],
                                         "by_nu": A3["same_s_signed"]["E1_eff_median_by_nu"]},
                               "paper": "Sec. 5.1; Appendix E", "source": "experiments/common/A3_bestknown.json (A1 efficiency with the A3 best-known optimum)"}
    ctrl = load_json("experiments/E1/analysis/summary.json")["controls"]
    tv = [r["share_tv"] for r in (dict((k, num(v)) for k, v in x.items()) for x in csv_rows("experiments/E1/analysis/per_run.csv")) if r["status"] == "ok"]
    E["budget_shares_vs_ystar"] = {"value": {"median_tv": float(np.median(tv)), "null_random_allocation_median_tv": float(np.median([c["null_rand_tv_median"] for c in ctrl]))},
                                   "paper": "Sec. 5.1; Appendix E (preregistered)", "source": "experiments/E1/analysis/per_run.csv (share_tv), summary.json (controls)"}
    # universal bound
    rat = {}; seen = set()
    for e in ("E1", "E7", "E8"):  # E8's analysis table also lists E1's nu=3 runs: count each run id once
        for r in csv_rows(f"experiments/{e}/analysis/per_run.csv"):
            g = num(r.get("gamma_over_universal_product"))
            if isinstance(g, float) and np.isfinite(g) and r["run_id"] not in seen:
                seen.add(r["run_id"]); rat.setdefault(e, []).append((g, float(r["s"])))
    for r in E3:
        g = r.get("gamma_over_ceiling_model")
        if isinstance(g, float) and np.isfinite(g):
            rat.setdefault("E3", []).append((g, 1.0))
    e6 = []
    for run in alib.load("E6"):
        m = run.get("metrics", {}).get("measurement") if run.get("metrics") else None
        if m and np.isfinite(m.get("gamma", np.nan)):
            e6.append(m["gamma"])
    G6 = PG.gamma(23, 1.5); B = 2 * (2 * 23 * 3) ** (-1.5)
    rat["E6"] = [(g / (B * G6), 1.5) for g in e6]
    allr = [x for v in rat.values() for x in v]
    E["universal_bound"] = {"value": {"n_runs_finite_gamma": len(allr), "max_ratio_b_lt_nu": max(g for g, s in allr if s > 1 + 1e-9),
                                      "max_ratio_b_eq_nu": max(g for g, s in allr if s <= 1 + 1e-9), "any_exceeds": any(g > 1 + 1e-9 for g, _ in allr),
                                      "by_experiment_max": {k: max(g for g, _ in v) for k, v in rat.items()}},
                            "paper": "Sec. 5.1; Appendix E", "source": "per_run.csv gamma_over_universal_product / gamma_over_ceiling_model; E6 from run metrics"}
    # 5.2 fine sweep + direction test
    e7 = load_json("experiments/E7/analysis/summary.json")
    e7ok = ok(A1["E7"])
    E["fine_sweep_nu4_p37"] = {"value": {"n_converged": len(e7ok), "n_not_converged": len([r for r in A1["E7"] if r["status"] == "not_converged"]),
                                         "converged_s1_count": [r["thr_20"] for r in e7ok if abs(r["s"] - 1) < 1e-9],
                                         "all_converged_s_ge_1.45_within_1": all(abs(r["thr_20"] - r["K_star"]) <= 1 for r in e7ok if r["s"] >= 1.45),
                                         "plateau_identity_matches": round(e7["ok"]["frac_plateau_match"] * e7["ok"]["n"]), "plateau_identity_n": e7["ok"]["n"]},
                               "paper": "Sec. 5.2, Figure 1a; Appendix E", "source": "experiments/E7/analysis/A1_per_run.csv, summary.json"}
    E["direction_test"] = {"value": {"E1_E8_with_A3": A3["T1_E1E8"], "E1_E8_all_trained_runs_thr20": A3.get("T1_all_trained_thr20"), "E7": {k: A2["ok"][k]["T1_qsweep_E7"] for k in A3["T1_E1E8"]},
                                     "pooled_within_1_thr20_with_A3": A3["pooled_E1_thr20_within1"], "pooled_best_constant": A3["pooled_E1_best_constant_within1"]},
                           "paper": "Sec. 5.2 (30 of 36); Appendix E table", "source": "experiments/common/A3_bestknown.json, A2_summary.json (post hoc)"}
    # dense end
    d = [r for r in E3 if r["status"] == "ok"]
    E["dense_end_b_eq_nu"] = {"value": [{k: r[k] for k in ("model", "nu", "p", "seed", "cnt_thr_10", "cnt_thr_20", "flat_min_over_max", "tv_to_uniform", "gamma_over_ceiling_model")} for r in d],
                              "paper": "Sec. 5.2; Appendix E", "source": "experiments/E3/analysis/per_run.csv"}
    # 5.3
    E["same_s_across_degree"] = {"value": A3["same_s_signed"], "paper": "Sec. 5.3; Appendix E", "source": "experiments/common/A3_bestknown.json (post hoc)"}
    E["degree_at_b2"] = {"value": {k: A2["ok"][k]["T2_nu_ladder_q2"] for k in A2["ok"] if isinstance(A2["ok"][k], dict) and "T2_nu_ladder_q2" in A2["ok"][k]},
                         "paper": "Appendix E", "source": "experiments/common/A2_summary.json (post hoc)"}
    E["across_primes_nu3"] = {"value": load_json("paper_figs/FIGURE_NUMBERS.json")["fig2"], "paper": "Figure 2; Sec. 5.3", "source": "paper_figs/FIGURE_NUMBERS.json (A3 applied)"}
    E["ablation_bound_E4"] = {"value": load_json("experiments/E4/analysis/summary.json")["ok"], "paper": "not in the main text (Lemma 19 is used in the proof of Theorem 11)", "source": "experiments/E4/analysis/summary.json (preregistered)"}
    E["relabelling_E6"] = {"value": load_json("experiments/E6/analysis/summary.json")["pairs_summary"], "paper": "not in the main text", "source": "experiments/E6/analysis/summary.json (preregistered)"}
    return E


def appendix_numbers():
    A = {"recipe": load_json("experiments/common/RECIPE.json")}
    st = {}
    for e in ("E1", "E3", "E6", "E7", "E8"):
        c = {}
        for r in alib.runs_status(e).values():
            c[r["status"]] = c.get(r["status"], 0) + 1
        st[e] = c
    A["run_status_counts"] = {"value": st, "source": "experiments/*/RUNS.csv (last status per run id)"}
    return A


def tables():
    per_run = []
    for e in ("E1", "E6", "E7", "E8"):
        per_run += [dict((k, num(v)) for k, v in r.items()) for r in csv_rows(f"experiments/{e}/analysis/A1_per_run.csv")]
    e3 = [dict((k, num(v)) for k, v in r.items()) for r in csv_rows("experiments/E3/analysis/per_run.csv")]
    for r in e3:
        r["exp"] = "E3"
    cells = {}
    for r in per_run:
        if r["status"] != "ok":
            continue
        k = (r["exp"], int(r["nu"]), int(r["p"]), round(r["s"], 6))
        cells.setdefault(k, []).append(r)
    cell_rows = [{"exp": k[0], "nu": k[1], "p": k[2], "s": k[3], "K_star_prediction": v[0]["K_star"], "n_converged": len(v),
                  **{f"median_{c}": float(np.median([x[c] for x in v])) for c in ("thr_20", "thr_10", "K_2pct", "N_05", "eff", "energy_clean")}}
                 for k, v in sorted(cells.items())]
    return {"per_run_A1": per_run, "per_run_E3": e3, "per_cell_converged": cell_rows,
            "note": "K_star_prediction is the preregistered prediction; Amendment A3 replaces it at (p=113, s=1.5) by 7 (and Gamma at p=59, s=2) - see results.experiments.direction_test and A3_bestknown.json."}


def build_results():
    t0 = time.time()
    R = {"meta": {"paper": "Explaining Feature Sparsity Through the Margin-Regularization Trade-off (anonymous submission)",
                  "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "generator": "supplement/build_supplement.py",
                  "conventions": "Every entry has 'value' and a 'source' (file in this supplement) or 'computed_by'. 'post hoc' = defined after the preregistered results were seen (see experiments/*/AMENDMENTS.md). Counts use A_j >= 0.2 max_k A_k unless stated. Converged runs only unless stated."},
         "theory_checks": theory_checks(), "experiments": experiments_numbers(), "appendix_E": appendix_numbers(), "tables": tables()}
    R["meta"]["seconds"] = round(time.time() - t0, 1)
    return R


# ============================================================ files
def scrub_text(s):
    for rx, rep in SCRUB:
        s = rx.sub(rep, s)
    return s


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.splitext(src)[1] in TEXT_EXT or os.path.basename(src) in ("uv.lock",):
        try:
            open(dst, "w").write(scrub_text(open(src).read()))
            return
        except UnicodeDecodeError:
            pass
    shutil.copy2(src, dst)


def build_files():
    n = 0
    for e in EXPS:
        base = os.path.join(ROOT, "experiments", e)
        for f in ("PREREG.md", "REPORT.md", "REPORT_A1.md", "REPORT_numbers.md", "AMENDMENTS.md", "predictions.json", "RUNS.csv", "q_grid.json", "reproduce.sh"):
            if os.path.exists(os.path.join(base, f)):
                copy(os.path.join(base, f), os.path.join(OUT, "experiments", e, f)); n += 1
        for sub in ("configs", "analysis"):
            for src in glob.glob(os.path.join(base, sub, "**", "*"), recursive=True):
                if os.path.isfile(src) and not src.endswith((".py", ".pyc")):
                    copy(src, os.path.join(OUT, "experiments", e, os.path.relpath(src, base))); n += 1
        for d in sorted(glob.glob(os.path.join(base, "runs", "*"))):
            for f in ("config.json", "metrics.json", "log.csv", "final.pt"):
                if f == "final.pt" and e == "E0_tuning":
                    continue  # tuning weights omitted for size (optimization quality only; no claim uses them)
                if os.path.exists(os.path.join(d, f)):
                    copy(os.path.join(d, f), os.path.join(OUT, "experiments", e, "runs", os.path.basename(d), f)); n += 1
    code = ["core/*.py", "core/RECIPES.md", "core/THEORY_SPEC.md", "core/data/*.npz", "core/data/*.json", "experiments/common/*.py", "experiments/common/*.json",
            "experiments/common/pred/*.json", "experiments/*/analysis/*.py", "paper_figs/*.py", "tests/*.py", "pyproject.toml", "uv.lock", "NORMS.md", "PLAN.md", "README.md"]
    for pat in code:
        for src in glob.glob(os.path.join(ROOT, pat)):
            copy(src, os.path.join(OUT, "code", rel(src))); n += 1
    for src in [os.path.join(ROOT, "AUDIT.md")] + glob.glob(os.path.join(ROOT, "audit", "**", "*"), recursive=True):
        if os.path.isfile(src) and "/runs/" not in src and not src.endswith((".pt", ".pyc")):
            copy(src, os.path.join(OUT, "audit", os.path.relpath(src, ROOT) if src == os.path.join(ROOT, "AUDIT.md") else os.path.relpath(src, os.path.join(ROOT, "audit")))); n += 1
    return n


def check_clean():
    hits = []
    for src in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
        if os.path.isfile(src):
            b = open(src, "rb").read()
            if any(x in b for x in BAD):
                hits.append(os.path.relpath(src, OUT))
    return hits


README = """# Supplementary material (anonymous)

- `results.json`: every number quoted in the paper, each with the file it was read from (paths relative to `code/` or
  `experiments/`) or the code that computed it; also per-run and per-cell tables. Section map:
  `theory_checks` (Sec. 4, Table 1, App. A.4), `experiments` (Sec. 5 and App. E), `appendix_E`, `tables`.
- `experiments/<E>/runs/<run_id>/final.pt`: trained weights (torch state dict, float32) of every run except E0 (tuning; omitted for size), with
  `config.json` (full recipe, seed, code hashes), `metrics.json` (all registered measurements, float64) and `log.csv`.
- `experiments/<E>/PREREG.md` (frozen before evaluation), `REPORT.md` (preregistered analysis), `AMENDMENTS.md`
  (post-hoc analyses A1-A3, with reasons), `RUNS.csv` (every launched run with its status; none omitted).
  E0 = tuning (optimization quality only), E1 = main sweep, E3 = b = nu, E4 = ablation bound, E6 = relabelling,
  E7 = fine sweep in b, E8 = sweep in p.
- `code/`: all code (`uv sync` then e.g. `uv run python experiments/common/amend_A2_shape.py`); `NORMS.md` lists the
  rules every run and analysis followed. `experiments/*/reproduce.sh` regenerate predictions, runs and analyses.
- `audit/`: independent audit of the empirical record, including re-measurement and retraining of sampled runs.
Local paths and host names were replaced by placeholders.
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    R = build_results()
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    n = build_files()
    open(os.path.join(OUT, "README.md"), "w").write(README)
    hits = check_clean()
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(f))
    print(f"wrote {OUT}: {n} files, {size / 1e6:.1f} MB; results.json {os.path.getsize(os.path.join(OUT, 'results.json')) / 1e6:.1f} MB")
    print("identifying strings remaining in:", hits[:20] if hits else "none")


if __name__ == "__main__":
    main()
