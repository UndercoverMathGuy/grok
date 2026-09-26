"""Render the NUMBERS sections of each experiment's REPORT (REPORT_numbers.md) from analysis/summary.json.
Every number printed here comes from the named analysis script. Usage: python experiments/common/write_reports.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import alib
f = alib.fmt


def load(exp):
    fn = os.path.join(ROOT, "experiments", exp, "analysis", "summary.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


def e1():
    S = load("E1"); L = []
    if not S: return "no summary"
    L.append(f"Source: `{S['script']}` -> experiments/E1/analysis/summary.json, cells.csv, per_run.csv.\n")
    L.append(f"Run statuses (all launched E1 runs): {S['status_counts']}\n")
    L.append("### Per cell (primary = status ok; 'nc' = not converged with margin > 0, shown when a cell has no ok run)\n")
    L.append("| nu | p | s | q | K* | label | runner-up gap | n ok / nc / other | grp | median K(thr10) | frac K=K* | frac within 1 | median TV(w,w*) | median cos(yhat,y*) | median R2 id (amp) | median R2 fit (amp) | median Spearman (share) | median Jaccard10 | median gamma/universal | median gamma/class-IF | median junk frac | cell supported |")
    L.append("|" + "---|" * 22)
    for d in S["cells"]:
        g = "ok" if d["n_ok"] else ("nc" if d["n_not_converged_pos_margin"] else None)
        if g is None:
            L.append(f"| {d['nu']} | {d['p']} | {d['s']} | {d['q']} | {d['K_star']} | {d['label'][:18]} | {f(d['runner_up_rel_gap'], 2)} | 0 / 0 / {d['n_other']} | - |" + " - |" * 12 + " no usable run |")
            continue
        L.append(f"| {d['nu']} | {d['p']} | {d['s']} | {d['q']} | {d['K_star']} | {d['label'][:18]} | {f(d['runner_up_rel_gap'], 2)} | {d['n_ok']} / {d['n_not_converged_pos_margin']} / {d['n_other']} | {g} | "
                 f"{f(d[g+'_median_thr10'])} | {f(d[g+'_frac_thr10_exact'], 2)} | {f(d[g+'_frac_thr10_within1'], 2)} | {f(d[g+'_median_share_tv'])} | {f(d[g+'_median_amp_cosine'])} | "
                 f"{f(d[g+'_median_amp_r2_identity'])} | {f(d[g+'_median_amp_r2_linear_fit'])} | {f(d[g+'_median_share_spearman'])} | {f(d[g+'_median_jaccard_thr_10'])} | "
                 f"{f(d[g+'_median_gamma_over_universal_product'])} | {f(d[g+'_median_gamma_over_class_IF'])} | {f(d[g+'_median_energy_junk'])} | {d['supported'] if g == 'ok' else 'n/a (no ok run)'} |")
    for name in ("ok", "nc"):
        sw = S.get(f"sweep_{name}")
        if not sw: continue
        L.append(f"\n### Sweep level, group {name} (n = {sw['n']} runs): measured count vs K*, EVERY definition (core.metrics.sweep_summary)\n")
        L.append("| definition | frac exact | frac within 1 | R2 identity | R2 fitted | Pearson | Spearman | Kendall | OLS slope [95% CI] | mean signed err |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in sw["counts"].items():
            lf = v["linfit_meas_on_pred"]
            L.append(f"| {k} | {f(v['frac_exact'], 2)} | {f(v['frac_within_tol'], 2)} | {f(v['r2_identity'])} | {f(v['r2_linear_fit'])} | {f(v['pearson'])} | {f(v['spearman'])} | {f(v['kendall'])} | {f(lf['slope'])} [{f(lf['slope_ci'][0])}, {f(lf['slope_ci'][1])}] | {f(v['mean_signed_error'])} |")
        L.append(f"\nPer-run metric distribution (median [10th, 90th pct]), group {name}:\n")
        for k, v in sw["per_run_metric_distribution"].items():
            if isinstance(v, dict):
                L.append(f"- {k}: {f(v['median'])} [{f(v['q10'])}, {f(v['q90'])}] (n = {v['n']})")
        L.append(f"- fraction of runs matching (K(thr10) = K* and TV <= 0.10 and cosine >= 0.95): {f(sw['per_run_metric_distribution']['frac_run_match'])}")
        L.append(f"- max gamma / universal ceiling over runs: {f(sw['per_run_metric_distribution']['max_gamma_over_universal'], 6)} (theorem: <= 1)")
    L.append("\n### E1b: equal s across nu (sorted budget-share vectors; TV)\n")
    L.append("| p | s | nus (n per nu) | within-s TV (across nu) | between-s TV (same nu) | within-cell TV (seeds) | ratio | within-s mean abs dK | between-s mean abs dK | supported (ratio < 0.5) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in S["E1b"]:
        L.append(f"| {e['p']} | {e['s']} | {e['n_per_nu']} | {f(e['within_s_tv'])} | {f(e['between_s_tv'])} | {f(e['within_cell_tv'])} | {f(e['ratio'])} | {f(e['within_s_absdK'])} | {f(e['between_s_absdK'])} | {e['supported']} |")
    if "controls" in S:
        L.append(f"\n### Controls (experiments/common/controls.py -> analysis/controls.json)\n\nAll positive controls perfect: **{S['controls_all_positive_perfect']}**\n")
        L.append("| nu | p | s | K* | pos K(thr10) | pos TV | pos gamma/class-IF | null-init K(thr10) | null-init TV | null-init cos | random-alloc median TV | random-alloc median cos | random-alloc frac K exact |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in S["controls"]:
            L.append(f"| {c['nu']} | {c['p']} | {c['s']} | {c['K_star']} | {c['pos_thr10']} | {f(c['pos_tv'], 2)} | {f(c['pos_gamma_over_IF'], 8)} | {c['null_init_thr10']} | {f(c['null_init_tv'])} | {f(c['null_init_cos'])} | {f(c['null_rand_tv_median'])} | {f(c['null_rand_cos_median'])} | {f(c['null_rand_thr10_exact_frac'])} |")
    return "\n".join(L)


def generic(exp):
    S = load(exp)
    if not S: return "no summary"
    return f"Source: `{S.get('script')}` -> experiments/{exp}/analysis/summary.json\n\n```json\n" + json.dumps(S, indent=1, default=str)[:60000] + "\n```"


if __name__ == "__main__":
    for exp, fn in (("E1", e1), ("E3", lambda: generic("E3")), ("E4", lambda: generic("E4")), ("E5", lambda: generic("E5")),
                    ("E6", lambda: generic("E6")), ("E7", lambda: generic("E7")), ("E8", lambda: generic("E8"))):
        txt = fn()
        open(os.path.join(ROOT, "experiments", exp, "REPORT_numbers.md"), "w").write(f"# {exp} — numbers (generated by experiments/common/write_reports.py)\n\n" + txt + "\n")
        print(exp, len(txt))
