"""Print headline numbers of E1/E4/E5/E8 from the analysis summaries (for writing the REPORTs)."""
import json, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = lambda e: json.load(open(os.path.join(R, "experiments", e, "analysis", "summary.json")))
S = L("E1")
print("E1 status", S["status_counts"], "controls perfect", S.get("controls_all_positive_perfect"))
for g in ("ok", "nc"):
    sw = S.get(f"sweep_{g}")
    if not sw: continue
    print(f"E1 [{g}] n={sw['n']}")
    for k, v in sw["counts"].items():
        if "same" in k: continue
        print(f"   {k:14s} exact {v['frac_exact']:.3f} pm1 {v['frac_within_tol']:.3f} R2id {v['r2_identity']:.3f} R2fit {v['r2_linear_fit']:.3f} pear {v['pearson']:.3f} spear {v['spearman']:.3f} kend {v['kendall']:.3f} slope {v['linfit_meas_on_pred']['slope']:.3f} [{v['linfit_meas_on_pred']['slope_ci'][0]:.2f},{v['linfit_meas_on_pred']['slope_ci'][1]:.2f}] mse {v['mean_signed_error']:.2f}")
    d = sw["per_run_metric_distribution"]
    print("   dist", {k: (round(v["median"], 3), round(v["q10"], 3), round(v["q90"], 3)) for k, v in d.items() if isinstance(v, dict)})
    print("   frac_run_match", round(d["frac_run_match"], 3), "max g/univ", round(d["max_gamma_over_universal"], 4))
sup = [c for c in S["cells"] if c.get("supported")]
print("E1 cells supported", len(sup), "of", sum(1 for c in S["cells"] if c["n_ok"]), "with ok runs;", [(c["nu"], c["p"], c["s"]) for c in sup])
print("E1 cells without ok runs", [(c["nu"], c["p"], c["s"], c["n_ok"], c["n_not_converged_pos_margin"], c["n_other"]) for c in S["cells"] if not c["n_ok"]])
for e in S["E1b"]:
    print("E1b", e["p"], e["s"], e["n_per_nu"], "within", round(e["within_s_tv"], 3), "between", e["between_s_tv"] and round(e["between_s_tv"], 3), "noise", e["within_cell_tv"] and round(e["within_cell_tv"], 3), "ratio", e["ratio"] and round(e["ratio"], 3), "dK", round(e["within_s_absdK"], 2), e["between_s_absdK"] and round(e["between_s_absdK"], 2), e["supported"])
c = S.get("controls", [])
print("E1 null-init median TV", sorted(round(x["null_init_tv"], 2) for x in c)[len(c)//2] if c else None, "rand-alloc median TV (median over cells)", sorted(round(x["null_rand_tv_median"], 2) for x in c)[len(c)//2] if c else None,
      "rand-alloc frac K exact (median)", sorted(round(x["null_rand_thr10_exact_frac"], 2) for x in c)[len(c)//2] if c else None)
E4 = L("E4")
for g in ("ok", "nc"):
    d = E4.get(g)
    if d: print("E4", g, "runs", d["n_runs"], "active", d["active_w_ge_0.01"]["n_satisfied"], "/", d["active_w_ge_0.01"]["n_pairs"], round(d["active_w_ge_0.01"]["frac"], 4), "all-j", round(d["all_j"]["frac"], 4), "perm null", round(d["perm_null_frac_mean"], 3), "by_nu", {k: round(v["frac"], 3) for k, v in d["by_nu"].items() if v["n_pairs"]}, "excess", {k: round(v, 3) for k, v in d["excess_quantiles_(rel_loss-bound)"].items()}, "viol", d["active_w_ge_0.01"]["violations"])
print("E4 positive", E4.get("positive_controls", {}).get("n_satisfied"), "/", E4.get("positive_controls", {}).get("n_pairs"))
E5 = L("E5")
for g in ("ok", "not_converged"):
    d = E5.get(g)
    if d: print("E5", g, "n", d["n"], "match", round(d["frac_match"], 3), "n_near>=K", round(d["frac_n_near_0.05_ge_K"], 3), "jacc .05", round(d["median_jacc_0.05"], 3), "jacc .01", round(d["median_jacc_0.01"], 3), "n_near.01 vs n_tight: exact", round(d["n_near_0.01_vs_n_tight_pred"]["frac_exact"], 3), "mse", round(d["n_near_0.01_vs_n_tight_pred"]["mean_signed_error"], 2), "| n_near.05-K mse", round(d["n_near_0.05_minus_K"]["mean_signed_error"], 2))
print("E5 controls all match", E5.get("positive_controls_all_match"))
E8 = L("E8")
for k, v in E8.items():
    if isinstance(v, dict) and "per_p" in v:
        sw = v["sweep_thr10"]
        print("E8", k, "p-median match", round(v["frac_p_median_match"], 3), "exact", round(sw["frac_exact"], 3), "pm1", round(sw["frac_within_tol"], 3), "R2id", round(sw["r2_identity"], 3), "R2fit", round(sw["r2_linear_fit"], 3), "spear", round(sw["spearman"], 3), "K*/logp", [round(x, 2) for x in v["Kstar_over_logp_range"]])
        print("    ", [(q["p"], q["K_star"], q["median_thr10"], q["n"], round(q["median_tv"], 2)) for q in v["per_p"]])
