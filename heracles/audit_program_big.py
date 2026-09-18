"""audit_program_big.py -- free-amplitude programme at the paper's primes, warm-started from the kstar.json K-sets
(K = 1..12) plus random starts. Reports Gamma_free / max_K M_K K^{-s} and the support of the free optimum.
A LOWER bound on Gamma_free (local search), which is all that is needed to show the equal-amplitude formula is
not the bound.   uv run python heracles/audit_program_big.py"""
import json, sys
import numpy as np
from audit_program import free

d = json.load(open(sys.path[0] + "/kstar.json"))
out = {}
for p in ("71", "113"):
    e = d[p]
    for s, key in ((1.5, "nu3_q2"), (2.0, "nu4_q2"), (2.5, "nu5_q2")):
        gh = {int(K): v for K, v in e["pred"][key]["gamma_hat"].items()}
        Ks = max(gh, key=gh.get)
        starts = [e["S"][str(K)] for K in range(1, 13)]
        val, x = free(int(p), s, starts, seeds=1500, steps=2500)
        supp = [(k + 1, round(float(v / x.max()), 2)) for k, v in enumerate(x) if v > 1e-3 * x.max()]
        big = sum(1 for _, v in supp if v > 0.2)
        out[f"p{p}_s{s}"] = dict(K_equal=Ks, gamma_equal=gh[Ks], gamma_free=float(val), K_free=len(supp), supp=supp)
        print(p, s, "equal K*", Ks, f"{gh[Ks]:.5f}", "| free >=", f"{val:.5f}", f"x{val / gh[Ks]:.4f}", "K_free", len(supp), f"(>{0.2} of max: {big})", supp, flush=True)
json.dump(out, open(sys.path[0] + "/audit_program_big.json", "w"), indent=1)
