"""ladder_verify.py -- the gate everything else depends on.

Checks, by brute force over the FULL p^n x p logit table in fp64:
  (J) the share-weighted block is EXACTLY junk-free: its logit table equals
      sum_k A_k cos(2 pi k D / p) with D = (sum_j a_j) - c, to machine precision;
  (I) the identity  gamma_q = mu * N^{-(nu-q)/q} * B * val(S,b),  B = 2(2p(n+1))^{-nu/2};
  (M) the margin read off the p-1 gap vector equals the brute-force margin over the whole table;
  (P) the penalty P_q computed from the construction equals the one computed from the tensors.
Also re-checks ADVERSARY's code correction: equal amplitudes miss the target when the programme's
optimal shares are unequal (the 0.677 at p = 11).

Usage: uv run python heracles/ladder_verify.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import (OUT, block_spec, find_xn_block, gamma_con, gamma_con_predicted, gmat,  # noqa: E402
                         solve_support, val_of, verify_block)

CASES = [
    # (family, p, n, q, K)
    ("chain", 11, 2, 2.0, 2), ("chain", 11, 2, 2.0, 3), ("chain", 13, 2, 2.5, 3),
    ("chain", 13, 2, 3.0, 3), ("chain", 17, 2, 2.0, 4), ("chain", 23, 2, 2.0, 4),
    ("chain", 11, 3, 2.0, 2), ("chain", 11, 3, 2.0, 3), ("chain", 13, 3, 8 / 3, 3),
    ("chain", 13, 3, 4.0, 3), ("chain", 11, 4, 2.0, 2), ("chain", 13, 4, 10 / 3, 3),
    ("chain", 13, 4, 5.0, 2),
    ("xn", 11, 2, 2.0, 2), ("xn", 13, 2, 2.0, 3), ("xn", 13, 2, 3.0, 3),
    ("xn", 11, 3, 2.0, 2), ("xn", 13, 3, 2.0, 3), ("xn", 11, 3, 4.0, 2),
    ("xn", 11, 4, 2.0, 2), ("xn", 11, 4, 5.0, 2),
]


def main():
    res = dict(blocks={}, cases=[], equal_vs_weighted=[])
    print("== junk-free blocks")
    for n in (2, 3, 4):
        N, alpha = find_xn_block(n)
        Nc, _, _, muc = block_spec("chain", n)
        Nx, _, _, mux = block_spec("xn", n)
        res["blocks"][n] = dict(xn_N=Nx, xn_alpha=alpha, chain_N=Nc, mu_xn=mux, mu_chain=muc)
        print(f"  n {n}: chain N {Nc} mu {muc:g} | x^n N {Nx} alpha {alpha} mu {mux:g}")

    print("\n== brute-force verification (junk / identity / margin / penalty, relative)")
    worst = dict(junk=0.0, id=0.0, margin=0.0, Pq=0.0, brute=0.0)
    for fam, p, n, q, K in CASES:
        g = gmat(p)
        nf = (p - 1) // 2
        if K > nf:
            continue
        s = (n + 1) / q
        from ladder_core import best_K_support
        bk = best_K_support(g, K, s, shortlist=60, polish=False)
        S, b = bk["S"], bk["b"]
        out = verify_block(fam, p, n, S, b, q)
        out.update(family=fam, p=p, n=n, q=q, K=K, S=S, b=b, val=bk["value"])
        res["cases"].append(out)
        worst["junk"] = max(worst["junk"], out["junk_rel"])
        worst["id"] = max(worst["id"], out["id_rel"])
        worst["margin"] = max(worst["margin"], out["margin_rel"])
        worst["Pq"] = max(worst["Pq"], out["Pq_rel"])
        worst["brute"] = max(worst["brute"], out["brute_rel"])
    print(f"\n  WORST over {len(res['cases'])} cases: junk {worst['junk']:.3e}  identity {worst['id']:.3e}  "
          f"margin {worst['margin']:.3e}  P_q {worst['Pq']:.3e}  brute-vs-predicted {worst['brute']:.3e}")
    res["worst"] = worst

    print("\n== ADVERSARY's code correction: equal amplitudes vs the programme's shares")
    for p, n, q, K in [(11, 2, 2.0, 3), (11, 2, 2.0, 4), (59, 2, 2.0, 5), (71, 3, 2.0, 4)]:
        g = gmat(p)
        s = (n + 1) / q
        from ladder_core import best_K_support
        bk = best_K_support(g, K, s, shortlist=60, polish=False)
        S, b = bk["S"], bk["b"]
        beq = [1.0 / K] * K
        ge, _, _, _, _ = gamma_con("chain", p, n, S, beq, q)
        gw, _, _, _, _ = gamma_con("chain", p, n, S, b, q)
        tgt = gamma_con_predicted("chain", p, n, bk["value"], q)
        row = dict(p=p, n=n, q=q, K=K, S=S, b=b, gamma_equal=ge, gamma_weighted=gw, target=tgt,
                   ratio_equal=ge / tgt, ratio_weighted=gw / tgt)
        res["equal_vs_weighted"].append(row)
        print(f"  p {p} n {n} q {q:g} K {K}: equal/target {ge / tgt:.6f}   weighted/target {gw / tgt:.12f}   "
              f"b = {' '.join(f'{x:.3f}' for x in b)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(res, indent=1))
    print("\nwrote", OUT / "verify.json")


if __name__ == "__main__":
    main()
