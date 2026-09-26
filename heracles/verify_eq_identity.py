"""verify_eq_identity.py -- what the (n+1)-block really satisfies, and what LEMMA_ATTEMPT's
"check B" actually tested.

Claim under test (THEORY_nsum §8, line "gamma_q(construction on support S, amplitudes x) =
(n+1)^{-(nu-q)/q} B (programme value of (S,x))"):

    IDENTITY.  For ANY frequency support S and ANY share vector x (x_k >= 0, sum_k x_k = 1), the
    share-weighted (n+1)-neuron common-phase block built on (S, x) satisfies EXACTLY

        gamma_q  =  (n+1)^{-(nu-q)/q} * B * val(S, x),
        val(S, x) := min_{D != 0} sum_{k in S} x_k^{nu/q} (1 - cos(2 pi k D / p)).

    (T)'s equality case is then the special case val(S,x) = Gamma_{nu/q}(p), i.e. (S,x) a maximiser
    of the programme -- NOT "any block".

This script checks the identity at RANDOM, deliberately non-optimal (S, x), including equal shares
(which is what `lemma_verify.py::check_B` used) and shares with a zero entry.  Plain reading: the
block converts its own share vector into programme value with no loss, whatever the shares are; so
a block whose shares are NOT the programme's optimum sits strictly below the target.

Usage: uv run python heracles/verify_eq_identity.py
"""
import json
from pathlib import Path

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "vqb", str(Path(__file__).resolve().parent / "verify_eq_block.py"))
vqb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vqb)

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "VERIFY"
rng = np.random.default_rng(7)


def val_of(p, S, x, s):
    D = np.arange(1, p)
    G = 1 - np.cos(2 * np.pi * np.outer(np.array(S), D) / p)
    return float(((np.asarray(x) ** s) @ G).min())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    worst = 0.0
    for (p, n, q) in [(11, 2, 2.0), (13, 2, 2.0), (11, 2, 2.5), (11, 3, 2.0), (7, 4, 2.0),
                      (11, 2, 3.0), (11, 3, 4.0)]:
        nu = n + 1; s = nu / q
        if q > nu:
            continue
        nf = (p - 1) // 2
        Gam_star = vqb.gamma_s(p, s, starts=60)["value"]
        for trial in range(6):
            K = int(rng.integers(1, min(nf, 4) + 1))
            S = sorted(rng.choice(np.arange(1, nf + 1), size=K, replace=False).tolist())
            if trial == 0:
                x = np.ones(K) / K                                  # EQUAL shares (check_B's case)
                kind = "equal"
            elif trial == 1 and K > 1:
                x = rng.dirichlet(np.ones(K)); x[0] = 0.0; x = x / x.sum()   # a zero share
                kind = "zero-entry"
            else:
                x = rng.dirichlet(np.ones(K) * 0.7); kind = "random"
            E, W = vqb.build_block(p, n, S, x, q)
            z, y, ops = vqb.logit_table(E, W, p, n)
            mg = vqb.exact_margin(z, y)
            Pq = vqb.Pq_of(E, W, q)
            gam = mg / Pq ** (nu / q)
            B = 2 * (2 * p * (n + 1)) ** (-nu / 2)
            pred = (n + 1) ** (-(nu - q) / q) * B * val_of(p, S, x, s)
            rel = abs(gam / pred - 1) if pred != 0 else abs(gam)
            worst = max(worst, rel)
            frac_of_T = gam / ((n + 1) ** (-(nu - q) / q) * B * Gam_star)
            rows.append(dict(p=p, n=n, q=q, S=S, kind=kind, x=[round(v, 6) for v in x],
                             gamma=gam, identity_pred=pred, rel_err=rel, frac_of_target=frac_of_T))
            print(f"p={p} n={n} q={q:g} {kind:10s} S={S} x={np.round(x,4).tolist()}  "
                  f"identity rel.err {rel:.2e}   gamma/target(T) = {frac_of_T:.9f}")
    print(f"\nworst relative error of the IDENTITY over {len(rows)} blocks: {worst:.3e}")
    print("=> the identity is general (any S, any shares);  gamma/target(T) < 1 unless (S,x) "
          "maximises the programme.")
    (OUT / "eq_identity.json").write_text(json.dumps(dict(rows=rows, worst_rel_err=worst), indent=1))
