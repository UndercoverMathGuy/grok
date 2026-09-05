"""Gold-check + polish of the constructive K* nets.

1. Materialise the product-grid construction (construct_gamma.py) as actual
   trainer-layout tensors (d=r=128, zero-padded), optimally block-balanced,
   and recompute gamma with train_nudial's own forward/margin code — must
   reproduce construct_gamma.json.
2. Polish the construction with the verified-monotone ascent
   (margin_polish.polish) — the resulting gamma is the best known lower bound
   on the true l2 optimum at each K.

Usage: uv run python nudial/construct_check.py [--steps 300]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from construct_gamma import committee_net, SETS  # noqa: E402
from margin_polish import polish  # noqa: E402
from train_nudial import Config  # noqa: E402

torch.set_grad_enabled(False)


def to_trainer(S, amps, m, n_a, nu):
    E, U, V, W1, W = committee_net(S, amps, m, n_a, n_a)
    d = r = 128
    K = len(S)
    blocks = {"U": np.zeros((m, d)), "V": np.zeros((m, d)),
              "W1": np.zeros((r, m)), "W": np.zeros((113, r))}
    if nu == 4:
        chain = {"E1": np.zeros((d, 113))}
        chain["E1"][:2 * K] = E
        blocks["U"][:, :2 * K] = U; blocks["V"][:, :2 * K] = V
        blocks["W1"][:2 * K] = W1; blocks["W"][:, :2 * K] = W
        groups = [["E1"], ["U", "V"], ["W1"], ["W"]]
        pr = {**chain, **blocks}
    else:  # nu=6, layout (2,2): E1 (d,113), E2 (d,d) id, W1, W2 (r,r) id, W
        pr = {"E1": np.zeros((d, 113)), "E2": np.zeros((d, d)),
              **blocks, "W2": np.zeros((r, r))}
        pr["E1"][:2 * K] = E
        pr["E2"][:2 * K, :2 * K] = np.eye(2 * K)
        pr["U"][:, :2 * K] = U; pr["V"][:, :2 * K] = V
        pr["W1"][:2 * K] = W1
        pr["W2"][:2 * K, :2 * K] = np.eye(2 * K)
        pr["W"][:, :2 * K] = W
        groups = [["E1"], ["E2"], ["U", "V"], ["W1"], ["W2"], ["W"]]
    # optimal balance: alpha_i^2 c_i equal, prod alpha = 1
    costs = np.array([sum(float((pr[k] ** 2).sum()) for k in g) for g in groups])
    al = (costs.prod() ** (1 / len(costs)) / costs) ** 0.5
    for g, a in zip(groups, al):
        for k in g:
            pr[k] = pr[k] * a
    return {k: torch.from_numpy(v).double().unsqueeze(0) for k, v in pr.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    ref = json.loads((ROOT / "notes" / "construct_gamma.json").read_text())
    out = {}
    for nu in (4, 6):
        lay = (1, 1) if nu == 4 else (2, 2)
        cfg = Config(family="relu", nu=nu, m=512, layout_pre=lay[0],
                     layout_post=lay[1], frac_train=1.0)
        out[nu] = {}
        for K in (3, 4, 5) if nu == 4 else (2, 3, 4):
            r = ref[str(nu)][str(K)]
            amps = np.array(r["amps"])
            pr = to_trainer(SETS[K], amps, 512, r["n_a"], nu)
            g0, g1, keff, Kg = polish(pr, cfg, nu, args.steps,
                                      label=f"relu{nu}-K{K}")
            print(f"relu-{nu} K={K}: gold-check gamma={g0[0]:.3e} "
                  f"(construct_gamma said {r['gamma_free']:.3e}) "
                  f"-> polished {g1[0]:.3e}  K_gap={Kg[0]} k_eff={keff[0]:.2f}")
            out[nu][K] = dict(gold=g0[0], polished=g1[0], K_gap=Kg[0])
    (ROOT / "notes" / "construct_polish.json").write_text(json.dumps(out, indent=1))
    print("wrote notes/construct_polish.json")


if __name__ == "__main__":
    main()
