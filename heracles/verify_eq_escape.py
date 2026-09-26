"""verify_eq_escape.py -- is the share-weighted block a strict local maximum of the ratio?

This is the sharpest test of the EQUALITY claim that does not need a proof: perturb the block by
noise of relative size eps, re-run the full annealed ascent, and see whether anything ever comes
back ABOVE 1.  (Cold random starts mostly test the optimiser's basin-finding; this tests the
geometry at the claimed optimum.)  Two extra probes:

  * "purify-off": the perturbation deliberately makes neurons IMPURE (adds a second frequency),
    since ADVERSARY §5 showed impurity can buy margin -- so that is where an escape would live;
  * "junk-on": the perturbation deliberately injects same-|k| junk by rotating one neuron's
    operand phase away from the common-phase pattern.

ratio > 1 anywhere = counterexample to (T).  Written independently of adv_net.py / lemma_ascent.py.

Usage: uv run python heracles/verify_eq_escape.py
"""
import json
from pathlib import Path

import numpy as np
import torch

import importlib.util
_p = Path(__file__).resolve().parent
_s1 = importlib.util.spec_from_file_location("vqb", str(_p / "verify_eq_block.py"))
vqb = importlib.util.module_from_spec(_s1); _s1.loader.exec_module(vqb)
_s2 = importlib.util.spec_from_file_location("vqa", str(_p / "verify_eq_ascent.py"))
vqa = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(vqa)

OUT = _p / "FINAL_RUNS" / "VERIFY"
torch.set_default_dtype(torch.float64)


def perturb(E, W, kind, eps, rng, p, n):
    E2, W2 = E.copy(), W.copy()
    sc = np.sqrt((E ** 2).mean())
    if kind == "gauss":
        E2 = E + eps * sc * rng.normal(size=E.shape)
        W2 = W + eps * sc * rng.normal(size=W.shape)
    elif kind == "impure":           # add a second frequency to every neuron
        a = np.arange(p) * 2 * np.pi / p
        nf = (p - 1) // 2
        for i in range(E.shape[1]):
            k2 = int(rng.integers(1, nf + 1))
            for j in range(n):
                E2[j, i] = E[j, i] + eps * sc * np.cos(k2 * a + rng.uniform(0, 2 * np.pi))
            W2[:, i] = W[:, i] + eps * sc * np.cos(k2 * a + rng.uniform(0, 2 * np.pi))
    elif kind == "junk":             # break the common-phase pattern on operand 0 of one neuron
        i = int(rng.integers(0, E.shape[1]))
        th = eps * np.pi
        c, s = np.cos(th), np.sin(th)
        v = E2[0, i].copy()
        hil = np.imag(np.fft.ifft(np.fft.fft(v) * np.concatenate(
            [[0], -1j * np.ones(p // 2), 1j * np.ones(p - 1 - p // 2)])))  # Hilbert-type rotation
        E2[0, i] = c * v + s * hil
    return E2, W2


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows, gmax, viol = [], -np.inf, 0
    rng = np.random.default_rng(11)
    for (p, n, q) in [(11, 2, 2.0), (13, 2, 2.0), (17, 2, 2.0), (11, 3, 2.0), (11, 2, 2.5)]:
        nu = n + 1
        prog = vqb.gamma_s(p, nu / q, starts=80)
        rhs = vqb.target_rhs(p, n, q, prog["value"])
        keep = [i for i, b in enumerate(prog["b"]) if b > 1e-9]
        S = [prog["S"][i] for i in keep]
        b = np.array([prog["b"][i] for i in keep]); b = b / b.sum()
        E, W = vqb.build_block(p, n, S, b, q)
        r0 = vqa.brute_ratio(E, W, p, n, q, rhs)
        ops, y = vqa.rows_of(p, n)
        cell_max = r0
        for kind in ("gauss", "impure", "junk"):
            for eps in (1e-4, 1e-2, 0.1, 0.4):
                for seed in range(4):
                    rr = np.random.default_rng(1000 * seed + hash(kind) % 977 + int(eps * 1e5))
                    E0, W0 = perturb(E, W, kind, eps, rr, p, n)
                    best, bE, bW = vqa._run_stages(E0, W0, ops, y, p, n, q, rhs, 0)
                    bt = vqa.brute_ratio(bE, bW, p, n, q, rhs)
                    best = max(best, bt)
                    cell_max = max(cell_max, best)
                    if best > 1.0 + 1e-12:
                        viol += 1
                        np.savez(OUT / f"ESCAPE_VIOLATION_p{p}_n{n}_q{q}_{kind}_{eps}_{seed}.npz",
                                 E=bE, W=bW, ratio=best)
                        print(f"  !!! ESCAPE ratio {best:.15f}  p={p} n={n} q={q} {kind} eps={eps} seed={seed}")
        print(f"p={p} n={n} q={q:g}: block ratio {r0:.15f};  best after "
              f"3 x 4 x 4 = 48 perturb-and-reascend runs: {cell_max:.15f}", flush=True)
        rows.append(dict(p=p, n=n, q=q, block_ratio=r0, best_after_escape=cell_max, S=S, b=b.tolist()))
        gmax = max(gmax, cell_max)
    print(f"\nMAX over all escape attempts: {gmax:.15f}   violations: {viol}")
    (OUT / "eq_escape.json").write_text(json.dumps(dict(rows=rows, max_ratio=gmax, violations=viol), indent=1))
    print("wrote", OUT / "eq_escape.json")
