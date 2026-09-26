"""Exact certificates plus an independent full-table check of the S5 witness.

No training or numerical optimization is used. The row check evaluates the
65-neuron weights, not the matrix-coefficient shortcut used in the proof.
Run from the repository: uv run --offline python heracles/astra/s5_counterexample_verify.py
"""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    permutations = list(itertools.permutations(range(5)))
    ps = np.asarray(permutations)
    lookup = {p: i for i, p in enumerate(permutations)}
    signs = np.array([
        (-1) ** sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5))
        for p in permutations
    ])
    v = np.array([-10, -4, 1, 4, 9], dtype=np.int64)
    assert v.sum() == 0 and v @ v == 214
    raw_gaps = 214 - (v[ps] * v).sum(axis=1)
    even_wrong = (signs == 1) & (np.arange(120) != 0)
    assert raw_gaps[even_wrong].min() == 45
    assert raw_gaps[signs == -1].min() == 9
    patched_gap_numerators = raw_gaps + 18 * (1 - signs)
    assert patched_gap_numerators[1:].min() == 45
    exact_margin = Fraction(45, 214)
    rational_ratio_lower_bound = Fraction(6_144_000, 5_685_766)
    assert rational_ratio_lower_bound > Fraction(27, 25)
    assert 25 * 6_144_000 - 27 * 5_685_766 == 84_318

    # Helmert basis of the sum-zero subspace; no gt_core representation is used.
    basis = np.zeros((5, 4))
    for k in range(1, 5):
        basis[:k, k - 1] = 1 / math.sqrt(k * (k + 1))
        basis[k, k - 1] = -k / math.sqrt(k * (k + 1))
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-14)
    assert np.allclose(basis.sum(axis=0), 0, atol=1e-14)
    # P_g e_j=e_{g(j)}, R_g=B^T P_g B, endpoint_i(g)=u^T P_g B_i.
    rho = np.einsum("gji,jk->gik", basis[ps], basis)
    endpoints = (v[ps] @ basis) / math.sqrt(214)
    operand_weights, readouts = [], []
    for i, j, k in itertools.product(range(4), repeat=3):
        operand_weights.append([endpoints[:, i], rho[:, i, j], rho[:, j, k]])
        readouts.append(endpoints[:, k])
    beta = Fraction(9, 107)
    sign_factor = float(beta) ** 0.25 * signs
    operand_weights.append([sign_factor.copy() for _ in range(3)])
    readouts.append(sign_factor.copy())
    E = np.array(operand_weights).transpose(1, 0, 2)
    W = np.array(readouts)
    assert E.shape == (3, 65, 120) and W.shape == (65, 120)
    factor_squared_norms = np.concatenate([(E**2).sum(axis=2), (W**2).sum(axis=1)[None]])
    assert np.allclose(factor_squared_norms[:, :64], 30, atol=1e-12)
    actual_penalty = float((E**2).sum() + (W**2).sum())
    predicted_penalty = 7680 + 1440 / math.sqrt(107)
    assert abs(actual_penalty / predicted_penalty - 1) < 1e-14

    # Independently build the multiplication labels from permutation composition.
    product = np.empty((120, 120), dtype=int)
    for a, pa in enumerate(permutations):
        for b, pb in enumerate(permutations):
            product[a, b] = lookup[tuple(pa[pb[t]] for t in range(5))]
    inverses = np.array([lookup[tuple(np.argsort(p))] for p in permutations])
    # Check representation convention on EVERY pair, not a sample.
    homomorphism_error = 0.0
    for a in range(120):
        homomorphism_error = max(homomorphism_error, float(np.abs(rho[a] @ rho - rho[product[a]]).max()))
    assert homomorphism_error < 1e-13

    # Expand actual neuron outputs for ALL 120^3 rows in 120 bounded batches.
    rest = (E[1, :, :, None] * E[2, :, None, :]).reshape(65, 120**2)
    minimum_gap = float("inf")
    formula_error = 0.0
    witness = None
    row_count = 0
    for a in range(120):
        hidden = E[0, :, a, None] * rest
        logits = hidden.T @ W
        correct = product[product[a, :, None], np.arange(120)[None, :]].reshape(-1)
        # Shortcut is used only for comparison AFTER the expanded logits exist.
        offsets = product[correct[:, None], inverses[None, :]]
        expected = 1 - raw_gaps[offsets] / 214 + float(beta) * signs[offsets]
        formula_error = max(formula_error, float(np.abs(logits - expected).max()))
        gaps = logits[np.arange(120**2), correct, None] - logits
        gaps[np.arange(120**2), correct] = np.inf
        flat = int(np.argmin(gaps))
        row, wrong = np.unravel_index(flat, gaps.shape)
        if float(gaps[row, wrong]) < minimum_gap:
            minimum_gap = float(gaps[row, wrong])
            witness = dict(a=a, b=int(row // 120), d=int(row % 120), correct=int(correct[row]), wrong=int(wrong))
        row_count += len(correct)
        if (a + 1) % 20 == 0:
            print(f"Verified {row_count:,} rows; minimum gap {minimum_gap:.15g}", flush=True)
    assert row_count == 120**3
    assert abs(minimum_gap - float(exact_margin)) < 1e-12
    assert formula_error < 1e-12
    ceiling = 3 / 30720**2
    gamma = minimum_gap / actual_penalty**2
    assert gamma / ceiling > 1.08
    # The same coefficient is not a win at degree three.
    nu3_penalty = 1440 + 360 * float(beta) ** (2 / 3)
    nu3_reference_ratio = (float(exact_margin) / nu3_penalty**1.5) / (3 / 5760**1.5)
    result = dict(
        exact=dict(vector=v.tolist(), squared_norm=214, even_gap_numerator=45,
                   odd_gap_numerator=9, beta=str(beta), margin=str(exact_margin),
                   rational_ratio_lower_bound=str(rational_ratio_lower_bound),
                   proven_ratio_greater_than="27/25"),
        full_table=dict(neurons=65, rows=row_count, wrong_gaps=row_count * 119,
                        minimum_gap=minimum_gap, penalty=actual_penalty, gamma=gamma,
                        symmetric_ceiling=ceiling, ratio=gamma / ceiling,
                        max_logit_formula_error=formula_error,
                        max_homomorphism_error=homomorphism_error, minimum_witness=witness),
        same_construction_nu3_ratio_to_schoolbook=nu3_reference_ratio,
    )
    (HERE / "s5_counterexample_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(HERE / "s5_counterexample_nu4.npz", E=E, W=W, permutations=ps)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
