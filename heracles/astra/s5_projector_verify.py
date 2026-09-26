"""Exact projector certificate and independent full-table network verification.

No optimizer and no inherited group/construction helpers are used.
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
    ps = np.array(permutations)
    lookup = {p: i for i, p in enumerate(permutations)}
    signs = np.array([(-1) ** sum(p[i] > p[j] for i in range(5) for j in range(i+1, 5)) for p in permutations])
    fixed = (ps == np.arange(5)).sum(axis=1)
    crossings = (ps[:, :2] >= 2).sum(axis=1)
    v = np.array([-3, -3, 2, 2, 2])
    assert v.sum() == 0 and v @ v == 30
    # Independent evaluation of the projector score via its rank-one complement.
    inner_numerators = (v[ps] * v).sum(axis=1)
    assert np.array_equal(30 - inner_numerators, 25 * crossings)
    score_times_30 = 30 * (fixed - 1) - inner_numerators + 15 * signs
    gap_times_30 = score_times_30[0] - score_times_30
    assert gap_times_30[1:].min() == 65
    assert np.array_equal(gap_times_30, 30 * (5 - fixed) - 25 * crossings + 15 * (1-signs))
    assert Fraction(13, 18) * Fraction(256, 195)**2 > Fraction(31, 25)
    assert Fraction(169, 324) * Fraction(24, 19)**3 > 1

    B = np.zeros((5, 4))
    for k in range(1, 5):
        B[:k, k-1] = 1 / math.sqrt(k*(k+1))
        B[k, k-1] = -k / math.sqrt(k*(k+1))
    F = np.array([[1, -1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 1, 1, -2]], dtype=float).T
    F /= np.sqrt(np.sum(F**2, axis=0))
    assert np.allclose(F.T @ F, np.eye(3))
    assert np.allclose(F.T @ v, 0)
    rho = np.einsum("gji,jk->gik", B[ps], B)
    endpoints = np.einsum("gja,jk->agk", F[ps], B)
    product = np.empty((120, 120), dtype=int)
    for a, pa in enumerate(permutations):
        for b, pb in enumerate(permutations):
            product[a, b] = lookup[tuple(pa[pb[t]] for t in range(5))]
    inverse = np.array([lookup[tuple(np.argsort(p))] for p in permutations])

    results = {}
    for nu in (3, 4):
        n = nu - 1
        Es, Ws = [], []
        for alpha in range(3):
            for indices in itertools.product(range(4), repeat=n):
                factors = [endpoints[alpha, :, indices[0]]]
                factors.extend(rho[:, indices[j-1], indices[j]] for j in range(1, n))
                Es.append(factors)
                Ws.append(endpoints[alpha, :, indices[-1]])
        sign_factor = 2**(-1/nu) * signs
        Es.append([sign_factor.copy() for _ in range(n)])
        Ws.append(sign_factor.copy())
        E = np.array(Es).transpose(1, 0, 2)
        W = np.array(Ws)
        width = 3*4**n + 1
        assert E.shape == (n, width, 120)
        assert np.allclose((E[:, :-1]**2).sum(axis=2), 30, atol=1e-12)
        assert np.allclose((W[:-1]**2).sum(axis=1), 30, atol=1e-12)
        penalty = float((E**2).sum() + (W**2).sum())
        predicted = 90*nu*4**n + 120*nu*2**(-2/nu)
        assert abs(penalty / predicted - 1) < 1e-14
        rest = E[1] if nu == 3 else (E[1, :, :, None] * E[2, :, None, :]).reshape(width, 120**2)
        min_gap = float("inf")
        max_error = 0.0
        row_count = 0
        for a in range(120):
            hidden = E[0, :, a, None] * rest
            logits = hidden.T @ W
            correct = product[a] if nu == 3 else product[product[a, :, None], np.arange(120)[None, :]].reshape(-1)
            expected = score_times_30[product[correct[:, None], inverse[None, :]]] / 30
            max_error = max(max_error, float(np.abs(logits - expected).max()))
            gaps = logits[np.arange(len(correct)), correct, None] - logits
            gaps[np.arange(len(correct)), correct] = np.inf
            min_gap = min(min_gap, float(gaps.min()))
            row_count += len(correct)
            if (a + 1) % 30 == 0:
                print(f"nu={nu}: verified {row_count:,} rows; minimum gap {min_gap:.15g}", flush=True)
        assert row_count == 120**n
        assert abs(min_gap - 13/6) < 1e-12
        assert max_error < 1e-12
        reference = 3 / (30*nu*4**nu)**(nu/2)
        gamma = min_gap / penalty**(nu/2)
        assert gamma / reference > (1.24 if nu == 4 else 1.02)
        results[f"nu{nu}"] = dict(
            neurons=width, input_rows=row_count, wrong_gaps=row_count*119,
            exact_margin="13/6", expanded_margin=min_gap,
            penalty=penalty, gamma=gamma, schoolbook_reference=reference,
            ratio_to_schoolbook=gamma/reference,
            reference_is_proven_full_symmetric_ceiling=(nu == 4),
            max_expanded_logit_error=max_error,
        )
        np.savez_compressed(HERE / f"s5_projector_nu{nu}.npz", E=E, W=W, permutations=ps)
    (HERE / "s5_projector_verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
