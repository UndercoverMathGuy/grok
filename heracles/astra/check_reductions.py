"""Numerical stress checks, not proof certificates, for sections 4 and 5."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from verify import ansatz, vertex_amplitudes


def compress(seed):
    rng = np.random.default_rng(seed)
    p, nu, r = 3, 3, 2/3
    d = p**(nu-1)*(p-1)
    count = d+5
    slots = rng.normal(size=(nu, count, p))
    slots[0] -= slots[0].mean(axis=1, keepdims=True)
    slots /= np.linalg.norm(slots, axis=2, keepdims=True)
    columns = np.einsum('ic,ia,ib->cabi', *slots).reshape(p**nu, count)
    coeff = np.exp(rng.normal(size=count))
    target = columns @ coeff
    before = np.sum(coeff**r)
    while len(coeff) > d:
        _, _, vh = np.linalg.svd(columns, full_matrices=True)
        v = vh[-1]
        assert np.linalg.norm(columns @ v) < 1e-12
        positive, negative = v > 1e-12, v < -1e-12
        candidates = []
        if positive.any():
            candidates.append(coeff - np.min(coeff[positive]/v[positive]) * v)
        if negative.any():
            candidates.append(coeff - np.max(coeff[negative]/v[negative]) * v)
        candidates = [np.maximum(x, 0) for x in candidates]
        coeff = min(candidates, key=lambda x: np.sum(x**r))
        keep = coeff > 1e-10
        coeff, columns = coeff[keep], columns[:, keep]
    error = np.max(np.abs(columns @ coeff - target))
    after = np.sum(coeff**r)
    assert error < 1e-10 and after < before
    return {'seed': seed, 'before_width': count, 'after_width': len(coeff),
            'tensor_max_error': float(error), 'cost_ratio': float(after/before)}


def radial_check(p=11):
    w, gamma = ansatz(p, vertex_amplitudes(p))
    logits = np.einsum('ia,ib,ic->abc', *w)
    y = (np.arange(p)[:, None] + np.arange(p)[None, :]) % p
    rows = logits.reshape(-1, p)
    labels = y.ravel()
    gaps = rows[np.arange(p*p), labels, None] - rows
    gaps = gaps[np.arange(p)[None, :] != labels[:, None]].reshape(p*p, p-1)
    norm = np.sum(w*w)
    gaps /= norm**1.5
    g = gaps.min()
    assert abs(g/(3*gamma/4/norm**1.5)-1) < 1e-11
    r = 2/3
    out = []
    for L in (10, 20, 50, 100, 200):
        lam = np.exp(-L)
        def derivative(t):
            exp = np.exp(-t*gaps)
            ce_grad = -np.mean(np.sum(gaps*exp, axis=1)/(1+exp.sum(axis=1)))
            return ce_grad + lam*r*t**(r-1)
        t = brentq(derivative, L/(2*g), 2*L/g, xtol=1e-9)
        ce = np.mean(np.log1p(np.exp(-t*gaps).sum(axis=1)))
        objective = ce + lam*t**r
        out.append({'L': L, 'radial_scale_over_prediction': float(t*g/L),
                    'objective_over_prediction': float(objective/(lam*L**r*g**(-r)))})
    assert abs(out[-1]['objective_over_prediction']-1) < .03
    return out


if __name__ == '__main__':
    result = {'compression': [compress(seed) for seed in range(10)],
              'radial_asymptotic': radial_check()}
    Path(__file__).with_name('reduction_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
