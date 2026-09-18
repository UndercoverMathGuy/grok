"""audit_common.py -- shared exact helpers for the hostile proof audit (PROOF_AUDIT.md). Local CPU, float64.
Family: z(c; a_1..a_n) = sum_i w_i(c) * act(e_i1(a_1), ..., e_in(a_n)),
  act = 'pow'  : (sum_j e_ij(a_j))^n      (THEORY_nsum sec 3)
  act = 'prod' : prod_j e_ij(a_j)         (THEORY_nsum sec 6, gated chain)
E: tensor (n, m, p), W: tensor (m, p)."""
import itertools, math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def rows(p, n):
    ops = torch.tensor(list(itertools.product(range(p), repeat=n)))
    return ops, ops.sum(1) % p


def logits(E, W, ops, act):
    n = E.shape[0]
    cols = [E[j][:, ops[:, j]] for j in range(n)]          # (m, N)
    if act == "prod":
        h = cols[0]
        for j in range(1, n):
            h = h * cols[j]
    else:
        x = sum(cols)
        h = x ** n
    return (W.T @ h).T                                      # (N, p)


def margin(z, y):
    zy = z.gather(1, y[:, None]).squeeze(1)
    zo = z.clone()
    zo[torch.arange(len(y)), y] = -float("inf")
    return (zy - zo.max(1).values)


def clean(z, y, p):
    D = torch.arange(p)
    idx = (y[:, None] - D[None, :]) % p
    return z.gather(1, idx).mean(0)                         # g(D)


def clean_table(g, y, p):
    c = torch.arange(p)
    return g[(y[:, None] - c[None, :]) % p]


def Gmat(p):
    k = np.arange(1, (p - 1) // 2 + 1); d = np.arange(1, p)
    return 1 - np.cos(2 * np.pi * np.outer(k, d) / p)       # (nf, p-1)


def cn(n, act):
    return (math.factorial(n) if act == "pow" else 1.0) * (n + 1) ** (-(n + 1) / 2)


def Pq(E, W, q):
    n2 = E.square().sum((0, 2)) + W.square().sum(1)
    return n2.pow(q / 2).sum()


def gamma(E, W, ops, y, act, q):
    nu = E.shape[0] + 1
    return margin(logits(E, W, ops, act), y).min() / Pq(E, W, q) ** (nu / q)
