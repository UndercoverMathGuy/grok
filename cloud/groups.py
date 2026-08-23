"""S5 group machinery for the multi-task generality arm.

Elements: the 120 permutations of range(5), lexicographically ordered.
Composition convention: (a @ b)(x) = a(b(x)) — apply b, then a; the label
for input pair (a, b) is index(a @ b).

Irrep instrumentation is character-based (no explicit irrep matrices):
the isotypic projector for irrep r acting on functions f: G -> R is
    P_r[x, y] = (d_r / |G|) * chi_r(x @ y^{-1})
(all S5 characters are real). The seven P_r are orthogonal projectors
summing to identity — asserted numerically at import.
"""

import itertools

import numpy as np

PERMS = sorted(itertools.permutations(range(5)))
N = len(PERMS)                      # 120
IDX = {p: i for i, p in enumerate(PERMS)}


def _compose(a, b):                 # (a o b)(x) = a(b(x))
    return tuple(a[b[i]] for i in range(5))


MULT = np.zeros((N, N), dtype=np.int64)
INV = np.zeros(N, dtype=np.int64)
for i, a in enumerate(PERMS):
    for j, b in enumerate(PERMS):
        MULT[i, j] = IDX[_compose(a, b)]
for i, a in enumerate(PERMS):
    inv = tuple(sorted(range(5), key=lambda x: a[x]))
    INV[i] = IDX[inv]


def _cycle_type(p):
    seen, cyc = set(), []
    for s in range(5):
        if s in seen:
            continue
        n, x = 0, s
        while x not in seen:
            seen.add(x)
            x = p[x]
            n += 1
        cyc.append(n)
    return tuple(sorted(cyc, reverse=True))


# class order: e, (12), (12)(34), (123), (123)(45), (1234), (12345)
CLASS_TYPES = [(1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1), (3, 1, 1),
               (3, 2), (4, 1), (5,)]
CLASS_OF = np.array([CLASS_TYPES.index(_cycle_type(p)) for p in PERMS])

# character table of S5, rows = irreps, cols = classes above
IRREP_NAMES = ["trivial", "sign", "std", "std_sign", "5a", "5b", "6"]
IRREP_DIMS = np.array([1, 1, 4, 4, 5, 5, 6])
CHAR = np.array([
    [1,  1,  1,  1,  1,  1,  1],
    [1, -1,  1,  1, -1, -1,  1],
    [4,  2,  0,  1, -1,  0, -1],
    [4, -2,  0,  1,  1,  0, -1],
    [5,  1,  1, -1,  1, -1,  0],
    [5, -1,  1, -1, -1,  1,  0],
    [6,  0, -2,  0,  0,  0,  1],
], dtype=np.float64)

# isotypic projectors P_r[x, y] = d_r/|G| * chi_r(x o y^{-1})
_xyinv = MULT[np.arange(N)[:, None], INV[None, :]]        # (N, N) index of x o y^-1
PROJ = np.stack([(IRREP_DIMS[r] / N) * CHAR[r][CLASS_OF[_xyinv]]
                 for r in range(len(IRREP_NAMES))])        # (7, N, N)

_s = PROJ.sum(0)
assert np.allclose(_s, np.eye(N), atol=1e-10), "projectors must sum to I"
for _r in range(7):
    assert np.allclose(PROJ[_r] @ PROJ[_r], PROJ[_r], atol=1e-10), \
        f"P_{IRREP_NAMES[_r]} not idempotent"
    assert np.allclose(PROJ[_r] @ PROJ[(_r + 1) % 7], 0, atol=1e-10)


def irrep_energy(F):
    """F: (..., N) functions on G -> per-irrep energy (..., 7)."""
    F = np.asarray(F, dtype=np.float64)
    return np.stack([((F @ PROJ[r].T) ** 2).sum(-1) for r in range(7)], -1)
