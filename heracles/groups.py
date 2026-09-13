"""Finite groups for the heracles group tasks: S5, S4, A5, D_113. Label of input pair (a, b) = index(a @ b).

Conventions copied from cloud/groups.py (S5): permutations of range(n) in lexicographic order, composition
(a @ b)(x) = a(b(x)). Irrep instrumentation is character-based: the isotypic projector for irrep r on functions
f: G -> R is P_r[x, y] = (d_r / |G|) chi_r(x @ y^-1) (all characters here are real; P_r is symmetric).
Conjugacy classes are computed numerically from the multiplication table; character tables are hard-coded
(S4, S5, A5) or closed form (D_n) and checked at build time: projectors sum to I, are idempotent and orthogonal,
sum d_r^2 = |G|.

Group object fields: name, N, MULT (N, N), INV (N,), e (identity index), class_of (N,), class_names,
irrep_names, irrep_dims, CHAR (R, C), PROJ (R, N, N), faithful (bool per irrep), sign (index of the sign irrep
or None), trivial (index 0).
"""
import itertools
from dataclasses import dataclass

import numpy as np

_CACHE = {}


@dataclass
class Group:
    name: str
    N: int
    MULT: np.ndarray
    INV: np.ndarray
    e: int
    class_of: np.ndarray
    class_names: list
    irrep_names: list
    irrep_dims: np.ndarray
    CHAR: np.ndarray
    PROJ: np.ndarray
    faithful: np.ndarray
    sign: object
    schur: np.ndarray = None      # <chi_r, chi_r> per irrep: 1 (absolutely irreducible) or 2 (complex-type real irrep, e.g. 2-dim irreps of abelian groups)

    def __post_init__(self):
        if self.schur is None:
            self.schur = np.ones(len(self.irrep_names))

    @property
    def R(self):
        return len(self.irrep_names)

    def irrep_energy(self, F):
        """F: (..., N) -> per-irrep energy (..., R) = ||P_r F||^2."""
        F = np.asarray(F, dtype=np.float64)
        return np.stack([((F @ self.PROJ[r].T) ** 2).sum(-1) for r in range(self.R)], -1)

    def amplitudes(self, f):
        """f: (N,) folded clean logit function f(x) = A-weighted sum chi_r(x)/d_r -> A_r = d_r sqrt(E_r / (N s_r))."""
        return self.irrep_dims * np.sqrt(self.irrep_energy(f) / (self.N * self.schur))

    def coverage(self):
        """(R, N) coverage of offset x by irrep r: 1 - chi_r(x) / d_r (0 at the identity; the mod-add 1 - cos)."""
        return 1 - self.CHAR[:, self.class_of] / self.irrep_dims[:, None]


def _perm_group(perms, n):
    perms = sorted(perms); N = len(perms); idx = {p: i for i, p in enumerate(perms)}
    MULT = np.zeros((N, N), dtype=np.int64); INV = np.zeros(N, dtype=np.int64)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            MULT[i, j] = idx[tuple(a[b[k]] for k in range(n))]
        INV[i] = idx[tuple(sorted(range(n), key=lambda x: a[x]))]
    return perms, MULT, INV, idx[tuple(range(n))]


def _cycle_type(p):
    seen, cyc = set(), []
    for s in range(len(p)):
        if s in seen: continue
        k, x = 0, s
        while x not in seen:
            seen.add(x); x = p[x]; k += 1
        cyc.append(k)
    return tuple(sorted(cyc, reverse=True))


def _classes(MULT, INV):
    """conjugacy classes by orbit; returns class_of (N,) with classes numbered by their smallest element."""
    N = MULT.shape[0]; class_of = -np.ones(N, dtype=np.int64); c = 0
    for x in range(N):
        if class_of[x] >= 0: continue
        orb = MULT[MULT[np.arange(N), x], INV]      # g x g^-1 for all g
        class_of[np.unique(orb)] = c; c += 1
    return class_of


def _finish(name, N, MULT, INV, e, class_of, class_names, irrep_names, irrep_dims, CHAR, faithful, sign, schur=None):
    irrep_dims = np.asarray(irrep_dims); CHAR = np.asarray(CHAR, dtype=np.float64)
    R = len(irrep_names); assert CHAR.shape == (R, len(class_names))
    schur = np.ones(R) if schur is None else np.asarray(schur, dtype=np.float64)
    assert abs((irrep_dims ** 2 / schur).sum() - N) < 1e-9, "sum d^2 / s must equal |G|"
    sizes = np.bincount(class_of, minlength=len(class_names)).astype(np.float64)
    gram = (CHAR * sizes) @ CHAR.T / N
    assert np.allclose(gram, np.diag(schur), atol=1e-9), f"{name}: character rows not orthogonal with the stated Schur multiplicities\n{gram}"
    xyinv = MULT[np.arange(N)[:, None], INV[None, :]]
    PROJ = np.stack([(irrep_dims[r] / (N * schur[r])) * CHAR[r][class_of[xyinv]] for r in range(R)])
    assert np.allclose(PROJ.sum(0), np.eye(N), atol=1e-9), f"{name}: projectors must sum to I"
    for r in range(R):
        assert np.allclose(PROJ[r] @ PROJ[r], PROJ[r], atol=1e-9), f"{name}: P_{irrep_names[r]} not idempotent"
        assert np.allclose(PROJ[r], PROJ[r].T, atol=1e-12)
    return Group(name, N, MULT, INV, e, class_of, class_names, list(irrep_names), irrep_dims, CHAR, PROJ,
                 np.asarray(faithful, dtype=bool), sign, schur)


def _by_type(perms, class_of, types):
    """map class ids to positions in `types` (cycle types) — asserts one class per type."""
    order = {}
    for i, p in enumerate(perms):
        t = _cycle_type(p); k = types.index(t)
        assert order.get(class_of[i], k) == k; order[class_of[i]] = k
    assert len(order) == len(types)
    perm = np.array([order[c] for c in range(len(types))])
    return perm


def S5():
    perms, MULT, INV, e = _perm_group(itertools.permutations(range(5)), 5)
    class_of = _classes(MULT, INV)
    types = [(1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1), (3, 1, 1), (3, 2), (4, 1), (5,)]
    class_of = _by_type(perms, class_of, types)[class_of]
    names = ["trivial", "sign", "std", "std_sign", "5a", "5b", "6"]; dims = [1, 1, 4, 4, 5, 5, 6]
    CHAR = [[1, 1, 1, 1, 1, 1, 1], [1, -1, 1, 1, -1, -1, 1], [4, 2, 0, 1, -1, 0, -1], [4, -2, 0, 1, 1, 0, -1],
            [5, 1, 1, -1, 1, -1, 0], [5, -1, 1, -1, -1, 1, 0], [6, 0, -2, 0, 0, 0, 1]]
    return _finish("s5", 120, MULT, INV, e, class_of, ["e", "(12)", "(12)(34)", "(123)", "(123)(45)", "(1234)", "(12345)"],
                   names, dims, CHAR, [0, 0, 1, 1, 1, 1, 1], 1)


def S4():
    perms, MULT, INV, e = _perm_group(itertools.permutations(range(4)), 4)
    class_of = _classes(MULT, INV)
    types = [(1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)]
    class_of = _by_type(perms, class_of, types)[class_of]
    names = ["trivial", "sign", "2", "std", "std_sign"]; dims = [1, 1, 2, 3, 3]
    CHAR = [[1, 1, 1, 1, 1], [1, -1, 1, 1, -1], [2, 0, 2, -1, 0], [3, 1, -1, 0, -1], [3, -1, -1, 0, 1]]
    return _finish("s4", 24, MULT, INV, e, class_of, ["e", "(12)", "(12)(34)", "(123)", "(1234)"],
                   names, dims, CHAR, [0, 0, 0, 1, 1], 1)


def A5():
    even = [p for p in itertools.permutations(range(5)) if sum(len(c) - 1 for c in _cycles(p)) % 2 == 0]
    perms, MULT, INV, e = _perm_group(even, 5)
    class_of = _classes(MULT, INV)
    # classes: e, (12)(34), (123), 5-cycles A (class of the lex-first 5-cycle), 5-cycles B
    cls = {}
    for i, p in enumerate(perms):
        t = _cycle_type(p)
        if t == (5,):
            key = ("5", class_of[i])
        else:
            key = (t,)
        cls.setdefault(key, class_of[i])
    five = sorted(c for k, c in cls.items() if k[0] == "5")
    assert len(five) == 2 and len(cls) == 5
    order = {cls[((1, 1, 1, 1, 1),)]: 0, cls[((2, 2, 1),)]: 1, cls[((3, 1, 1),)]: 2, five[0]: 3, five[1]: 4}
    class_of = np.array([order[c] for c in class_of])
    phi = (1 + 5 ** 0.5) / 2
    names = ["trivial", "3a", "3b", "4", "5"]; dims = [1, 3, 3, 4, 5]
    CHAR = [[1, 1, 1, 1, 1], [3, -1, 0, phi, 1 - phi], [3, -1, 0, 1 - phi, phi], [4, 0, 1, -1, -1], [5, 1, -1, 0, 0]]
    return _finish("a5", 60, MULT, INV, e, class_of, ["e", "(12)(34)", "(123)", "5A", "5B"],
                   names, dims, CHAR, [0, 1, 1, 1, 1], None)


def _cycles(p):
    seen, out = set(), []
    for s in range(len(p)):
        if s in seen: continue
        c, x = [], s
        while x not in seen:
            seen.add(x); c.append(x); x = p[x]
        out.append(c)
    return out


def D(n):
    """dihedral group of order 2n, n odd. element (k, e) = r^k s^e at index e*n + k; (k1,e1)(k2,e2) = (k1 + (-1)^e1 k2, e1+e2)."""
    assert n % 2 == 1
    N = 2 * n; MULT = np.zeros((N, N), dtype=np.int64); INV = np.zeros(N, dtype=np.int64)
    for i in range(N):
        k1, e1 = i % n, i // n
        for j in range(N):
            k2, e2 = j % n, j // n
            MULT[i, j] = ((e1 + e2) % 2) * n + (k1 + (-1) ** e1 * k2) % n
        INV[i] = k1 if e1 else (-k1) % n
        INV[i] += e1 * n
    h = (n - 1) // 2
    class_of = np.array([(min(i, n - i) if i < n else h + 1) for i in range(N)])   # 0: e; 1..h: {r^k, r^-k}; h+1: reflections
    class_names = ["e"] + [f"r^{k}" for k in range(1, h + 1)] + ["s"]
    names = ["trivial", "sign"] + [f"rho{j}" for j in range(1, h + 1)]; dims = [1, 1] + [2] * h
    CHAR = np.zeros((h + 2, h + 2)); CHAR[0] = 1; CHAR[1, :h + 1] = 1; CHAR[1, h + 1] = -1
    for j in range(1, h + 1):
        CHAR[1 + j, 0] = 2
        for k in range(1, h + 1):
            CHAR[1 + j, k] = 2 * np.cos(2 * np.pi * j * k / n)
    return _finish(f"d{n}", N, MULT, INV, 0, class_of, class_names, names, dims, CHAR, [0, 0] + [1] * h, 1)


def abelian(orders, name):
    """Z_{n1} x ... x Z_{nk}. Elements = tuples in lexicographic order; every class is a singleton. Real irreps: the
    trivial character, every real character of order 2 (1-dim), and one 2-dim real irrep chi + conj(chi) per pair of
    conjugate complex characters (Schur multiplicity 2). Coverage of an offset x by the 2-dim irrep (j): 1 - cos(2 pi j.x / n).
    'faithful' marks every nontrivial irrep (the leader readout needs candidates; for abelian groups it means 'nontrivial')."""
    import itertools as it
    els = list(it.product(*[range(n) for n in orders])); N = len(els); idx = {x: i for i, x in enumerate(els)}
    MULT = np.zeros((N, N), dtype=np.int64); INV = np.zeros(N, dtype=np.int64)
    for i, a in enumerate(els):
        INV[i] = idx[tuple((-x) % n for x, n in zip(a, orders))]
        for j, b in enumerate(els):
            MULT[i, j] = idx[tuple((x + y) % n for x, y, n in zip(a, b, orders))]
    class_of = np.arange(N); class_names = [str(x) for x in els]
    chars = list(it.product(*[range(n) for n in orders]))            # character labels j, chi_j(x) = exp(2 pi i j.x/n)
    seen = set(); names, dims, rows, schur, faithful = [], [], [], [], []
    for j in chars:
        if j in seen: continue
        jb = tuple((-y) % n for y, n in zip(j, orders)); seen.add(j); seen.add(jb)
        ph = np.array([sum(y * x / n for x, y, n in zip(a, j, orders)) for a in els]) * 2 * np.pi
        lab = "+".join(str(y) for y in j) if len(orders) > 1 else str(j[0])
        if j == jb:                                                   # real character (trivial or order 2)
            names.append("trivial" if all(y == 0 for y in j) else f"sgn{lab}"); dims.append(1); rows.append(np.cos(ph)); schur.append(1)
        else:
            names.append(f"f{lab}"); dims.append(2); rows.append(2 * np.cos(ph)); schur.append(2)
        faithful.append(not all(y == 0 for y in j))
    order = np.argsort([0 if n == "trivial" else 1 for n in names], kind="stable")
    names = [names[i] for i in order]; dims = [dims[i] for i in order]; rows = [rows[i] for i in order]
    schur = [schur[i] for i in order]; faithful = [faithful[i] for i in order]
    sign = next((i for i, n in enumerate(names) if n.startswith("sgn")), None)
    return _finish(name, N, MULT, INV, idx[tuple(0 for _ in orders)], class_of, class_names, names, dims, np.stack(rows), faithful, sign, schur)


BUILDERS = {"s5": S5, "s4": S4, "a5": A5, "d113": lambda: D(113), "z120": lambda: abelian((120,), "z120"),
            "z11x11": lambda: abelian((11, 11), "z11x11"), "z113": lambda: abelian((113,), "z113")}


def get_group(name):
    if name not in _CACHE:
        _CACHE[name] = BUILDERS[name]()
    return _CACHE[name]


if __name__ == "__main__":
    import time
    for nm in BUILDERS:
        t = time.time(); G = get_group(nm)
        print(f"{nm}: |G| {G.N}, {G.R} irreps {G.irrep_names[:8]}{'...' if G.R > 8 else ''}, e {G.e}, "
              f"faithful {[G.irrep_names[i] for i in np.flatnonzero(G.faithful)][:6]}, sign {G.sign}, {time.time()-t:.1f}s")
