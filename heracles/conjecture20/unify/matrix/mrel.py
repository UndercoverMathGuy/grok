"""Core library for the MATRIX angle: chain-matrix relaxations of the margin problem.

Conventions (atom units, see REPORT.md section 1):
  * features j in S (subset of F={1..(p-1)/2}); field Phi_j : Z_p^n -> C;
  * g(x,delta) = sum_j Re(Phi_j(x) c_j(delta)),  c_j(delta) = 1 - rho^{-j delta};
  * own characters of feature j: chi_{j,T} = 2j 1_T  (T subset of [n]; T = {} is the clean character 0);
    the field coefficient there is conj(beta_{j,T}) (Lemma "supp" of the companion note).  Since every cost below is
    invariant under global conjugation we work with a_{j,T} := field coefficient directly.
  * chain matrix of feature j (positions 0..n):  H = a_0 I + L + U,  L[t,t'] = a_{(t',t]} (t>t'), U strictly upper FREE.
  * relaxed cost of feature j:  q_j = nu^{-r} ||H_j||_{S_r}^r  (>= atomic cost Q_j = sum_i |beta_ij|^r by Rotfel'd).
  * IF-normalised ratio:  h / ( nu^{1-s} Gamma (sum_j q_j)^s ),  Gamma = Gamma_s(p) (or kappa_min for one feature).
Margin LP: the own coefficients are fixed; every other Fourier coefficient of every field is either FREE (mode='free'),
zero (mode='C', class C), or free only on a prescribed set of characters (mode=dict j -> list of characters).
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix, vstack, hstack, identity


def subsets(n):
    return [T for k in range(n + 1) for T in itertools.combinations(range(n), k)]


def interval(tp, t):
    """slots (t', t] in 0-based slot indices: positions t' < t of the chain correspond to slots tp..t-1"""
    return tuple(range(tp, t))


class Model:
    def __init__(self, p, n, S, mode='free', s=1.5):
        self.p, self.n, self.S, self.mode = p, n, list(S), mode
        self.nu = n + 1
        self.s = s
        self.r = 1.0 / s
        self.X = np.array(list(itertools.product(range(p), repeat=n)))  # p^n x n
        self.N = len(self.X)
        self.rho = np.exp(2j * np.pi / p)
        self.Ts = subsets(n)
        self.dels = list(range(1, p))
        self.c = {j: {d: 1 - self.rho ** (-j * d) for d in self.dels} for j in self.S}
        # own characters
        self.own = {}
        for j in self.S:
            self.own[j] = {}
            for T in self.Ts:
                chi = np.zeros(n, dtype=int)
                chi[list(T)] = (2 * j) % p
                self.own[j][T] = tuple(chi)
        self.allchars = [tuple(c) for c in self.X]
        self._build()

    def _constrained_chars(self, j):
        """characters of Phi_j whose coefficient is fixed (own: given value; others: zero) in this mode"""
        own = set(self.own[j].values())
        if self.mode == 'free':
            return [c for c in self.allchars if c in own]
        if self.mode == 'C':
            return list(self.allchars)
        # dict: j -> set of FREE characters
        free = set(self.mode.get(j, []))
        return [c for c in self.allchars if (c in own) or (c not in free)]

    def _build(self):
        p, n, N = self.p, self.n, self.N
        K = len(self.S)
        # variables: for each j: Re Phi_j(x) (N), Im Phi_j(x) (N); then h.  total 2KN+1
        nv = 2 * K * N + 1
        # inequality rows: for each x, delta: -sum_j Re(Phi_j(x) c_j(d)) + h <= 0
        rows = []
        A = lil_matrix((N * len(self.dels), nv))
        r = 0
        for di, d in enumerate(self.dels):
            for xi in range(N):
                for k, j in enumerate(self.S):
                    cj = self.c[j][d]
                    A[r, 2 * k * N + xi] = -cj.real          # Re Phi * Re c
                    A[r, 2 * k * N + N + xi] = cj.imag       # - Im Phi * Im c  -> coefficient +Im c with minus sign
                A[r, nv - 1] = 1.0
                r += 1
        self.Aub = csr_matrix(A)
        self.bub = np.zeros(A.shape[0])
        # equality rows: for each j and each constrained character chi: (1/N) sum_x Phi_j(x) conj(rho^{chi.x}) = value
        eq_rows = []
        self.eq_index = []  # (j, chi, part) per row
        Aeq = []
        for k, j in enumerate(self.S):
            for chi in self._constrained_chars(j):
                ph = np.exp(-2j * np.pi * (self.X @ np.array(chi)) / p) / N  # conj character / N
                # value = sum_x (ReP + i ImP)(Re ph + i Im ph)
                re_row = np.zeros(nv); im_row = np.zeros(nv)
                re_row[2 * k * N:2 * k * N + N] = ph.real
                re_row[2 * k * N + N:2 * k * N + 2 * N] = -ph.imag
                im_row[2 * k * N:2 * k * N + N] = ph.imag
                im_row[2 * k * N + N:2 * k * N + 2 * N] = ph.real
                Aeq.append(re_row); self.eq_index.append((j, chi, 0))
                Aeq.append(im_row); self.eq_index.append((j, chi, 1))
        self.Aeq = csr_matrix(np.array(Aeq))
        self.nv = nv
        # map from (j,T) own to equality row indices
        self.own_rows = {}
        for idx, (j, chi, part) in enumerate(self.eq_index):
            for T, c in self.own[j].items():
                if c == chi:
                    self.own_rows[(j, T, part)] = idx

    def margin(self, a):
        """a: dict j -> dict T -> complex (own field coefficients).  Returns h, grad dict (j,T)->complex (dh/dRe + i dh/dIm)."""
        beq = np.zeros(self.Aeq.shape[0])
        for (j, T, part), idx in self.own_rows.items():
            v = a[j][T]
            beq[idx] = v.real if part == 0 else v.imag
        cobj = np.zeros(self.nv); cobj[-1] = -1.0
        res = linprog(cobj, A_ub=self.Aub, b_ub=self.bub, A_eq=self.Aeq, b_eq=beq,
                      bounds=[(None, None)] * self.nv, method='highs')
        if res.status != 0:
            return None, None
        h = res.x[-1]
        lam = res.eqlin.marginals  # d(objective)/d(beq); objective = -h
        grad = {}
        for (j, T, part), idx in self.own_rows.items():
            gval = -lam[idx]
            grad.setdefault((j, T), 0j)
            grad[(j, T)] += gval if part == 0 else 1j * gval
        return h, grad

    def field_from_atoms(self, j, beta, xi):
        """own coefficients of Phi_j generated by atoms beta_i (complex) with slot phases xi_i in T^n:
        Phi_j has coefficient conj(beta_{j,T}) at 2j1_T, beta_{j,T} = sum_i beta_i prod_{t in T} xi_it."""
        out = {}
        for T in self.Ts:
            bT = sum(b * np.prod([x[t] for t in T]) for b, x in zip(beta, xi))
            out[T] = np.conj(bT)
        return out


def chain_matrix(aj, n, U=None):
    """H = a_0 I + L + U with L[t,t'] = a_{(t',t]}; U strictly upper (array of complex in row-major order) or None."""
    nu = n + 1
    H = np.zeros((nu, nu), dtype=complex)
    for t in range(nu):
        H[t, t] = aj[()]
    for t in range(nu):
        for tp in range(t):
            H[t, tp] = aj[interval(tp, t)]
    if U is not None:
        k = 0
        for t in range(nu):
            for tp in range(t + 1, nu):
                H[t, tp] = U[k]; k += 1
    return H


def upper_index(nu):
    return [(t, tp) for t in range(nu) for tp in range(t + 1, nu)]


def schatten_r(M, r, eps=0.0):
    sv = np.linalg.svd(M, compute_uv=False)
    return np.sum(np.sqrt(sv ** 2 + eps ** 2) ** r) - (len(sv) * eps ** r if eps > 0 else 0.0)


def schatten_r_grad(M, r, eps=1e-9):
    """value and gradient G (complex) such that d value = Re tr(G^* dM)"""
    Uu, sv, Vh = np.linalg.svd(M)
    se = np.sqrt(sv ** 2 + eps ** 2)
    val = np.sum(se ** r)
    w = r * se ** (r - 2) * sv  # d(se^r)/d sv
    G = (Uu * w) @ Vh
    return val, G
