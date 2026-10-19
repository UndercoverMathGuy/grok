"""Channel-resolved isolation machinery at n = 2, general odd prime p and feature set L (subset of F).

Pair classes (l, chi), l in F, chi in Z_p^2 (field Phi_l, character chi; mode input k = chi - l*1):
  clean: chi = 0;  own: l in L, k in {+-l}^2, chi != 0;  free: l in L, k in (+-L)^2, not own/clean;  paid: else.
Single-character test component for feature j at chi_{j,T} = 2j 1_T, parametrised by its pairing vector
  w in C^{p-1} (index m in Z_p^*):  T_l(chi_{j,T}) = w_{-l},  T_l(-chi_{j,T}) = conj(w_l),
  delta-profile z(d) = -(1/p) sum_m w_m rho^{-m d},   own detection uses w_{-j}.
"""
import itertools
import numpy as np


def cls(k, p):
    k %= p
    return min(k, p - k)


class Chan:
    def __init__(self, p, L):
        self.p = p
        self.F = list(range(1, (p - 1) // 2 + 1))
        self.L = tuple(sorted(L))
        self.rho = np.exp(2j * np.pi / p)
        self.DEL = np.arange(1, p)
        self.Tsets = {'1': (1, 0), '2': (0, 1), '12': (1, 1)}
        # inverse-DFT matrix: z(d) = IDFT[d, m] w_m, m = 1..p-1
        self.IDFT = np.array([[-(1.0 / p) * self.rho ** (-m * d) for m in range(1, p)] for d in self.DEL])
        self.KM = np.array([[1 - np.cos(2 * np.pi * l * d / p) for d in self.DEL] for l in self.F])

    def pair_class(self, l, chi):
        p = self.p
        chi = (chi[0] % p, chi[1] % p)
        if chi == (0, 0):
            return 'clean'
        k = ((chi[0] - l) % p, (chi[1] - l) % p)
        if l in self.L and all(cls(kk, p) == l for kk in k):
            return 'own'
        if l in self.L and all(cls(kk, p) in self.L for kk in k):
            return 'free'
        return 'paid'

    def component(self, j, T):
        """for the single-character component of feature j at chi_{j,T}: list over m in Z_p^* (1..p-1) of
        (m, role, family, dc) with role in {'own','free','paid'}; family = (l, cls k1, cls k2)."""
        p = self.p
        t = self.Tsets[T]
        chi = ((2 * j * t[0]) % p, (2 * j * t[1]) % p)
        out = []
        for m in range(1, p):
            if (-m) % p in self.F:          # m = -l : pairs with phi_l(chi)
                l = (-m) % p; c = chi
            else:                            # m = l : pairs with conj phi_l(-chi)
                l = m; c = ((-chi[0]) % p, (-chi[1]) % p)
            role = self.pair_class(l, c)
            if role == 'own':
                assert l == j and c == chi
            k = ((c[0] - l) % p, (c[1] - l) % p)
            fam = (l, cls(k[0], p), cls(k[1], p))
            dc = int(k[0] == 0) + int(k[1] == 0)
            out.append((m, role, fam, dc))
        return out


def block_assign(fam, L):
    """first non-L factor in order (output, input1, input2): 0, 1, 2"""
    l, a, b = fam
    if l not in L:
        return 0
    if a not in L:
        return 1
    if b not in L:
        return 2
    raise ValueError("free family has no paid factor")


if __name__ == '__main__':
    C = Chan(7, (1, 2))
    for j in (1, 2):
        for T in ('1', '2', '12'):
            print(j, T, [(m, r, f, d) for (m, r, f, d) in C.component(j, T)])
    # free channels at general p for L = F minus {j0}
    for p in (11, 13):
        Fp = list(range(1, (p - 1) // 2 + 1))
        L = tuple(l for l in Fp if l != 1)
        C = Chan(p, L)
        nfree = {(j, T): sum(1 for x in C.component(j, T) if x[1] == 'free') for j in L for T in ('1', '12')}
        print(p, L, nfree)
