"""Leak structure of single-character (semi-rigid) tests at (7,2), L = {1,2}.

Component j (feature j) at chi_{j,T}, delta-profile z(d).  Pairing coefficients:
   own   : S = sum_d z(d) c_j(d)                                  (field coefficient conj beta_{j,T})
   (l,+) : sum_d z(d) c_l(d)        with phi_l(chi),   l != j      (Fourier index m = -l)
   (l,-) : sum_d z(d) conj c_l(d)   with conj phi_l(-chi)          (Fourier index m = +l)
Mode of (l, chi'): input k = chi' - l*1.  Classes: free iff l in L and k in (+-L)^2 (not own/clean); else paid.
This module enumerates, for each T, the leak positions with their mode (l; class k1, class k2), DC count, sign of k,
and whether they are free.  It also builds the nonnegative neuron tensor of a leak vector and its block-injective norm.
"""
import itertools
import numpy as np

p = 7
L = (1, 2)
TS = {'1': (1, 0), '2': (0, 1), '12': (1, 1)}


def cls(k):
    k %= p
    return min(k, p - k)


def leak_positions(j, T):
    """list of dicts: key ('+',l) or ('-',l), chi', k (mod p), family (l, cls k1, cls k2), z (#DC), free flag"""
    t = TS[T]
    chi = np.array([(2 * j * t[0]) % p, (2 * j * t[1]) % p])
    out = []
    for sgn in ('+', '-'):
        for l in (1, 2, 3):
            if sgn == '+' and l == j:
                continue          # own
            c = chi if sgn == '+' else (-chi) % p
            k = (c - l) % p
            fam = (l, cls(k[0]), cls(k[1]))
            z = int(k[0] == 0) + int(k[1] == 0)
            free = (l in L) and all(cls(kk) in L for kk in k)
            out.append(dict(key=(sgn, l), chi=tuple(c), k=tuple(k), fam=fam, z=z, free=free))
    return out


def tensor(entries):
    """entries: list of (fam, coefficient>=0) -> nonneg tensor Tn[X-1, Y, Z] (X in 1..3, Y,Z in 0..3); coincident
    families add (same neuron modulus, worst-case phases)."""
    Tn = np.zeros((3, 4, 4))
    for (l, a, b), cval in entries:
        Tn[l - 1, a, b] += cval
    return Tn


def inj_norm_upper(Tn, ngrid=2000):
    """rigorous-ish upper bound on the injective norm of a small nonnegative 3-tensor (sup over unit X,Y,Z):
    max over X on a grid of the 2-norm of the contracted matrix, plus a Lipschitz correction.  X has <= 3 entries.
    (For a rigorous certificate the float spectral norms must be enclosed; done in the certificate script.)"""
    nzX = [i for i in range(Tn.shape[0]) if np.any(Tn[i] > 0)]
    if not nzX:
        return 0.0
    if len(nzX) == 1:
        return np.linalg.norm(Tn[nzX[0]], 2)
    if len(nzX) == 2:
        M1, M2 = Tn[nzX[0]], Tn[nzX[1]]
        ts = np.linspace(0, np.pi / 2, ngrid)
        vals = [np.linalg.norm(np.cos(t) * M1 + np.sin(t) * M2, 2) for t in ts]
        h = ts[1] - ts[0]
        lip = np.sqrt(np.linalg.norm(M1, 'fro') ** 2 + np.linalg.norm(M2, 'fro') ** 2)
        return max(vals) + lip * h / 2
    raise ValueError("3 X-slices not expected in a block")


def blocks(Tn, order=(0, 1, 2)):
    """split the paid tensor into blocks by the first factor (in 'order') that is non-L.
    non-L index sets: X: {3} (index 2 in Tn's X axis), Y,Z: {0,3}."""
    nonL = [{2}, {0, 3}, {0, 3}]
    Lset = [{0, 1}, {1, 2}, {1, 2}]
    out = []
    shape = Tn.shape
    for bi in range(3):
        B = np.zeros_like(Tn)
        for idx in itertools.product(*[range(n) for n in shape]):
            if Tn[idx] == 0:
                continue
            # first non-L factor in order
            first = None
            for f in order:
                if idx[f] in nonL[f]:
                    first = f; break
            assert first is not None, "free entry in paid tensor"
            if first == order[bi]:
                B[idx] = Tn[idx]
        out.append(B)
    return out


def K_block(Tn):
    """best block-injective neuron constant over the 6 block orders"""
    best = np.inf
    for order in itertools.permutations(range(3)):
        Bs = blocks(Tn, order)
        val = 0.0
        for B in Bs:
            # move the 'first' axis so that the X-contraction is over the smallest support
            dims = [len([i for i in range(B.shape[a]) if np.any(np.take(B, i, axis=a) > 0)]) for a in range(3)]
            a = int(np.argmin([d if d > 0 else 99 for d in dims]))
            Bt = np.moveaxis(B, a, 0)
            nz = [i for i in range(Bt.shape[0]) if np.any(Bt[i] > 0)]
            if len(nz) > 2:
                val = np.inf; break
            val += inj_norm_upper(Bt) ** 2
        best = min(best, np.sqrt(val))
    return best


if __name__ == '__main__':
    for j in (1, 2):
        for T in TS:
            print(f"feature {j}, T={T}:")
            for d in leak_positions(j, T):
                print(f"   {d['key']}: chi'={d['chi']} k={d['k']} family={d['fam']} DC={d['z']} {'FREE' if d['free'] else 'paid'}")
