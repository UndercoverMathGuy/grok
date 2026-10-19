"""Free (F2) coordinates of single-character tests, Lemma 3(c): for L = F minus {j0} (the dense-count case) and
for the programme supports, at small primes.  Free(j) = {-l : l in L-{j}, |2j-l| in L} u {l in L : |2j+l| in L}."""
import numpy as np
from chanlib import Chan, cls


def free_set(p, L, j):
    out = []
    for l in L:
        if l != j and cls(2 * j - l, p) in L:
            out.append(-l % p)
        if cls(2 * j + l, p) in L:
            out.append(l)
    return sorted(out)


for p in (5, 7, 11, 13, 17, 19, 23, 29, 31):
    F = list(range(1, (p - 1) // 2 + 1))
    L = [l for l in F if l != 1]
    nfree = [len(free_set(p, L, j)) for j in L]
    # cross-check with chanlib classification
    C = Chan(p, L)
    chk = [sum(1 for x in C.component(j, '1') if x[1] == 'free') for j in L]
    assert nfree == chk, (p, nfree, chk)
    single = [len(free_set(p, [j], j)) for j in F]
    print(f"p={p:2d} |F|={len(F):2d}  L=F-{{1}}: free coords per component (of p-2={p-2}) = {nfree};  single feature: {set(single)}")
