# For every frequency set T at prime p (up to dilation and k -> -k), certify one- and two-sided depths, then the exact
# flip-enhanced transfer constant c; T passes if c > 0.272 (so eta-form 3b applies with strict slack).
import sys, json, itertools
from depth_cert import certify as dcert
from transfer_cert import certify as tcert

def classes(p):
    h = (p - 1) // 2
    fold = lambda k: min(k % p, (-k) % p)
    seen, reps = set(), []
    for K in range(1, h + 1):
        for T in itertools.combinations(range(1, h + 1), K):
            key = tuple(sorted(T))
            if key in seen: continue
            orbit = set()
            for lam in range(1, p):
                orbit.add(tuple(sorted(fold(lam * k) for k in T)))
            seen |= orbit
            reps.append(min(orbit))
    return reps

def grid_L(p, K):
    return {1: 24, 2: 16, 3: 12, 4: 8, 5: 6, 6: 6}.get(K, 5)

if __name__ == '__main__':
    p = int(sys.argv[1]); out = []
    for T in classes(p):
        K = len(T); L = grid_L(p, K)
        A1, _ = dcert(p, T, L, False)
        A2, _ = dcert(p, T, L, True)
        c, ok, lam, nv = tcert(A2, A1, 0.272)
        row = dict(p=p, T=list(T), L=L, one_sided=round(A1, 6), two_sided=round(A2, 6), transfer=round(c, 6), passes=ok)
        out.append(row); print(json.dumps(row), flush=True)
    json.dump(out, open(f'enum_p{p}.json', 'w'), indent=1)
