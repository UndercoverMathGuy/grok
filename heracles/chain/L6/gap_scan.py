# Gap scan for L6 in the share-floor accounting (VERIFY.md "L5 fork results"):
# for each candidate heavy junky set T: structural parameters, one-/two-sided K-mode depths (polar LP search = UPPER
# estimates of the true depths; two-sided depth A_s is also an upper bound on the flip-enlarged l2-polyball depth),
# the flip-enhanced transfer lower bound c_lo, and the share-floored programme ratio at the discount available at depth a
# (lone-atom curve delta(a) = max(1, sqrt3 (1 - sqrt3 a))) for a = A_s (optimistic) and a = c_lo (pessimistic).
# ratio > 1 at a = A_s  => this accounting fails on T even with the best single-offset depth ("danger").
# ratio <= 1 at a = c_lo => T is handled (modulo the numerics of delta_crit).
# usage: gap_scan.py p   (writes gap_p{p}.jsonl)
import sys, json, time, numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../L5')
from tools import *
from floor_test import value as floored_ratio

p = int(sys.argv[1]); rng = np.random.default_rng(p)
h = (p - 1) // 2
Gam, Sstar = gamma_star(p, rng)
delta = lambda a: max(1.0, np.sqrt(3) * (1 - np.sqrt(3) * a))

def fam():
    yield 'Gamma*-support', sorted(Sstar)
    old = {59: [1, 4, 12, 18, 25, 27], 113: [1, 5, 12, 15, 38, 46], 211: None}.get(p)
    if old: yield 'old paper support', old
    for L in (3, 5, 8):
        yield f'AP{{1..{L}}}', list(range(1, L + 1))
    rest = [k for k in range(5, h + 1)]
    yield 'AP{1..4}+3 random', sorted([1, 2, 3, 4] + list(rng.choice(rest, 3, replace=False)))
    for t in range(2):
        base = list(rng.choice(np.arange(1, h + 1), 4, replace=False))
        pl = {fold(base[0] + base[1], p), fold(base[0] + base[2], p)}
        yield f'planted 2 sums #{t}', sorted(set(int(b) for b in base) | pl)
    for t in range(2):
        yield f'random K={len(Sstar)} #{t}', sorted(int(k) for k in rng.choice(np.arange(1, h + 1), len(Sstar), replace=False))
    # multiplicative cosets: subgroup of order m of Z_p^*, folded to {1..h}
    g = next(x for x in range(2, p) if all(pow(x, (p - 1) // q, p) != 1 for q in {q for q in range(2, p) if (p - 1) % q == 0 and all(q % r for r in range(2, q))}))
    for m in sorted(m for m in range(4, 20) if (p - 1) % m == 0):
        G = {pow(g, (p - 1) // m * i, p) for i in range(m)}
        H = sorted({fold(x, p) for x in G})
        if 3 <= len(H) <= 10:
            yield f'mult subgroup order {m}', H
    if h <= 30:
        yield 'dense', list(range(1, h + 1))

out = open(f'gap_p{p}.jsonl', 'a')
for name, T in fam():
    t0 = time.time()
    K = len(T)
    rec = dict(p=p, name=name, T=[int(x) for x in T], K=K, Gamma_star=Gam)
    rec['diss_dim'] = dissociated_dimension(p, T) if K <= 16 else None
    rec['shortest_rel'] = shortest_relation(p, T, jmax=5 if K <= 10 else 3)
    rec['energy'] = energy(p, T)
    a1, _ = depth_polar(p, T, False, rng, starts=10, iters=150)
    a2, _ = depth_polar(p, T, True, rng, starts=10, iters=150)
    rec['A_star'] = a1; rec['A_s'] = a2; rec['c_lo'] = transfer(a2, a1)
    rec['Gamma_H'] = gamma_H(p, T, rng, starts=6)[0]
    for eps in (0.01, 0.03):
        if eps * K > 0.9:
            continue
        rec[f'ratio_opt_eps{eps}'] = floored_ratio(p, T, delta(a2), eps, Gam, starts=15)
        rec[f'ratio_pess_eps{eps}'] = floored_ratio(p, T, delta(rec['c_lo']), eps, Gam, starts=15)
    rec['secs'] = time.time() - t0
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
