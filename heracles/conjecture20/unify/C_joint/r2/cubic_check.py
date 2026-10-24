# Checks for the cubic detection lemma and the class-C classification of cubic relations.
# (a) random laws: m(sigma^2 - m^2) <= S, and m >= sigma when S <= 0; moment hierarchy m >= sqrt(M4/M2) if M3=0, M5<=0.
# (b) class-C fluctuations f_l = sum_T B_{l,T} rho^{2 l x_T}: E_x d_delta^3 equals the sum over (A) common-set additive
#     triples and (B) single-feature disjoint-union triples; zero when neither exists.  Parseval E d^2 = sum kappa E.
import numpy as np, itertools
rng = np.random.default_rng(0)
worst = -1e9; worst2 = 1e9
for t in range(20000):
    N = rng.integers(2, 12); x = rng.normal(size=N)*rng.exponential(size=N); w = rng.dirichlet(np.ones(N))
    x -= w @ x; m = -x.min(); s2 = w @ x**2; S = w @ x**3
    worst = max(worst, (m*(s2 - m*m) - S)/max(s2**1.5, 1e-12))
    if S <= 0: worst2 = min(worst2, m/np.sqrt(s2))
print(f"(a) max [m(sigma^2-m^2) - S]/sigma^3 over 20000 random laws = {worst:.3e} (<= 0);  min m/sigma given S<=0 = {worst2:.4f} (>= 1)")
# symmetric laws (all odd moments 0): hierarchy check
worst3 = 1e9
for t in range(5000):
    N = rng.integers(1, 8); a = rng.exponential(size=N); w = rng.dirichlet(np.ones(N))
    x = np.concatenate([a, -a]); ww = np.concatenate([w, w])/2
    m = -x.min(); M2 = ww @ x**2; M4 = ww @ x**4
    worst3 = min(worst3, m/np.sqrt(M4/M2))
print(f"    symmetric laws: min m / sqrt(M4/M2) = {worst3:.4f} (>= 1)")
def classC(p, n, fam, rng):
    F = (p-1)//2; X = np.indices((p,)*n).reshape(n, -1).T
    coef = {(l, T): (rng.normal()+1j*rng.normal()) for (l, T) in fam}
    return F, X, coef
def d_delta(p, X, coef, d):
    tot = np.zeros(len(X))
    for (l, T), B in coef.items():
        xT = X[:, list(T)].sum(1) % p
        tot += np.real(B*np.exp(2j*np.pi*2*l*xT/p)*(1-np.exp(-2j*np.pi*l*d/p)))
    return tot
def predicted_S(p, n, coef, d):
    # d = (1/2) sum_c (a_c chi_c + conj); E d^3 = (1/8) sum over ordered triples of signed characters with zero sum
    items = []
    for (l, T), B in coef.items():
        a = B*(1-np.exp(-2j*np.pi*l*d/p)); v = np.zeros(n, int); v[list(T)] = 2*l
        items.append((a, v % p)); items.append((np.conj(a), (-v) % p))
    tot = 0
    for (a1, v1), (a2, v2), (a3, v3) in itertools.product(items, repeat=3):
        if not np.any((v1+v2+v3) % p): tot += a1*a2*a3
    return np.real(tot)/8
def types_present(p, fam):
    # (A) common set, eps_a l_a + eps_b l_b + eps_c l_c = 0 ; (B) same l, T_a = T_b disjoint-union T_c
    A = any(T1 == T2 == T3 and any((e1*l1+e2*l2+e3*l3) % p == 0 for e1 in (1,-1) for e2 in (1,-1) for e3 in (1,-1))
            for (l1,T1) in fam for (l2,T2) in fam for (l3,T3) in fam)
    B = any(l1 == l2 == l3 and not (set(T2) & set(T3)) and set(T1) == set(T2) | set(T3)
            for (l1,T1) in fam for (l2,T2) in fam for (l3,T3) in fam)
    return A, B
maxerr = 0; zero_ok = 0; zero_tot = 0; parerr = 0
for trial in range(60):
    p = [5, 7, 11, 13][trial % 4]; n = [2, 3][trial % 2]; F = (p-1)//2
    sets = [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]
    K = rng.integers(2, 5)
    if trial % 3 == 0:   # single-set family, random distinct sets
        fam = [(int(l), sets[rng.integers(len(sets))]) for l in rng.choice(np.arange(1, F+1), size=min(K, F), replace=False)]
    else:                # multi-set
        fam = list({(int(rng.integers(1, F+1)), sets[rng.integers(len(sets))]) for _ in range(K+2)})
    _, X, coef = classC(p, n, fam, rng)
    A, B = types_present(p, fam)
    for d in range(1, p):
        dd = d_delta(p, X, coef, d)
        S = np.mean(dd**3); Sp = predicted_S(p, n, coef, d)
        maxerr = max(maxerr, abs(S - Sp)/(np.mean(dd**2)**1.5 + 1e-12))
        Epred = sum(abs(B)**2*(1-np.cos(2*np.pi*l*d/p)) for (l, T), B in coef.items())
        parerr = max(parerr, abs(np.mean(dd**2) - Epred)/Epred)
        if not A and not B:
            zero_tot += 1; zero_ok += abs(S) < 1e-9*np.mean(dd**2)**1.5
print(f"(b) |E d^3 - triple formula|/sigma^3 max = {maxerr:.2e};  Parseval rel err {parerr:.2e};  "
      f"no (A)/(B) triples => E d^3 = 0 in {zero_ok}/{zero_tot} (delta, network) cases")
