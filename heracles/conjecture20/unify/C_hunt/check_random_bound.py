# Random-sign designs: a_{l,T} = eps_{l,T} (Rademacher), all l in F, all T (n = 2).  Proposition: for every odd prime p,
#   theta*(p,n) <= t / L1,  t = sqrt(2 (2^n-1) p log(4 p^n)),  L1 = sum |c_l(1)| (2^n-1) = (2^n-1) cot(pi/(2p)).
# Check: (i) the identities sum_l kappa_l(1) = p/2 and sum_l 2 sin(pi l/p) = cot(pi/(2p)); (ii) sampled designs satisfy
# max_sigma(-min_x D) <= t (most draws); (iii) the first prime where the bound < c_12(b=2) = 1 - sqrt(2/3).
import numpy as np, itertools, sympy
rng = np.random.default_rng(0)
n = 2
for p in [5, 7, 11, 31, 101, 199]:
    F = (p - 1) // 2; l = np.arange(1, F + 1)
    k1 = np.sum(1 - np.cos(2 * np.pi * l / p)); s1 = np.sum(2 * np.sin(np.pi * l / p))
    X = np.indices((p, p)).reshape(2, -1).T
    Ts = [(1, 0), (0, 1), (1, 1)]
    chi = np.array([np.exp(2j * np.pi * 2 * ll * (X @ np.array(T)) / p) for ll in l for T in Ts])
    cvec = {d: np.array([1 - np.exp(-2j * np.pi * ll * d / p) for ll in l for T in Ts]) for d in (1, p - 1)}
    L1 = np.sum(np.abs(cvec[1])); t = np.sqrt(2 * 3 * p * np.log(4 * p ** 2))
    worst = []
    for draw in range(20):
        eps = rng.choice([-1.0, 1.0], len(l) * 3)
        loss = max(-(np.real((eps * cvec[d])[:, None] * chi).sum(0)).min() for d in (1, p - 1))
        worst.append(loss)
    print(f"p={p}: sum kappa={k1:.6f} (p/2={p/2}), sum 2 sin={s1:.6f} cot={1/np.tan(np.pi/(2*p)):.6f}; "
          f"sampled loss/L1 max={max(worst)/L1:.4f} median={np.median(worst)/L1:.4f} <= bound t/L1={t/L1:.4f}: {max(worst) <= t}")
c12 = 1 - np.sqrt(2 / 3)
for p in sympy.primerange(5, 5000):
    b = np.sqrt(2 * 3 * p * np.log(4 * p ** 2)) / (3 / np.tan(np.pi / (2 * p)))
    if b < c12:
        print("first prime with random-sign bound below c_12(b=2):", p, b); break
for n_ in [3, 4, 5]:
    th = 1 - (n_ / (n_ + 1)) ** ((n_ + 1) / 2 - 1)
    for p in sympy.primerange(5, 10 ** 6):
        b = np.sqrt(2 * p * np.log(4 * float(p) ** n_) / (2 ** n_ - 1)) / (1 / np.tan(np.pi / (2 * p)))
        if b < th:
            print(f"n={n_}: theta_P(b=2)={th:.4f}; first prime with bound below: {p} (bound {b:.4f})"); break
