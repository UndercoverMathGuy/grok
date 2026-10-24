"""Referee check of E7 (mode budgets). Non-own mode mass computed by brute-force enumeration of modes.
Also: at p=3 (|F|=1) a positive constant DOES exist at b=2 (c = 1/nu), so the 'no positive constant at b=2' remark needs p>=5."""
import itertools, math
import numpy as np
from vlib import *
rng = np.random.default_rng(99)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def neuron_modes(u, w, p, n):
    """returns beta_j (dict), N (non-own mass), Nl (dict: mass of all modes except own modes of l), e (with B_p)."""
    nu = n + 1
    uh = np.fft.fft(u, axis=-1); wh = np.fft.fft(w)
    beta = {}; tot = 0.0; own = {}
    for j in F_of(p):
        beta[j] = 2 * p ** (-nu) * np.conj(wh[j]) * np.prod(uh[:, j])
        own[j] = 0.0
        for k in itertools.product(range(p), repeat=n):
            v = abs(2 * p ** (-nu) * wh[j] * np.prod([uh[t, k[t]] for t in range(n)])) ** 2
            tot += v
            if all(kt in (j, (-j) % p) for kt in k):
                own[j] += v
    N = tot - sum(own.values())
    Nl = {l: tot - own[l] for l in F_of(p)}
    e = Bp(p, n) * (np.sum(u ** 2) + np.sum(w ** 2)) ** (nu / 2)
    return beta, N, Nl, e


def rand_neuron(p, n, kind):
    nu = n + 1
    a = np.arange(p)
    if kind == 'gauss':
        u = rng.standard_normal((n, p)); w = rng.standard_normal(p)
    elif kind == 'near-pure':
        j = rng.choice(F_of(p)); leak = 10 ** rng.uniform(-4, -0.5)
        u = np.array([np.cos(2 * np.pi * j * a / p + rng.uniform(0, 6.3)) for t in range(n)]) + leak * rng.standard_normal((n, p))
        w = np.cos(2 * np.pi * j * a / p + rng.uniform(0, 6.3)) + leak * rng.standard_normal(p)
    elif kind == 'common':
        K = rng.integers(1, len(F_of(p)) + 1); fr = rng.choice(F_of(p), K, replace=False); amp = rng.dirichlet(np.ones(K))
        def v():
            return sum(math.sqrt(amp[q]) * np.cos(2 * np.pi * fr[q] * a / p + rng.uniform(0, 6.3)) for q in range(K))
        u = np.array([v() for t in range(n)]); w = v()
    else:  # unbalanced with DC
        u = rng.standard_normal((n, p)) + rng.uniform(-1, 1, (n, 1)); w = rng.standard_normal(p) * rng.uniform(0.2, 3)
    return u, w


worst1 = -1e9; worst2 = -1e9; ratio_min = 1e9
cnt = 0
for p in [3, 5, 7, 11]:
    for n in [2, 3]:
        if p ** n > 1400:
            continue
        nu = n + 1
        for kind in ['gauss', 'near-pure', 'common', 'unbal']:
            for trial in range(40):
                u, w = rand_neuron(p, n, kind)
                beta, N, Nl, e = neuron_modes(u, w, p, n)
                worst1 = max(worst1, (sum(abs(v) ** 2 for v in beta.values()) + 2 ** (-n) * N) / e ** 2 - 1)
                for b in [2.0, 2.3, (2 + nu) / 2, nu]:
                    r = b / nu; cb = (b - 2) / (2 * nu)
                    lhs = sum(abs(v) ** r for v in beta.values()) + cb * e ** (r - 2) * 2 ** (-n) * N
                    worst2 = max(worst2, lhs / e ** r - 1)
                cnt += 1
say(f"(i)  max [sum|beta|^2 + 2^-n N]/e^2 - 1 over {cnt} neurons (p in 3..11, n in 2,3): {worst1:.2e}")
say(f"(ii) max [sum|beta|^r + c_b e^(r-2) 2^-n N]/e^r - 1: {worst2:.2e}")

# b=2 example: common frequency distribution, equal factor norms, no DC
for (p, n) in [(7, 2), (11, 2), (7, 3)]:
    nu = n + 1; a = np.arange(p)
    fr = F_of(p)[:3] if len(F_of(p)) >= 3 else F_of(p); amp = np.array([0.5, 0.3, 0.2])[:len(fr)]; amp = amp / amp.sum()
    def v(ph):
        x = sum(math.sqrt(amp[q]) * np.cos(2 * np.pi * fr[q] * a / p + ph[q]) for q in range(len(fr)))
        return x / np.linalg.norm(x)
    u = np.array([v(rng.uniform(0, 6.3, len(fr))) for t in range(n)]); w = v(rng.uniform(0, 6.3, len(fr)))
    beta, N, Nl, e = neuron_modes(u, w, p, n)
    r = 2 / nu
    say(f"b=2 example (p,n)=({p},{n}): sum|beta|^(2/nu)/e^(2/nu) = {sum(abs(x)**r for x in beta.values())/e**r:.12f}, 2^-n N/e^2 = {2**(-n)*N/e**2:.6f} vs 1-sum a^nu = {1-(amp**nu).sum():.6f}")

# p = 3, b = 2: the constant 1/nu works (contradicting 'no positive constant at b=2' when |F| = 1)
for n in [2, 3, 4]:
    nu = n + 1; r = 2 / nu; worst = 1e9
    for trial in range(400):
        u, w = rand_neuron(3, n, ['gauss', 'near-pure', 'unbal'][trial % 3])
        beta, N, Nl, e = neuron_modes(u, w, 3, n)
        slack = e ** r - abs(beta[1]) ** r
        if N > 1e-14:
            worst = min(worst, slack / (e ** (r - 2) * 2 ** (-n) * N))
    say(f"p=3, n={n}, b=2: min over 400 neurons of [e^r - |beta|^r]/(e^(r-2) 2^-n N) = {worst:.4f} >= 1/nu = {1/nu:.4f}")

# (iii) network forms
w3 = -1e9; w4 = -1e9
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    nu = n + 1
    for trial in range(60):
        m = rng.integers(2, 7)
        neurons = [rand_neuron(p, n, ['gauss', 'near-pure', 'common', 'unbal'][rng.integers(4)]) for _ in range(m)]
        data = [neuron_modes(u, w, p, n) for (u, w) in neurons]
        for b in [2.0, 2.4, nu]:
            s = nu / b; r = 1 / s; cb = (b - 2) / (2 * nu)
            P = sum((np.sum(u ** 2) + np.sum(w ** 2)) ** (b / 2) for (u, w) in neurons)
            B = Bp(p, n)
            for l in F_of(p):
                Ql = sum(abs(d[0][l]) ** r for d in data)
                Rl = sum(math.sqrt(max(d[2][l], 0)) for d in data)
                w3 = max(w3, math.sqrt(Ql ** (2 * s) + 2 ** (-n) * Rl ** 2) / (B * P ** s) - 1)
            if b > 2:
                SQ = sum(sum(abs(x) ** r for x in d[0].values()) for d in data)
                R = sum(math.sqrt(max(d[1], 0)) for d in data)
                w4 = max(w4, (SQ - B ** r * P * (1 - cb * 2 ** (-n) * (R / (B * P ** s)) ** 2)) / (B ** r * P))
say(f"(iii) single own feature: max sqrt(Q^2s + 2^-n R^2)/(B P^s) - 1 = {w3:.2e} (<= 0)")
say(f"(iii) several features (b>2): max normalised excess = {w4:.2e} (<= 0)")
open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
