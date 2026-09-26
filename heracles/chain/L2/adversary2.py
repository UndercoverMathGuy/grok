# Bisection for the numerical truth eps_true(p, R): largest eps with no light configuration beating Gamma*/sqrt3.
# Relaxed optimiser (progopt), then PROJECT to a genuine configuration: a frequency with x + u >= eps is heavy (block rate),
# otherwise light (rate R). The projected value is a genuine (non-relaxed) configuration value.
import sys, json, numpy as np
from progopt import solve, cover
from lemma_eps import eps_ref
from polish import polish
p = int(sys.argv[1]); h = (p - 1) // 2; rng = np.random.default_rng(7)
hb = json.load(open(f'heavy_{p}.json')); G = hb['Gamma_lb']; x0 = np.array(hb['x']); A = cover(p)
f = A @ (x0 ** 1.5 / np.sqrt(3)); bind = np.nonzero(f <= f.min() * (1 + 1e-6))[0]
score = A[bind].min(0); score[x0 > 1e-7] = -1; order = np.argsort(-score)
def seeds(eps):
    out = []
    for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, h):
        m = min(m, h)
        if m * eps >= 1: continue
        u = np.zeros(h); u[order[:m]] = eps; out.append(np.r_[x0 * (1 - m * eps), u])
    u = np.full(h, min(eps, 1 / h)); out.append(np.r_[np.zeros(h), u / u.sum()] if u.sum() >= 1 else np.r_[x0 * (1 - u.sum()), u])
    return out
def project(z, eps, R):
    t = z[:h] + z[h:]
    amp = np.where(t >= eps, t ** 1.5 / np.sqrt(3), R * t ** 1.5)
    return (A @ amp).min()
out = []
for R in (1.0, 25 / 16, 4.0):
    global_G = G
    e_c = G ** 2 / (12 * R ** 2); e_r = eps_ref(p, G, R)
    lo, hi = e_c, 16 * e_c; rows = []
    for it in range(7):
        eps = np.sqrt(lo * hi) if it else lo
        v, z = solve(p, eps, R, 8, rng, x0s=seeds(eps), maxiter=400)
        # safeguard: the relaxed optimiser can find a better HEAVY spectrum than G (Gamma*_lb too low). Re-optimise heavy-only
        # from the found shares and raise G if it improves; a violation must beat the best heavy value.
        t = z[:h] + z[h:]; t = t / t.sum()
        vh, zh = solve(p, 0.0, 1.0, 0, rng, x0s=[np.r_[t, np.zeros(h)]], maxiter=800)
        if vh * np.sqrt(3) > G:
            G = float(vh * np.sqrt(3)); x0[:] = zh[:h]
            print(f"   (Gamma*_lb raised to {G:.6f})", flush=True)
        rel = v / (G / np.sqrt(3)); proj = polish(p, z, eps, R)[0] / (G / np.sqrt(3))
        rows.append(dict(eps=eps, relaxed=rel, projected=proj))
        print(f"p={p} R={R:.3f} eps={eps:.5f} (c={eps / G ** 2:.4f}): relaxed {rel:.6f}  genuine(polished) {proj:.6f}", flush=True)
        if it == 0: continue
        if proj > 1 + 1e-6: hi = eps
        else: lo = eps
    out.append(dict(p=p, R=R, Gamma=G, c_clean=1 / (12 * R ** 2), c_ref=e_r / G ** 2, c_true_lo=lo / G ** 2, c_true_hi=hi / G ** 2, rows=rows))
    print(f"==> p={p} R={R:.3f}: proven c_clean {1 / (12 * R ** 2):.4f}, c_ref {e_r / G ** 2:.4f}; numerical c_true in [{lo / G ** 2:.4f}, {hi / G ** 2:.4f}]", flush=True)
json.dump(out, open(f'adversary2_{p}.json', 'w'), indent=1)
