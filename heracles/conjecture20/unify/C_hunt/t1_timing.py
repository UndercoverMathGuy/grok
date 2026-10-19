import numpy as np, time
import hlib as H
rng = np.random.default_rng(0)
for p in [17, 31, 43]:
    s = 1.5; F = (p - 1) // 2
    t = time.time(); G, y, bind = H.programme(p, s, restarts=30); tp = time.time() - t
    S = [l for l in range(1, F + 1) if y[l - 1] > 0]
    print(f"p={p} Gamma={G:.8f} S*={S} bind={bind} ({tp:.1f}s)", flush=True)
    # cheap: 2 neurons per feature on S*, random phases
    freqs = [l for l in S for _ in range(2)]
    cc = H.CC(p, 2, freqs, s)
    b = np.repeat(y[np.array(S) - 1] / 2, 2) * np.exp(1j * 0.3 * rng.standard_normal(len(freqs)))
    x = cc.pack(b, rng.uniform(0, 2 * np.pi, (len(freqs), 2)))
    t = time.time(); x1 = cc.lbfgs(x, fix_mag=True); t1 = time.time() - t
    r1 = cc.gnorm(x1) / (3 ** (1 - s) * G)
    t = time.time(); x2 = cc.lbfgs(x1); t2 = time.time() - t
    r2 = cc.gnorm(x2) / (3 ** (1 - s) * G)
    t = time.time(); x3, v = cc.slp(x2); t3 = time.time() - t
    print(f"  cheap2 on S*: phase-only {r1:.6f} ({t1:.1f}s)  full {r2:.6f} ({t2:.1f}s)  slp {v/(3**(1-s)*G):.8f} ({t3:.1f}s)",
          cc.feature_data(x3), flush=True)
    # all F, 3 neurons each, random
    freqs = [l for l in range(1, F + 1) for _ in range(3)]
    cc = H.CC(p, 2, freqs, s)
    x = cc.pack(rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs)), rng.uniform(0, 2 * np.pi, (len(freqs), 2)))
    t = time.time(); x2 = cc.lbfgs(x); t2 = time.time() - t
    t = time.time(); x3, v = cc.slp(x2); t3 = time.time() - t
    fd = cc.feature_data(x3)
    print(f"  random 3/feature on F: lbfgs {cc.gnorm(x2)/(3**(1-s)*G):.6f} ({t2:.1f}s) slp {v/(3**(1-s)*G):.8f} ({t3:.1f}s) "
          f"features={[l for l in fd if fd[l]['A'] > 1e-4 * max(d['A'] for d in fd.values())]}", flush=True)
