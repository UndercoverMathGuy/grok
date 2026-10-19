# HUNT (a'): explicit FLAT-BOTTOM (Fejer) compensation designs, n = 2, class C. Usage:
#   python hunt_fejer.py p b Smode nrest seed      (Smode as in hunt_a: F, Sstar, AP<d>_<K>_<a>)
# Single-character cheap feature l (masses A_l (1,1,c_l)/(2+c_l), atoms aligned => gain G(c_l) fixed):
#   slot-2 phases xi_i2 = (e^{i beta}, e^{-i beta}, -1), 2 cos beta = c  (closed triangle: v-character killed);
#   'u'  : xi_i1 = e^{i phi} conj(xi_i2)^2           -> fluctuation  -(1-c) A_l e^{i phi} u^l only   (u = zeta^{x_1})
#   'uv' : xi_i1 = e^{i phi} xi_i2                   -> fluctuation  -(1-c) A_l e^{i phi} (uv)^l only
#   relative fluctuation eps_l = 1 - c_l EXACTLY (check below).  eps_l profiles: Fejer (1 - rank/(K+1)), flat, random;
#   overall level eps in {0.25, 0.5, 0.75, 1.0}; fluctuation phases phi_l: 0, pi, pi/2, -pi/2, linear in l, random.
# For each design: ratio as built; ratio after phase-only design (atoms frozen: prescribed gains); after full release.
import sys, os, json, time
import numpy as np
import hlib as H

p = int(sys.argv[1]); bexp = float(sys.argv[2]); Smode = sys.argv[3]; nrest = int(sys.argv[4]); seed = int(sys.argv[5])
n = 2; nu = 3; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
tag = f"fejer_p{p}_b{bexp}_{Smode}_s{seed}"


def red(l):
    l %= p
    return min(l, p - l)


if Smode == 'F': S = list(range(1, F + 1))
elif Smode == 'Sstar':
    G0, y0, _ = H.programme(p, s); S = [l for l in range(1, F + 1) if y0[l - 1] > 0]
else:
    d, K, a = [int(v) for v in Smode[2:].split('_')]; S = sorted(set(red(a + k * d) for k in range(K)) - {0})
GS, yS, bindS = H.programme(p, s, S=S, restarts=60)
sup = [l for l in S if yS[l - 1] > 0]
D = nu ** (1 - s) * GS
print(f"# {tag}: S={S} Gamma^S={GS:.10f} supp={sup} bind={bindS}", flush=True)


def gain(c):
    w = np.array([1.0, 1.0, c]); w = w[w > 0] / w.sum(); return 1 / (nu ** (1 - s) * np.sum(w ** r) ** s)


def design(eps, prof, chars, phmode, feats, amps=None):
    amps = yS if amps is None else amps
    K = len(feats)
    if prof == 'fejer': w = 1 - np.arange(K) / (K + 1.0); w = w[rng.permutation(K)] if rng.random() < 0.5 else w
    elif prof == 'flat': w = np.ones(K)
    else: w = rng.uniform(0.2, 1.0, K)
    epsl = np.clip(eps * w / w.max(), 0, 1)
    tau = rng.uniform(0, 2 * np.pi)
    freqs = []; b = []; phi = []
    for k, l in enumerate(feats):
        c = 1 - epsl[k]
        if c < 1e-9:     # width-2 feature: b = A/2 each; xi_2 = (1, -1); 'u': xi_1 = (1, 1); 'uv': xi_1 = xi_2
            m = np.array([0.5, 0.5]); x2 = np.array([1.0, -1.0], complex)
        else:
            beta = np.arccos(np.clip(c / 2, -1, 1)); m = np.array([1, 1, c]) / (2 + c)
            x2 = np.array([np.exp(1j * beta), np.exp(-1j * beta), -1.0])
        ph = {'0': 0.0, 'pi': np.pi, '+i': np.pi / 2, '-i': -np.pi / 2, 'lin': tau * l, 'rand': rng.uniform(0, 6.3)}[phmode]
        ch = chars if chars != 'mix' else ('u' if k % 2 == 0 else 'uv')
        x1 = np.exp(1j * ph) * (np.conj(x2) ** 2 if ch == 'u' else x2)
        for i in range(len(m)):
            freqs.append(l); b.append(amps[l - 1] * m[i]); phi.append([np.angle(x1[i]), np.angle(x2[i])])
    cc = H.CC(p, n, freqs, s)
    return cc, cc.pack(np.array(b, complex), np.array(phi)), epsl


t0 = time.time(); recs = []
# exactness check of the construction: relative fluctuation = 1 - c on the chosen character, zero on the others
cc, x, epsl = design(0.6, 'flat', 'u', '0', sup)
Phi = cc.fields(x)[0]
for k, l in enumerate(cc.feats):
    co = np.fft.fft2(Phi[k].reshape(p, p)) / p ** 2
    A = co[0, 0]; u_ = co[(-2 * l) % p, 0]; v_ = co[0, (-2 * l) % p]; uv = co[(-2 * l) % p, (-2 * l) % p]
    u2 = co[(2 * l) % p, 0]; v2 = co[0, (2 * l) % p]; uv2 = co[(2 * l) % p, (2 * l) % p]
    print(f"# check l={l}: |A|={abs(A):.6f} target eps={epsl[k]:.4f} rel fluct on chars: "
          f"{max(abs(u_), abs(u2)) / abs(A):.6f} {max(abs(v_), abs(v2)) / abs(A):.2e} {max(abs(uv), abs(uv2)) / abs(A):.2e}",
          flush=True)
best = -np.inf
for it in range(nrest):
    eps = [0.25, 0.5, 0.75, 1.0][it % 4]; prof = ['fejer', 'flat', 'rand'][(it // 4) % 3]
    chars = ['u', 'uv', 'mix'][(it // 12) % 3]; phm = ['0', 'pi', '+i', '-i', 'lin', 'rand'][it % 6]
    feats = sup if (it % 5 != 4 or len(S) == len(sup)) else sorted(set(sup) | set(rng.choice([l for l in S if l not in sup], 1).tolist()))
    yS2 = yS.copy()
    if feats != sup:
        extra = [l for l in feats if l not in sup][0]; yS2[extra - 1] = 0.3 * yS.max()
    cc, x, epsl = design(eps, prof, chars, phm, feats, yS2)
    built = cc.gnorm(x) / D
    Gs = [gain(1 - e) for e in epsl]
    free = np.r_[np.zeros(2 * cc.m, bool), np.ones(cc.m * n, bool)]
    xd = cc.lbfgs(x, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800, free=free)
    xd, vd = cc.slp(xd, iters=60, free=free)
    xr = cc.lbfgs(xd, taus=(1e-2, 3e-3, 1e-3, 3e-4), iters=800)
    xr, vr = cc.slp(xr, iters=120)
    fd = cc.feature_data(xr); Amax = max(dd['A'] for dd in fd.values())
    fe = [l for l in fd if fd[l]['A'] > 1e-6 * Amax]
    rec = dict(it=it, eps=eps, prof=prof, chars=chars, phase=phm, feats=feats, gains=[round(g, 5) for g in Gs],
               built=built, designed=vd / D, released=vr / D, rel_feats=fe, rel_gains={l: round(fd[l]['gain'], 5) for l in fe})
    recs.append(rec); best = max(best, vr / D, vd / D)
    print(json.dumps(rec, default=float), flush=True)
    if max(vr, vd) / D > 1 + 1e-9:
        np.savez(os.path.join(OUT, f"VIOL_{tag}_{it}.npz"), freqs=cc.freqs, x=xr, xd=xd, p=p, n=n, b=bexp, S=S)
byeps = {e: max([z['designed'] for z in recs if z['eps'] == e] or [None]) for e in [0.25, 0.5, 0.75, 1.0]}
print(f"# SUMMARY {tag}: best={best:.10f} best designed by eps={byeps} best built={max(z['built'] for z in recs):.6f} "
      f"time={time.time() - t0:.0f}s", flush=True)
