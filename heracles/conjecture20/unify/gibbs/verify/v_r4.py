# Referee check of R4 (joint detection, class C, n=2) on networks built from neurons; plus sawtooth network (R1 network level).
import numpy as np, sys
from netlib import *
rng = np.random.default_rng(11)
def analyse(U1, U2, W, fr, p):
    f = table(U1, U2, W); g = gaps(f); h = g[:, :, 1:].min(); co = coeffs(f)
    return f, g, h, co
def check_delta(g, co, p, d, h):
    Z = g[:, :, d] - g[:, :, d].mean(); Zm = g[:, :, (-d) % p] - g[:, :, (-d) % p].mean()
    A = -Z.min(); Ap = -Zm.min()
    m = g[:, :, d].mean() - h; mm = g[:, :, (-d) % p].mean() - h
    Fn = (p - 1)//2
    VT = [sum(abs(co[(l, T)])**2*kap(l, d, p) for l in range(1, Fn + 1)) for T in [(1,), (2,), (1, 2)]]
    # direct profile variances: D_T(w) = E[Z | 2 s_T = w]
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    VTd = []
    for sT in [x1, x2, x1 + x2]:
        w = (2*sT) % p; D = np.array([Z[w == k].mean() for k in range(p)]); VTd.append((D**2).mean())
    E = sum(VT)
    b2 = E - 2/(p + 1)*(np.sqrt(VT[0]*VT[1]) + np.sqrt(VT[0]*VT[2]) + np.sqrt(VT[1]*VT[2]))
    return dict(A=A, Ap=Ap, mm=m*mm, E=E, b2=b2, b3=(p - 1)/(p + 1)*E, Eerr=abs(E - sum(VTd))/max(E, 1e-300))
for p in [3, 5, 7, 11, 13]:
    w = dict(Eerr=0, s1=np.inf, s2=np.inf, s3=np.inf, rat=np.inf)
    for t in range(40 if p < 13 else 15):
        U1, U2, W, fr = random_classC(p, rng)
        f, g, h, co = analyse(U1, U2, W, fr, p)
        for d in range(1, p):
            c = check_delta(g, co, p, d, h)
            w['Eerr'] = max(w['Eerr'], c['Eerr'])
            w['s1'] = min(w['s1'], (c['mm'] - c['A']*c['Ap'])/c['E'])
            w['s2'] = min(w['s2'], (c['A']*c['Ap'] - c['b2'])/c['E'])
            w['s3'] = min(w['s3'], (c['b2'] - c['b3'])/c['E'])
            w['rat'] = min(w['rat'], c['A']*c['Ap']/c['E'])
    print(f"R4 p={p}: E formula (sum |beta_lT|^2 kappa vs profile variances) rel err {w['Eerr']:.1e}; min (mm'-AA')/E {w['s1']:.3e}; "
          f"min (AA'-b2)/E {w['s2']:.3e}; min (b2-b3)/E {w['s3']:.3e}; min AA'/E {w['rat']:.4f} vs (p-1)/(p+1) {(p-1)/(p+1):.4f}")
# Sawtooth network, delta = 1: realise D_1 = D_2 = r, D_12(w) = r(1 - w) by least squares on output weights.
for p in [3, 5, 7, 11, 13]:
    d = 1; Fn = (p - 1)//2
    r = ((-np.arange(p)) % p).astype(float); r -= r.mean()
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    target = r[(2*x1) % p] + r[(2*x2) % p] + r[(1 - 2*(x1 + x2)) % p]
    cols = []; specs = []
    for l in range(1, Fn + 1):
        for ph1 in (0, np.pi/2):
            for ph2 in (0, np.pi/2):
                for phw in (0, np.pi/2):
                    U1 = neuron_vec(p, l, 1, ph1)[None]; U2 = neuron_vec(p, l, 1, ph2)[None]; W = neuron_vec(p, l, 1, phw)[None]
                    g = gaps(table(U1, U2, W))[:, :, d]; cols.append((g - g.mean()).ravel()); specs.append((l, ph1, ph2, phw))
    M = np.array(cols).T; coef, *_ = np.linalg.lstsq(M, target.ravel(), rcond=None)
    res = np.abs(M @ coef - target.ravel()).max()
    # build network: weight goes into w amplitude
    U1 = np.array([neuron_vec(p, l, 1, a) for (l, a, b_, c_) in specs]); U2 = np.array([neuron_vec(p, l, 1, b_) for (l, a, b_, c_) in specs])
    W = np.array([coef[k]*neuron_vec(p, specs[k][0], 1, specs[k][3]) for k in range(len(specs))])
    f = table(U1, U2, W); g = gaps(f); co = coeffs(f); h = g[:, :, 1:].min()
    c = check_delta(g, co, p, d, h)
    print(f"sawtooth net p={p}: realisation residual {res:.1e}; A={c['A']:.6f} A'={c['Ap']:.6f} E={c['E']:.6f} AA'/E={c['A']*c['Ap']/c['E']:.8f} vs {(p-1)/(p+1):.8f}")
