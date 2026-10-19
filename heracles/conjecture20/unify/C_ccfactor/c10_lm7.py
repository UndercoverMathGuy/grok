# C_ccfactor c10: is the E_kkt (7,2,2) network k6 (canonical CD_prop = 1.0098) a numerical local maximiser of gamma?
import sys, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
import lib as GL
rng = np.random.default_rng(29)
for f in ['lm_p7_n2_b2.0_k6', 'lm_p5_n2_b2.0_k4']:
    d = np.load('/home/user/grok/heracles/conjecture20/unify/E_kkt/' + f + '.npz')
    U0, W0 = d['U'], d['W']; base = GL.Net(U0, W0, 2.0).gamma()
    out = []
    for sc in [1e-3, 1e-2, 3e-2]:
        for t in range(2):
            U = U0 + sc * rng.normal(size=U0.shape) * np.abs(U0).mean(); W = W0 + sc * rng.normal(size=W0.shape) * np.abs(W0).mean()
            out.append(GL.optimise(U, W, 2.0).gamma() / base)
    print(f'{f}: gamma after re-optimisation from perturbations (1e-3..3e-2) / gamma = {np.round(out, 6)}')
