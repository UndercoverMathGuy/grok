"""Driver for the D3c referee check (replaces the main loop of v3_kkt.py, whose NNLS step stalled).
Local maximisers from random Gaussian starts (find_local_max of v3_kkt.py); KKT measure q on the near-active gap
constraints by bounded least squares (q >= 0, lambda >= 0, sum q = 1 enforced by a heavy row); then
   t_i = E_q g(unit factors of neuron i)  vs  h nu^{nu/2} a_i^{r-1}   (unit b-cost),
   F = log gamma + (s-1)H(w)  vs  E_w log t - (nu/2) log nu."""
import numpy as np, sys, os, time
from scipy.optimize import lsq_linear
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, 'v3_kkt.py')).read()
src = src[:src.index('for (p, n, b, m, seed) in')]
ns = {'__file__': os.path.join(HERE, 'v3_kkt.py')}
exec(compile(src, 'v3', 'exec'), ns)
from vlib import *

CASES = [(5, 2, 2.0, 6, 1), (5, 2, 2.5, 7, 2), (7, 2, 2.2, 8, 3), (5, 3, 2.4, 8, 4), (5, 2, 2.0, 9, 5),
         (7, 2, 2.6, 6, 6), (3, 3, 2.0, 6, 7), (5, 2, 2.3, 6, 8), (5, 2, 2.0, 6, 9)]
SEL = [int(a) for a in sys.argv[1:]] or range(len(CASES))
for (p, n, b, m, seed) in [CASES[k] for k in SEL]:
    t0 = time.time()
    nu = n + 1; s = nu / b; r = 1 / s
    v, g, J, c, gc = ns['find_local_max'](p, n, b, m, np.random.default_rng(seed))
    U, W = ns['unpackv'](v, m, n, p)
    h = g.min()
    act = g <= h + 1e-6 * abs(h)
    Ja = J[act]; k = int(act.sum())
    A = np.vstack([np.hstack([Ja.T, -gc[:, None]]), 1e3 * np.concatenate([np.ones(k), [0]])[None, :]])
    rhs = np.concatenate([np.zeros(J.shape[1]), [1e3]])
    sol = lsq_linear(A, rhs, bounds=(0, np.inf), lsmr_tol='auto', max_iter=20000).x
    q = sol[:-1]; lam = sol[-1]
    stat = np.linalg.norm(Ja.T @ q - lam * gc) / np.linalg.norm(lam * gc)
    a = acost(U, W); keep = a > 1e-8 * a.max()
    t = []
    for i in np.nonzero(keep)[0]:
        Ui = U[i:i + 1] / np.linalg.norm(U[i], axis=1)[None, :, None]; Wi = W[i:i + 1] / np.linalg.norm(W[i])
        gi = gaps(table(Ui, Wi), p, n).ravel()
        t.append((q * gi[act]).sum())
    t = np.array(t); ak = a[keep]
    pred = h * nu ** (nu / 2) * ak ** (r - 1)
    w = ak ** r / (ak ** r).sum()
    Fv = np.log(h) + (s - 1) * shannon(w)
    beta = clean(table(U, W), p, n)
    print('p=%d n=%d b=%.2f m=%d: h=%.4e (unit cost) ; #active=%d ; stationarity rel resid %.1e ; lambda b/nu / h = %.6f ;'
          ' max|t/pred-1| = %.2e ; F-[E_w log t-(nu/2)log nu] = %.1e ; neuron sizes a_i/max in [%.3f,1] ; '
          '#features=%d ; %.0fs' % (p, n, b, m, h, k, stat, lam * b / nu / h, np.max(np.abs(t / pred - 1)),
                                    Fv - ((w * np.log(t)).sum() - nu / 2 * np.log(nu)), (ak / ak.max()).min(),
                                    int((abs(beta) > 1e-6 * abs(beta).max()).sum()), time.time() - t0), flush=True)
