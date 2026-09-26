"""gopt4_junk.py -- how much does SAME-|k| JUNK actually buy in the cost?

Question (statement S2 of the GLOBAL_OPT4 brief).  The robust version of the
peeling inequality only gives

    sum_{l not in S} |c_l|  >=  2^{-h} |A| - (1 - 2^{-h}) J,

i.e. junk of size J buys Theta(J) of cost relief, which beats the only known
margin penalty O(J/p).  That is an upper bound on the relief, not the relief.
Here we measure the truth: for a single frequency,

    Cmin_r(J) := min { sum_l |c_l|^r :  C_empty = 1,  |C_T| <= J for all T != empty }

over all numbers of atoms N and all unit phases v_lj.  If Cmin_r(J) =
(n+1)^{1-r} - alpha J + O(J^2) with alpha > 0 the obstruction is real; if the
linear term vanishes the obstruction is an artefact of the peeling relaxation.

All float64, local CPU.
"""
import json
import argparse
import itertools
import numpy as np
from scipy.optimize import minimize

EPS2 = 1e-30  # smoothing inside |c|^r, contributes <= N*1e-10 to the objective


def subsets(n):
    out = []
    for k in range(n + 1):
        for T in itertools.combinations(range(n), k):
            out.append(T)
    return out


def pack(cr, ci, phi):
    return np.concatenate([cr, ci, phi.ravel()])


def unpack(x, N, n):
    cr = x[:N]
    ci = x[N:2 * N]
    phi = x[2 * N:].reshape(N, n)
    return cr, ci, phi


def obj_and_grad(x, N, n, r):
    cr, ci, phi = unpack(x, N, n)
    m2 = cr ** 2 + ci ** 2 + EPS2
    f = np.sum(m2 ** (r / 2.0))
    g = np.zeros_like(x)
    d = r * m2 ** (r / 2.0 - 1.0)
    g[:N] = d * cr
    g[N:2 * N] = d * ci
    return f, g


def moments(x, N, n, Ts):
    """C_T for every T, plus the pieces needed for gradients."""
    cr, ci, phi = unpack(x, N, n)
    c = cr + 1j * ci
    S = np.array([[phi[l, list(T)].sum() if len(T) else 0.0 for T in Ts]
                  for l in range(N)])          # (N, nT)
    E = np.exp(1j * S)                          # (N, nT)
    C = (c[:, None] * E).sum(axis=0)            # (nT,)
    return c, E, C


def make_problem(N, n, r, J, Ts):
    nT = len(Ts)
    i_empty = 0  # Ts[0] == ()

    def f(x):
        return obj_and_grad(x, N, n, r)[0]

    def fp(x):
        return obj_and_grad(x, N, n, r)[1]

    def eq(x):
        c, E, C = moments(x, N, n, Ts)
        return np.array([C[i_empty].real - 1.0, C[i_empty].imag])

    def eq_jac(x):
        c, E, C = moments(x, N, n, Ts)
        Jm = np.zeros((2, x.size))
        e = E[:, i_empty]
        Jm[0, :N] = e.real
        Jm[0, N:2 * N] = -e.imag          # d Re(c_l e)/d ci = Re(i e) = -Im e
        Jm[1, :N] = e.imag
        Jm[1, N:2 * N] = e.real
        # empty T does not involve phases
        return Jm

    def ineq(x):
        """J^2 - |C_T|^2 >= 0 for T != empty."""
        c, E, C = moments(x, N, n, Ts)
        return J ** 2 - np.abs(C[1:]) ** 2

    def ineq_jac(x):
        c, E, C = moments(x, N, n, Ts)
        Jm = np.zeros((nT - 1, x.size))
        for a, T in enumerate(Ts[1:]):
            e = E[:, a + 1]
            CT = C[a + 1]
            # d|C|^2/dcr_l = 2 Re(conj(C) e_l);  d/dci_l = 2 Re(conj(C) i e_l)
            Jm[a, :N] = -2.0 * (np.conj(CT) * e).real
            Jm[a, N:2 * N] = -2.0 * (np.conj(CT) * 1j * e).real
            for j in T:
                Jm[a, 2 * N + np.arange(N) * n + j] = \
                    -2.0 * (np.conj(CT) * (1j * c * e)).real
        return Jm

    return f, fp, eq, eq_jac, ineq, ineq_jac


def block_start(N, n, rng, noise=0.0):
    """(n+1)-block on the first n+1 atoms, zeros (or noise) on the rest."""
    nb = n + 1
    cr = np.zeros(N)
    ci = np.zeros(N)
    cr[:nb] = 1.0 / nb
    phi = rng.uniform(0, 2 * np.pi, size=(N, n))
    for l in range(nb):
        phi[l, :] = -2.0 * np.pi * l / nb
    x = pack(cr, ci, phi)
    if noise:
        x = x + noise * rng.standard_normal(x.size)
    return x


def solve(N, n, r, J, starts, rng, Ts):
    f, fp, eq, eq_jac, ineq, ineq_jac = make_problem(N, n, r, J, Ts)
    cons = [{'type': 'eq', 'fun': eq, 'jac': eq_jac}]
    if len(Ts) > 1:
        cons.append({'type': 'ineq', 'fun': ineq, 'jac': ineq_jac})
    best = np.inf
    bestx = None
    for s in range(starts):
        if s == 0:
            x0 = block_start(N, n, rng)
        elif s < 4:
            x0 = block_start(N, n, rng, noise=10 ** (-s))
        else:
            cr = rng.standard_normal(N) / N
            ci = rng.standard_normal(N) / N
            phi = rng.uniform(0, 2 * np.pi, size=(N, n))
            x0 = pack(cr, ci, phi)
        try:
            res = minimize(f, x0, jac=fp, constraints=cons, method='SLSQP',
                           options={'maxiter': 600, 'ftol': 1e-14})
        except Exception:
            continue
        if not res.success:
            continue
        # re-verify feasibility honestly
        if np.max(np.abs(eq(res.x))) > 1e-8:
            continue
        if len(Ts) > 1 and np.min(ineq(res.x)) < -1e-9:
            continue
        v = f(res.x)
        if v < best:
            best, bestx = v, res.x
    return best, bestx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--starts', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    n = args.n
    nu = n + 1
    r = args.q / nu
    Ts = subsets(n)
    rng = np.random.default_rng(args.seed)
    target = (n + 1) ** (1 - r)

    Js = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.06, 0.1, 0.2, 0.3, 0.5]
    rows = []
    print(f"n={n} nu={nu} q={args.q} r={r:.6f}   (n+1)^(1-r) = {target:.12f}")
    print(f"{'J':>8} " + " ".join(f"N={N:<14}" for N in args.Ns))
    for J in Js:
        vals = []
        for N in args.Ns:
            v, _ = solve(N, n, r, J, args.starts, rng, Ts)
            vals.append(v)
        best = min(vals)
        rows.append({'J': J, 'vals': vals, 'best': best,
                     'rel': best / target})
        print(f"{J:8.4g} " + " ".join(f"{v:<16.12f}" for v in vals)
              + f"  rel={best/target:.12f}")

    # local slope estimate at J=0
    out = {'n': n, 'q': args.q, 'r': r, 'target': target, 'Ns': args.Ns,
           'rows': rows}
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
