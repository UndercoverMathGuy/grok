"""Convention audit of cert_hs.cert_data / eval_cells against a direct computation from q(x,delta):
a0 = 25 Chat_1(0), a_T = 25 Chat_1(-chi_T), n = 25 ||Chat over non-own||; and the rigorous cell bound vs direct float bound.
Then an end-to-end check on actual networks with beta_2 = 0 using my own torus/conjugation/swap normalisation."""
import sys, itertools, numpy as np
sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
from cert_hs import cert_data, eval_cells
from iv import dn
from hs2 import elim_ratio
from pay_adapt import payment
rng = np.random.default_rng(5)
p = 5; rho = np.exp(2j*np.pi/p)
X = np.array(list(itertools.product(range(p), repeat=2)))
E = [(a, b) for a in range(3) for b in range(3)]
c = {j: np.array([1 - rho**(-j*d) for d in (1, 2, 3, 4)]) for j in (1, 2)}
def direct(Gs):
    M = sum(np.trace(G).real for G in Gs)
    q = np.zeros((p, p, 4))
    for x1, x2 in X:
        th = 2*np.pi*2*np.array([x1, x2])/p
        v = np.array([np.exp(1j*(e[0]*th[0]+e[1]*th[1])) for e in E])
        for d in range(4): q[x1, x2, d] = np.real(np.conj(v) @ Gs[d] @ v)/25/M
    Cj = {j: np.einsum('abd,d->ab', q, c[j]) for j in (1, 2)}
    Chat = lambda j, psi: np.mean([Cj[j][x1, x2]*rho**(-(psi[0]*x1+psi[1]*x2)) for x1, x2 in X])
    a0 = 25*Chat(1, (0, 0))
    aT = [25*Chat(1, ((-chi[0]) % p, (-chi[1]) % p)) for chi in [(2, 0), (0, 2), (2, 2)]]
    own = [(0, 0), (2, 0), (0, 2), (2, 2)]; n2 = 0
    for j in (1, 2):
        for chi in itertools.product(range(p), repeat=2):
            if (j == 1 and chi in own) or (j == 2 and chi == (0, 0)): continue
            n2 += abs(25*Chat(j, ((-chi[0]) % p, (-chi[1]) % p)))**2
    return a0, aT, np.sqrt(n2), q
def inside(iv_, val, tol=1e-13):
    return iv_.lo - tol <= val <= iv_.hi + tol
bad = 0
for t in range(20):
    Gs = []
    for d in range(4):
        Z = rng.normal(size=(9, 4)) + 1j*rng.normal(size=(9, 4)); Gs.append(Z @ Z.conj().T)
    cd = cert_data(Gs)
    a0, aT, n, q = direct(Gs)
    ok = inside(cd['a0'][0], a0.real) and inside(cd['a0'][1], a0.imag) and inside(cd['n'], n)
    for k in range(3): ok = ok and inside(cd['aT'][k][0], aT[k].real) and inside(cd['aT'][k][1], aT[k].imag)
    bad += (not ok)
print(f"cert_data vs direct Fourier computation: {20-bad}/20 enclosures contain the direct values")
# rigorous cell bound vs direct float bound at points
mx = 0
for t in range(15):
    al = rng.uniform(0, 1.2); m = rng.uniform(0, 1.2, 3); ph = rng.uniform(-np.pi, np.pi)
    F = 0.7
    val, Gs = elim_ratio(al, m[0], m[1], m[2], ph, F, deg=2, ret=True)
    cd = cert_data(Gs)
    rig = eval_cells(cd, np.array([al]), np.array([al]), m[None, :], m[None, :], np.array([ph]), np.array([ph]), np.array([F]))[0]
    a0, aT, n, q = direct(Gs)
    A = np.exp(1j*al); bT = [m[0], m[1], m[2]*np.exp(1j*ph)]
    a = np.real(np.conj(A)*a0) + sum(np.real(np.conj(bT[k])*aT[k]) for k in range(3))
    fl = np.sqrt(max(a, 0)**2/F**2 + 12*n**2)/(2.5*2**-1.5)
    mx = max(mx, abs(rig-fl))
    assert rig >= fl - 1e-12
print(f"eval_cells point bound vs direct float bound: max |diff| = {mx:.2e} (rigorous >= float: ok)")
