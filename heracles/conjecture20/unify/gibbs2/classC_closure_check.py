# Sanity checks of the class-C closure at n = 2 with the sharp L2 payment c(s) = (1 - 3^{1-s})/sqrt3:
#  (K1) payment, per feature of random class-C networks:  |beta_l| - a_l <= c(s) R_l;
#  (K2) chain:  h <= sum_l a_l k_l + c(s) sum_{L'} R_l k_l - mbar_{delta*};
#  (K3) decoupling:  h <= W(theta) = min_delta sum_l a_l kappa_l(delta)  whenever  c(s)^2 sum_{L'} k_l < (p-1)/(p+1);
#  (K4) a local adversarial search (random-restart hill climbing on atoms, 1-2 atoms per feature, all features present)
#       for max (h - W)/W at p = 29, 31 (b = 2): must stay <= 0 by the theorem.
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classC_net import net_from_atoms, analyse
rng = np.random.default_rng(11)
def cs(s): return (1 - 3**(1 - s))/np.sqrt(3)
def random_atoms(p, mode):
    atoms = []
    for l in range(1, (p - 1)//2 + 1):
        if mode == 'sparse' and rng.random() < 0.4: continue
        amp = rng.exponential()
        if rng.random() < 0.6:
            k = int(rng.integers(1, 3))
            for _ in range(k):
                atoms.append((l, amp/k*np.exp(0.3j*rng.normal()), np.exp(2j*np.pi*rng.random()), np.exp(2j*np.pi*rng.random())))
        else:
            for k3 in range(3):
                z = np.exp(2j*np.pi*k3/3)
                atoms.append((l, amp/3*(1 + 0.15*rng.normal()), z*np.exp(0.15j*rng.normal()), z*np.exp(0.15j*rng.normal())))
    return atoms
for (p, b) in [(11, 2.0), (13, 2.0), (17, 2.0), (19, 2.5), (23, 2.0), (29, 2.0), (31, 2.0), (31, 2.5), (37, 2.0)]:
    s = 3/b; c = cs(s)
    wK1 = np.inf; wK2 = np.inf; wK3 = np.inf; nK3 = 0; hpos = 0
    for t in range(30):
        atoms = random_atoms(p, 'sparse' if t % 2 else 'full')
        if not atoms: continue
        U, W, fr = net_from_atoms(p, atoms)
        out = analyse(U, W, fr, p, b=b)
        k = out['K'][out['dstar']]
        cheap = np.abs(out['beta']) > out['a'] + 1e-12
        wK1 = min(wK1, np.min(c*out['R'] - (np.abs(out['beta']) - out['a'])))
        mbar = 0.5*(out['m'][out['dstar']] + out['m'][(-out['dstar']) % p])
        rhs = (out['a']*k).sum() + c*(out['R'][cheap]*k[cheap]).sum() - mbar
        wK2 = min(wK2, rhs - out['h'])
        if c**2*k[cheap].sum() < (p - 1)/(p + 1):
            nK3 += 1; wK3 = min(wK3, out['W'] - out['h'])
        hpos += out['h'] > 0
    print(f"p={p} b={b}: (K1) min[c R - (|beta|-a)] = {wK1:.3e}  (K2) min[chain - h] = {wK2:.3e}  "
          f"(K3) min[W - h] = {wK3:.3e} over {nK3} nets meeting the condition;  h>0 in {hpos}/30", flush=True)
# (K4) local adversarial search for (h - W)/W
def objective(params, p, layout, b):
    atoms = []; i = 0
    for (l, k) in layout:
        for _ in range(k):
            amp, ph, a1, a2 = params[i:i + 4]; i += 4
            atoms.append((l, np.exp(amp)*np.exp(1j*ph), np.exp(1j*a1), np.exp(1j*a2)))
    U, W, fr = net_from_atoms(p, atoms)
    out = analyse(U, W, fr, p, b=b)
    return (out['h'] - out['W'])/out['W'], out
for p in [29, 31]:
    best = -np.inf
    for rep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
        layout = [(l, int(rng.integers(1, 3))) for l in range(1, (p - 1)//2 + 1)]
        npar = 4*sum(k for _, k in layout)
        x = np.concatenate([[rng.normal()*0.3, 0.2*rng.normal(), rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)] for _ in range(npar//4)])
        val, _ = objective(x, p, layout, 2.0); step = 0.3
        for it in range(150):
            cand = x + step*rng.normal(size=npar)*(rng.random(npar) < 0.1)
            v, _ = objective(cand, p, layout, 2.0)
            if v > val: x, val = cand, v
            else: step *= 0.99
        best = max(best, val)
    print(f"(K4) p={p} b=2: best (h - W)/W found by local search = {best:.4f}  (theorem: <= 0)", flush=True)
