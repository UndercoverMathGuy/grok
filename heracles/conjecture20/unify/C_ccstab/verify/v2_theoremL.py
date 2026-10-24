# Referee check of Theorem L (and L1, L2 => L) on broad class-C families, independent implementation.
import numpy as np, math, itertools
from vlib import *
rng = np.random.default_rng(77)

def intervals(n):
    return [tuple(1 if t1 < t + 1 <= t2 else 0 for t in range(n)) for t2 in range(1, n + 1) for t1 in range(t2)]

def ledger(net, S):
    p, n, nu, s = net.p, net.n, net.nu, net.s; r = 1 / s; F = (p - 1) // 2; B = Bp(p, n)
    g = net.gaps(); h = g.min()
    Phi = net.fields(); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    at = net.atoms(); A = np.abs(at.sum(0)); L = np.abs(at).sum(0); Q = (np.abs(at) ** r).sum(0)
    a = nu ** (1 - s) * Q ** s
    Iv = intervals(n)
    aminus = np.zeros(F); EI = np.zeros(F); EIall = np.zeros(F)
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
    for l in range(1, F + 1):
        bT = {T: net.signflip_atoms(l, T).sum() for T in Ts}
        EIall[l - 1] = sum(abs(v) ** 2 for v in bT.values())
        EI[l - 1] = sum(abs(bT[T]) ** 2 for T in Iv)
        if L[l - 1] > 0:
            tau = math.sqrt(max(2 * L[l - 1] * (L[l - 1] - A[l - 1]), 0))
            Vp = 2 / (nu * L[l - 1] ** 2) * sum((abs(bT[T]) + tau) ** 2 for T in Iv)
            aminus[l - 1] = L[l - 1] * max((1 + Vp) ** (-(s - 1)), nu ** (1 - s))
    assert np.allclose(EIall, E, rtol=1e-9, atol=1e-12 * L.sum() ** 2), (EIall, E)   # class-C variance lemma
    K = kappa(p); act = [l - 1 for l in S]
    d0 = int(np.argmin(K[:, act] @ a[act])); dm = p - 2 - d0
    O = max(np.ptp(g[:, d0]), np.ptp(g[:, dm]))
    det = (K[d0] * E).sum() / O if O > 0 else 0.0
    M = (K[d0, act] * a[act]).sum()
    X = (det - (K[d0, act] * (A[act] - aminus[act])).sum()) / M
    GamS = programme_S(p, s, S); gIF = nu ** (1 - s) * B * GamS
    P = net.cost()
    return dict(h=h, X=X, M=M, pay=np.min(a[act] - aminus[act]) / L.sum(), bound=(1 - X) * M - h,
                gam=h / P ** s, gIF=gIF, Mbound=nu ** (1 - s) * GamS * B * P ** s - M)

def rand_net(p, n, b):
    nu = n + 1; F = (p - 1) // 2
    S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False).tolist())
    specs = []
    noise = rng.choice([0.0, 0.05, 0.2, 0.6, 1.5])
    for l in S:
        lam = rng.uniform(0.5, 1.5)
        for k in range(nu):
            al = np.pi * k / nu
            amps = lam * np.exp(noise * 0.5 * rng.normal(size=nu))
            phis = al + noise * rng.normal(size=n)
            psi = phis.sum() + noise * rng.normal()          # independent output phase: atoms NOT aligned
            specs.append((l, amps, phis, psi))
        for k in range(rng.integers(0, 3)):
            specs.append((l, lam * noise * np.exp(rng.normal(size=nu)), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi)))
    # ghost feature: two neurons with opposite atoms (beta_l = 0) but different xi  -> E_l > 0, l not in S
    ghosts = [l for l in range(1, F + 1) if l not in S]
    if ghosts and rng.random() < 0.5 and noise > 0:
        l = ghosts[0]; amp = 0.3 * noise * np.ones(nu); ph1 = rng.uniform(0, 2 * np.pi, n); ph2 = rng.uniform(0, 2 * np.pi, n)
        specs.append((l, amp, ph1, ph1.sum())); specs.append((l, amp, ph2, ph2.sum() + np.pi))
    net = classC(p, n, specs, b)
    beta = np.abs(net.atoms().sum(0)); Sact = [l for l in range(1, F + 1) if beta[l - 1] > 1e-10 * beta.max()]
    return net, Sact

def main():
  res = dict(n=0, pos=0, minbound=np.inf, minpay=np.inf, minquant=np.inf, maxratio_Xpos=0, Xneg_pos=0, ghost=0, minMb=np.inf)
  for (p, n, b) in [(3, 2, 2.0), (3, 3, 2.0), (3, 5, 2.0), (5, 2, 2.0), (5, 3, 2.0), (5, 4, 2.0), (7, 2, 2.0), (7, 2, 2.6), (5, 3, 3.3), (11, 2, 2.0), (7, 3, 2.0)]:
      for t in range(40 if p ** n < 2000 else 15):
          net, S = rand_net(p, n, b)
          if not S: continue
          d = ledger(net, S); res['n'] += 1
          res['minbound'] = min(res['minbound'], d['bound'] / d['M'])
          res['minpay'] = min(res['minpay'], d['pay'])
          res['minMb'] = min(res['minMb'], d['Mbound'] / d['M'])
          if d['h'] > 0:
              res['pos'] += 1
              res['minquant'] = min(res['minquant'], (1 - d['X']) - d['gam'] / d['gIF'])
              if d['X'] >= 0: res['maxratio_Xpos'] = max(res['maxratio_Xpos'], d['gam'] / d['gIF'])
              else: res['Xneg_pos'] += 1
  print(res)

if __name__ == '__main__':
    main()
