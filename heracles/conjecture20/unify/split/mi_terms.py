"""Master-inequality bookkeeping for a network theta and its split (point-mass tests):
   sum_delta mu_delta min_x g(.,delta) = IF*P^s + bonus_mu - Det_mu - U - G      (identity)
and hidden_mu = sum_delta mu_delta [min_x g_theta - min_x g_split] = Det_mu(split) - Det_mu(theta),
DeltaU = U(theta) - U(split) = gamma_IF (P^s - P_split^s).  mu = uniform on Delta(theta) = argmin_delta sum_l a_l kappa_l(delta).
usage: python mi_terms.py net.npz index p n b
"""
import sys, numpy as np
import lib
from netopt import programme, gamma_IF, kappa_mat, own
f, k = sys.argv[1], int(sys.argv[2]); p, n, b = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
nu = n + 1; s = nu / b; r = 1 / s
Gam, y = programme(p, s); gIF = gamma_IF(p, n, b, Gam); Bp = lib.Bp(p, n); Cb = Bp ** (1 / s)
K = kappa_mat(p); F = lib.Fset(p)
d = np.load(f); U, W = d[f'U{k}'], d[f'W{k}']


def terms(U, W, mu=None):
    P = lib.cost(U, W, b)
    at = own(U, W, p, n)
    beta = np.array([at[j].sum() for j in F]); Q = np.array([(np.abs(at[j]) ** r).sum() for j in F])
    a = nu ** (1 - s) * Q ** s
    cov_a = K @ a
    Delta = np.where(cov_a <= cov_a.min() * (1 + 1e-9))[0]          # indices delta-1
    if mu is None:
        mu = np.zeros(p - 1); mu[Delta] = 1; mu[(p - 2 - Delta)] = 1; mu /= mu.sum()   # symmetrise
    Kmu = mu @ K
    G_ = lib.gaps_from_table(lib.table(U, W), p, n)
    ming = G_[:, 1:].min(0); Eg = G_[:, 1:].mean(0)
    Det = mu @ (Eg - ming)
    bonus = ((beta.real - a) * Kmu).sum()
    Ucost = nu ** (1 - s) * ((Cb * P) ** s - Q.sum() ** s) * Gam
    Gcov = nu ** (1 - s) * Q.sum() ** s * Gam - cov_a.min()
    lhs = mu @ ming
    rhs = gIF * P ** s + bonus - Det - Ucost - Gcov
    return dict(P=P, h=G_[:, 1:].min(), Delta=(Delta + 1).tolist(), mu=mu, bonus=bonus, Det=Det, U=Ucost, G=Gcov,
                identity_err=abs(lhs - rhs), IFPs=gIF * P ** s, cheap=(np.abs(beta) / a).round(4).tolist(),
                min_by_delta=ming, Eg=Eg)


t = terms(U, W)
Us, Ws = lib.split(U, W)
ts = terms(Us, Ws, mu=t['mu'])
sc = t['IFPs']
print(f"theta: gamma/gIF={t['h'] / sc:.4f}  Delta(theta)={t['Delta']}  |beta|/a={t['cheap']}  identity err={t['identity_err']:.2e}")
print(f"  in units of IF*P^s:  bonus={t['bonus'] / sc:.4f}  Det={t['Det'] / sc:.4f}  U={t['U'] / sc:.4f}  G={t['G'] / sc:.4f}")
print(f"split: gamma/gIF(at P)={ts['h'] / sc:.4f}  Det(split)={ts['Det'] / sc:.4f}  U(split)={ts['U'] / sc:.4f}  "
      f"(same bonus {ts['bonus'] / sc:.4f}, same G {ts['G'] / sc:.4f})")
hidden = ts['Det'] - t['Det']; dU = t['U'] - ts['U']
print(f"hidden = {hidden / sc:.4f}  (fraction of Det(split): {hidden / ts['Det']:.3f});  DeltaU = {dU / sc:.5f}")
print(f"class-C MI slack of split = Det+U+G-bonus = {(ts['Det'] + ts['U'] + ts['G'] - ts['bonus']) / sc:+.4f};  "
      f"MI slack of theta = {(t['Det'] + t['U'] + t['G'] - t['bonus']) / sc:+.4f}")
