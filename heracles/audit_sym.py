"""audit_sym.py -- independent check of THEORY_relu sec 3 (iv): ||Xbar||_* = p D_X, ||Omega_bar||_* = D_Omega / p,
z_Nbar = clean projection of N, margin(Nbar) >= margin(N).  Explicit stacking, p = 7, random ReLU net."""
import numpy as np
rng = np.random.default_rng(0); p, m = 7, 4
U, V, Om = rng.normal(size=(m, p)), rng.normal(size=(m, p)), rng.normal(size=(p, m))
def net(U, V, Om): return np.einsum("ci,iab->abc", Om, np.maximum(U[:, :, None] + V[:, None, :], 0))
def marg(z):
    a, b = np.meshgrid(range(p), range(p), indexing="ij"); y = (a + b) % p
    zy = np.take_along_axis(z, y[..., None], 2)[..., 0]; zz = z.copy(); np.put_along_axis(zz, y[..., None], -np.inf, 2)
    return (zy - zz.max(2)).min()
Ub, Vb, Ob = [], [], []
for s in range(p):
    for t in range(p):
        Ub.append(np.roll(U, -s, 1)); Vb.append(np.roll(V, -t, 1)); Ob.append(np.roll(Om, -(s + t), 0) / p**2)
Ub, Vb, Ob = np.vstack(Ub), np.vstack(Vb), np.hstack(Ob)
z, zb = net(U, V, Om), net(Ub, Vb, Ob)
a, b = np.meshgrid(range(p), range(p), indexing="ij")
g = np.array([np.mean([z[i, j, (i + j - D) % p] for i in range(p) for j in range(p)]) for D in range(p)])
print("Nbar logits == clean projection:", np.abs(zb - g[(a[..., None] + b[..., None] - np.arange(p)) % p]).max())
print("margin N, Nbar:", marg(z), marg(zb))
F = np.fft.fft(np.eye(p)) / np.sqrt(p)
def D(M):      # M (rows, p): ||M Pi_0||_F + sqrt2 sum_k ||M Pi_k||_F
    Mh = M @ F; e = (np.abs(Mh) ** 2).sum(0); nf = (p - 1) // 2
    return np.sqrt(e[0]) + np.sqrt(2) * sum(np.sqrt(e[k] + e[p - k]) for k in range(1, nf + 1))
X, Xb = np.vstack([U, V]), np.vstack([Ub, Vb])
print("||Xbar||_* vs p D_X:", np.linalg.norm(Xb, "nuc"), p * D(X), "  L_X =", D(X) / np.linalg.norm(X, "nuc"))
print("||Ombar||_* vs D_Om/p:", np.linalg.norm(Ob, "nuc"), D(Om.T) / p, "  L_Om =", D(Om.T) / np.linalg.norm(Om, "nuc"))
