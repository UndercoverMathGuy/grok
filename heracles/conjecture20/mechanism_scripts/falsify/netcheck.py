# Full-network check (n=2, b=2) of a relations-model configuration.
#   IF feature j, field amplitude A:   3 neurons  u1=u2=t cos(2pi j x/p + al_i), w = t cos(2pi j c/p + 2 al_i + eps_j), al_i = pi i/3
#       -> logits (3/4) t^3 cos(2pi j (y-c)/p - eps_j)     (A=(3/4)t^3)
#   pair feature l, clean V, interference phase th:  2 neurons
#       (phi1,phi2,psi)=(0,0,th/2) and (pi/2,pi/2,th/2+pi), each scaled by t
#       -> logits (1/2) t^3 [cos(2pi l (y-c)/p - eps_l) + cos(2pi l (y+c)/p + th)]   (V=(1/2)t^3)
#   margin h = min_{x in Z_p^2} min_{c != y} f(x)[y]-f(x)[c];  norm^2 = sum_i ||omega_i||^2;  gamma = h / norm^3;
#   gamma_IF = 3^{-1/2} * 2 (6p)^{-3/2} * Gamma_{3/2}(p).
import numpy as np, itertools

def build(p, feats):
    """feats: list of (kind, l, amp, th, eps).  kind 'IF' (amp=A) or 'pair' (amp=V).  returns list of neurons (u1,u2,w)."""
    a = np.arange(p); N = []
    for kind, l, amp, th, eps in feats:
        if amp <= 0: continue
        if kind == 'IF':
            t = (4 * amp / 3) ** (1 / 3)
            for i in range(3):
                al = np.pi * i / 3
                u = t * np.cos(2 * np.pi * l * a / p + al)
                N.append((u, u.copy(), t * np.cos(2 * np.pi * l * a / p + 2 * al + eps)))
        else:
            t = (2 * amp) ** (1 / 3)
            ps = th / 2
            # rotate the clean phase by eps: psi -> psi+eps on output would also rotate the interference; instead shift
            # input phases: (phi1,phi2) -> (phi1+e/2, phi2+e/2) rotates clean by -e... keep it explicit below
            for (f1, f2, psi) in [(0.0, 0.0, ps), (np.pi / 2, np.pi / 2, ps + np.pi)]:
                # clean term phase: phi1+phi2-psi ; interference phase: phi1+phi2+psi.  Want clean=-eps... we use
                # phi shift q on both inputs: clean -> 2q - psi, interf -> 2q + psi.  Solve 2q-psi0 = -eps, 2q+psi0 = th.
                q = (th - eps) / 4; psi0 = (th + eps) / 2
                N.append((t * np.cos(2 * np.pi * l * a / p + f1 + q), t * np.cos(2 * np.pi * l * a / p + f2 + q),
                          t * np.cos(2 * np.pi * l * a / p + psi0 + (psi - ps))))
    return N

def margin(p, N):
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
    y = (x1 + x2) % p
    f = np.zeros((p * p, p))
    for u1, u2, w in N: f += (u1[x1] * u2[x2])[:, None] * w[None, :]
    fy = f[np.arange(p * p), y]
    g = fy[:, None] - f; g[np.arange(p * p), y] = np.inf
    return g.min()

def gamma(p, N):
    h = margin(p, N); P = sum(np.sum(u1 ** 2) + np.sum(u2 ** 2) + np.sum(w ** 2) for u1, u2, w in N)
    return h, P, h / P ** 1.5

def gamma_if(p, G):
    return 3 ** -0.5 * 2 * (6 * p) ** -1.5 * G
