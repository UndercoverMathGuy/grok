"""Exact rational certificates that Conjecture 20 is false (p = 3, b = 2, n = 11 and n = 13).

Product network (paper, Sec. 2):  f(x)[c] = sum_i w_i(c) prod_{t=1}^{n} u_i(x_t),  x in Z_3^n,  label y = sum_t x_t mod 3.
Each neuron uses the same input factor u_i in all n slots, so f(x) depends on x only through the counts
N_a = #{t : x_t = a}; the minimum over the (n+1)(n+2)/2 count vectors equals the minimum over all 3^n inputs.

All weights are 7-decimal rationals and every quantity below is computed exactly with fractions.Fraction:
    h        = min_x min_{c != y(x)} ( f(x)[y(x)] - f(x)[c] )                       (margin, Sec. 3)
    P_2      = sum_i ( n ||u_i||^2 + ||w_i||^2 ) = ||theta||_{2,2}^2
    gamma    = h / P_2^{nu/2}                                                      (normalized margin, nu = n+1)
    gamma_IF = nu^{1-s} B_3 Gamma_s(3),  s = nu/2,  B_3 = 2 (6 nu)^{-nu/2},  Gamma_s(3) = 3/2   (Theorem 8)
             = 3 nu (6 nu^2)^{-nu/2}
For n odd, nu/2 is an integer, so gamma and gamma_IF are rational and  gamma > gamma_IF  is decided exactly.
Run:  python3 exact_certificate.py      (standard library only)
"""
from fractions import Fraction as Fr

NETWORKS = {
    11: {  # width 8  (nu = 12)
        "u": [("-0.4846467", "0.0986351", "2.3735957"), ("-0.9492087", "0.4121459", "2.1787251"),
              ("-1.2303519", "0.7873084", "1.915895"), ("-1.3685236", "1.1830027", "1.577257"),
              ("-1.3684776", "1.5721532", "1.186851"), ("-1.230377", "1.9114843", "0.7892217"),
              ("-0.9512944", "2.1719362", "0.4152123"), ("-0.4871743", "2.3639747", "0.1000316")],
        "w": [("0.9287971", "1.0496311", "-1.9784282"), ("-0.7213797", "-1.2263075", "1.9476873"),
              ("0.4609319", "1.4256721", "-1.886604"), ("-0.1610123", "-1.6108253", "1.7718376"),
              ("-0.1574121", "1.7692451", "-1.611833"), ("0.4583909", "-1.8837781", "1.4253872"),
              ("-0.717387", "1.9434179", "-1.2260309"), ("0.9243126", "-1.9711611", "1.0468485")],
    },
    13: {  # width 10 (nu = 14)
        "u": [("-0.4216991", "0.0741045", "2.3968740"), ("-0.8161270", "0.2921794", "2.2790904"),
              ("-1.1038905", "0.5802922", "2.0996357"), ("-1.2981542", "0.9056724", "1.8605446"),
              ("-1.3956983", "1.2405298", "1.5701614"), ("-1.3964372", "1.5691030", "1.2427003"),
              ("-1.3005131", "1.8592842", "0.9096283"), ("-1.1082066", "2.0993605", "0.5849830"),
              ("-0.8279848", "2.2705893", "0.3019431"), ("-0.4283569", "2.3824187", "0.0770962")],
        "w": [("-1.9687294", "0.7451127", "1.2236167"), ("1.7636163", "-0.0817546", "-1.6818618"),
              ("-1.3442330", "-0.6032933", "1.9475262"), ("0.8235997", "1.1613496", "-1.9849493"),
              ("-0.2701486", "-1.5741396", "1.8442882"), ("-0.2979786", "1.8553360", "-1.5573574"),
              ("0.8482616", "-1.9887992", "1.1405376"), ("-1.3639519", "1.9443337", "-0.5803819"),
              ("1.7615922", "-1.6800124", "-0.0815798"), ("-1.9580485", "1.2182119", "0.7398366")],
    },
}


def check(n, net):
    p, nu = 3, n + 1
    assert nu % 2 == 0
    U = [[Fr(a) for a in row] for row in net["u"]]
    W = [[Fr(a) for a in row] for row in net["w"]]
    m = len(U)
    h, where = None, None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1
            y = (N1 + 2 * N2) % p
            z = [U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in range(m)]
            f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(p)]
            for c in range(p):
                if c != y and (h is None or f[y] - f[c] < h):
                    h, where = f[y] - f[c], (N0, N1, N2, c)
    P2 = sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(m))
    gamma = h / P2 ** (nu // 2)
    gamma_IF = Fr(3 * nu) / Fr(6 * nu * nu) ** (nu // 2)
    ok = h > 0 and gamma > gamma_IF
    print("n = %d, width %d, nu = %d" % (n, m, nu))
    print("  margin h          = %.15f   (binding at counts %s, wrong class %d)" % (float(h), where[:3], where[3]))
    print("  P_2               = %.15f" % float(P2))
    print("  gamma = h/P_2^%d   = %.15e" % (nu // 2, float(gamma)))
    print("  gamma_IF (Thm 8)  = %.15e" % float(gamma_IF))
    print("  gamma / gamma_IF  = %.12f      exact check gamma > gamma_IF: %s" % (float(gamma / gamma_IF), ok))
    return ok


if __name__ == "__main__":
    results = [check(n, net) for n, net in NETWORKS.items()]
    assert all(results)
    print("Conjecture 20 fails: both networks beat the interference-free value (exact rational arithmetic).")
