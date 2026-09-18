"""audit_construct.py -- exact checks of the junk-free constructions and of the attainment constants.
(1) gated chain (prod): doc's 2^n population with phases in {0, pi/2}; and a CHEAPER n+1 population with a common
    phase phi = pi l/(n+1) on every operand (not in the doc).  (2) x^2: doc's 16 (4x4 grid) vs a 6-neuron cyclic family
    (phi_1, phi_2) = 2 pi l (1, 4)/6.  (3) x^3 on the 3-sum: smallest cyclic family found by search.
Reports junk energy, gamma / gamma_bound, where gamma_bound = 2 c_n (2p)^{-nu/2} K^{-nu/2} M_S  (the support's own
equal-amplitude bound), so the printed ratio is the attainment constant.  uv run python heracles/audit_construct.py"""
import itertools, math
import numpy as np, torch
from audit_common import rows, logits, margin, clean, clean_table, Gmat, cn


def build(p, n, S, phases, act, amp=1.0):
    """phases: list of n-tuples; readout phase = sum.  Balanced: every factor has the same l2 norm."""
    a = torch.arange(p) * 2 * math.pi / p
    E = torch.zeros(n, len(S) * len(phases), p); W = torch.zeros(len(S) * len(phases), p)
    i = 0
    for k in S:
        for ph in phases:
            for j in range(n):
                E[j, i] = amp * torch.cos(k * a + ph[j])
            W[i] = amp * torch.cos(k * a + sum(ph))
            i += 1
    return E, W


def report(tag, p, n, S, phases, act):
    ops, y = rows(p, n); nu = n + 1
    E, W = build(p, n, S, phases, act)
    z = logits(E, W, ops, act); g = clean(z, y, p)
    zc = z - z.mean(1, keepdim=True); gt = clean_table(g, y, p); gt = gt - gt.mean(1, keepdim=True)
    junk = float((zc - gt).square().sum() / zc.square().sum())
    mg = float(margin(z, y).min()); n2 = float(E.square().sum() + W.square().sum())
    gam = mg / n2 ** (nu / 2)
    MS = Gmat(p)[[k - 1 for k in S]].sum(0).min()
    bound = 2 * cn(n, act) * (2 * p) ** (-nu / 2) * len(S) ** (-nu / 2) * MS
    print(f"{tag:34s} p{p} n{n} {act} K{len(S)} n_k={len(phases):3d} junk={junk:.2e} margin={mg:+.4f} "
          f"gamma/bound={gam / bound:.4f}  predicted n_k^-(nu-2)/2={len(phases) ** (-(nu - 2) / 2):.4f}")
    return junk, gam / bound


def cyclic_search(n, act, Gmax=40):
    """smallest cyclic family phi_j = 2 pi l c_j / G (l = 0..G-1) killing every non-clean term.
    Term list: t in Z^n (operand multiples), readout sign -1 after global-sign normalisation; phase = sum_j (t_j - 1) phi_j."""
    if act == "prod":
        T = [t for t in itertools.product((1, -1), repeat=n) if any(x != 1 for x in t)]
    else:  # x^n: monomial exponents m_j, t_j in {-m_j, -m_j+2, .., m_j}
        T = set()
        for m in itertools.product(range(n + 1), repeat=n):
            if sum(m) != n: continue
            for t in itertools.product(*[range(-mj, mj + 1, 2) for mj in m]):
                T.add(t); T.add(tuple(-x for x in t))       # both readout signs
        T = [t for t in T if any(x != 1 for x in t)]
    for G in range(2, Gmax):
        for c in itertools.product(range(G), repeat=n):
            if all(sum((tj - 1) * cj for tj, cj in zip(t, c)) % G != 0 for t in T):
                return G, c
    return None


if __name__ == "__main__":
    p = 13; S = [1, 2, 5]
    for n in (2, 3):
        ph = [tuple(x) for x in itertools.product((0, math.pi / 2), repeat=n)]
        report("prod doc 2^n {0,pi/2}", p if n == 2 else 7, n, S if n == 2 else [1, 2], ph, "prod")
        ph = [tuple([math.pi * l / (n + 1)] * n) for l in range(n + 1)]
        report("prod common-phase n+1 (NEW)", p if n == 2 else 7, n, S if n == 2 else [1, 2], ph, "prod")
    n = 4; ph = [tuple([math.pi * l / (n + 1)] * n) for l in range(n + 1)]
    report("prod common-phase n+1 (NEW)", 5, 4, [1, 2], ph, "prod")
    ph = [tuple(x) for x in itertools.product((0, math.pi / 2), repeat=4)]
    report("prod doc 2^n", 5, 4, [1, 2], ph, "prod")
    # x^2
    ph = [(math.pi / 2 * i, math.pi / 2 * j) for i in range(4) for j in range(4)]
    report("pow x^2 doc 4x4 grid (16)", p, 2, S, ph, "pow")
    for n in (2, 3):
        G, c = cyclic_search(n, "pow")
        print(f"pow n={n}: smallest cyclic junk-free family G={G}, c={c}")
        ph = [tuple(2 * math.pi * l * cj / G for cj in c) for l in range(G)]
        report(f"pow x^{n} cyclic G={G} (NEW)", p if n == 2 else 7, n, S if n == 2 else [1, 2], ph, "pow")
    for n in (2, 3, 4):
        print("prod cyclic search n", n, cyclic_search(n, "prod"))
    # single neuron / too-few-neuron sanity: doc says 1 and 4 neurons leave junk + negative margin at x^2
    report("pow x^2 one neuron per freq", p, 2, S, [(0.0, 0.0)], "pow")
    report("prod one neuron per freq", p, 2, S, [(0.0, 0.0)], "prod")
