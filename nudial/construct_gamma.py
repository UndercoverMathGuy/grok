"""Constructive gamma(K) curves for the nudial relu architectures — the
"computable version" of the +1 question (notes/nudial_math.md par 8).

Builds explicit K-committee networks in the EXACT cohort-A architectures
(relu-4 layout (1,1): E -> U,V -> ReLU -> W1 -> W; relu-6 layout (2,2) via
identity-extended pre/post stacks), computes the exact full-grid margin in
f64, and normalises by ||theta||_2^nu with the closed-form optimal block
balance:

  For a chain of B scale-groups with costs c_1..c_B and margin multilinear in
  the group scales, max_alpha g*prod(alpha)/ (sum alpha_i^2 c_i)^{B/2}
                    = g / (B^{B/2} * sqrt(prod c_i)).

Neuron design per frequency k: n_a x n_b PRODUCT grid of phases (phi, psi);
preact = s*(cos(k a + phi) + cos(k b + psi)); readout rows cos/sin(phi+psi)
-> W columns cos/sin(k c). On a product grid every ReLU harmonic (m,n) with
(m-1) % n_a != 0 or (n-1) % n_b != 0 cancels exactly, so leak is controlled
by the grid size (tied diagonal grids psi=phi leak at (2,0) — also built,
for comparison). Free per-frequency amplitudes are polished by coordinate
ascent on the exact normalised margin.

Outputs notes/construct_gamma.json and a printed table:
  gamma_constructed(K) per nu, vs the trained AdamW medians and cohort-D.

Usage: uv run python nudial/construct_gamma.py
"""
import itertools, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = 113

# optimal committees from V1 (kstar_table.py, exhaustive for K<=6, local K>=7)
SETS = {1: (1,), 2: (1, 31), 3: (1, 6, 43), 4: (1, 7, 11, 43),
        5: (1, 14, 20, 44, 52), 6: (1, 5, 12, 15, 38, 46),
        7: (1, 7, 13, 29, 32, 43, 47)}


def grids(n_a, n_b, tied=False):
    if tied:
        ph = np.arange(n_a) * 2 * np.pi / n_a
        return np.stack([ph, ph], 1)
    pairs = list(itertools.product(range(n_a), range(n_b)))
    return np.array([[2 * np.pi * i / n_a, 2 * np.pi * j / n_b]
                     for i, j in pairs])


def committee_net(S, amps, m, n_a, n_b, tied=False):
    """Explicit blocks for layout (1,1): returns (E, U, V, W1, W, counts).
    amps: per-frequency relative amplitude (embedded as s_j on U/V rows)."""
    K = len(S)
    pairs = grids(n_a, n_b, tied)
    n = len(pairs)
    assert K * n <= m, (K, n, m)
    x = np.arange(P)
    d = 2 * K
    E = np.zeros((d, P))
    for j, k in enumerate(S):
        E[2 * j] = np.cos(2 * np.pi * k * x / P)
        E[2 * j + 1] = np.sin(2 * np.pi * k * x / P)
    U = np.zeros((m, d)); V = np.zeros((m, d))
    W1 = np.zeros((2 * K, m)); W = np.zeros((P, 2 * K))
    for j, k in enumerate(S):
        s = amps[j] ** 0.25          # amplitude split evenly over 4 blocks
        rows = slice(j * n, (j + 1) * n)
        phi, psi = pairs[:, 0], pairs[:, 1]
        U[rows, 2 * j] = s * np.cos(phi); U[rows, 2 * j + 1] = -s * np.sin(phi)
        V[rows, 2 * j] = s * np.cos(psi); V[rows, 2 * j + 1] = -s * np.sin(psi)
        # readout cos(k c + phi + psi): the act's (1,1) term has phase
        # +(phi+psi), so cos A cos B with B = k c + phi + psi makes the
        # Delta-term phase-free. (The docket's "cos(kc - phi - psi)" is a
        # sign slip — it averages the signal to zero.)
        W1[2 * j, rows] = s * np.cos(phi + psi) / n
        W1[2 * j + 1, rows] = -s * np.sin(phi + psi) / n
        W[:, 2 * j] = s * np.cos(2 * np.pi * k * x / P)
        W[:, 2 * j + 1] = s * np.sin(2 * np.pi * k * x / P)
        E[2 * j] *= s; E[2 * j + 1] *= s
    return E, U, V, W1, W


def full_margin(E, U, V, W1, W):
    """Exact min margin over the p^2 grid, f64."""
    a = np.repeat(np.arange(P), P); b = np.tile(np.arange(P), P)
    Xa = (U @ E)[:, a]; Xb = (V @ E)[:, b]              # (m, p^2)
    H = np.maximum(Xa + Xb, 0.0)
    L = (W @ (W1 @ H)).T                                 # (p^2, p)
    y = (a + b) % P
    zy = L[np.arange(P * P), y]
    L[np.arange(P * P), y] = -np.inf
    return float((zy - L.max(1)).min())


def gamma_balanced(E, U, V, W1, W, nu):
    """Closed-form optimally-balanced normalised margin for the group chain."""
    g = full_margin(E, U, V, W1, W)
    costs = [float((E ** 2).sum()), float((U ** 2).sum() + (V ** 2).sum()),
             float((W1 ** 2).sum()), float((W ** 2).sum())]
    B = len(costs)  # 4 scale groups; identity-extended layers add cost later
    if nu == 4:
        return g / (B ** (B / 2) * math.sqrt(np.prod(costs))), g, costs
    # relu-6 layout (2,2): insert identity maps E2 (d x d) and W2 (2K x 2K)
    # as their own scale groups; identity on the 2K used dims costs 2K each.
    d = E.shape[0]
    costs = [costs[0], float(d), costs[1], costs[2], float(d), costs[3]]
    B = 6
    return g / (B ** (B / 2) * math.sqrt(np.prod(costs))), g, costs


def polish_amps(S, m, n_a, n_b, nu, tied=False, iters=60):
    """Coordinate ascent on per-frequency log-amplitudes."""
    K = len(S)
    amps = np.ones(K)
    def ev(am):
        return gamma_balanced(*committee_net(S, am, m, n_a, n_b, tied), nu)[0]
    best = ev(amps)
    step = 0.15
    for it in range(iters):
        improved = False
        for j in range(K):
            for sgn in (1, -1):
                cand = amps.copy(); cand[j] *= math.exp(sgn * step)
                v = ev(cand)
                if v > best * (1 + 1e-12):
                    best, amps, improved = v, cand, True
        if not improved:
            step *= 0.5
            if step < 0.01:
                break
    return best, amps


def main():
    out = {}
    m = 512
    print(f"p={P} m={m}; product-grid constructions, closed-form balance")
    for nu in (4, 6):
        out[str(nu)] = {}
        print(f"\n== relu-{nu} ==   (trained AdamW medians: "
              f"{'4.4-4.9e-6 K=5' if nu == 4 else '4.0-4.7e-8 K=4'}; "
              f"cohort D: {'6.4e-7 K=4' if nu == 4 else '2.2e-9 K=3'})")
        for K in range(1, 8):
            S = SETS[K]
            n_a = n_b = max(2, int(math.floor(math.sqrt(m // K))))
            g_eq, _, _ = gamma_balanced(
                *committee_net(S, np.ones(K), m, n_a, n_b), nu)
            g_free, amps = polish_amps(S, m, n_a, n_b, nu)
            g_tied, _, _ = gamma_balanced(
                *committee_net(S, np.ones(K), m, min(m // K, 64), 0, tied=True), nu)
            out[str(nu)][K] = dict(S=S, n_a=n_a, gamma_eq=g_eq,
                                   gamma_free=g_free, amps=amps.tolist(),
                                   gamma_tied=g_tied)
            print(f" K={K} S={S}: gamma_eq={g_eq:.3e} free={g_free:.3e} "
                  f"tied={g_tied:.3e} amps={np.round(amps/amps.max(),2)}")
        row = out[str(nu)]
        ks = max(row, key=lambda k: row[k]["gamma_free"])
        print(f" argmax_K gamma_free = {ks}")
    (ROOT / "notes" / "construct_gamma.json").write_text(json.dumps(out, indent=1))
    print("\nwrote notes/construct_gamma.json")


if __name__ == "__main__":
    main()
