"""gopt_c3.py -- check of LEMMA C3 / inequality (5.2) of GLOBAL_OPT.md.

Claim.  For a pure single-frequency population with atom data (c, v), clean coefficient C_0 = sum_l c_l,
largest junk coefficient J = max over non-empty T of |C_T| (C_T = sum_l c_l prod_{j in T} v_lj), and
kappa = cot(pi/p):

    Lambda / M_1  <=  Re C_0  -  kappa ( |Im C_0| + J/2 ),        M_1 = 1 - cos(2 pi / p),
    Lambda        =  min over u in (mu_p)^n, D = 1..p-1  of  Re[(1 - e^{2 pi i D/p}) h(u)]  =  2^n margin.

Plain reading: junk of size J costs the margin kappa*J/2, and kappa ~ p/pi is large, so at a SINGLE
frequency junk is a badly losing trade as soon as p exceeds ~2 pi (2^n - 1).  This is the sharp form of
the junk penalty; feeding it into the perturbed cost bound reproduces yesterday's Theorem 2 threshold.

Proof of the claim (four lines, in GLOBAL_OPT.md Sec. 5): take the single-mode probability density
w(u) = p^{-n}(1 + 2 Re[lambda conj(u^T)]) with |lambda| = 1/2 on the grid; it is non-negative, sums to 1,
and has E_w[u^{T'}] = [T' empty] + lambda [T' = T], so E_w[h] = C_0 + lambda C_T.  Then
Lambda/M_1 <= Re E_w[h] - kappa |Im E_w[h]| by (2.1) and Jensen, and the phase of lambda is chosen so
that lambda C_T is purely imaginary and points the same way as Im C_0.
"""
import argparse, json, math, sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gopt_atom import h_on_points, block  # noqa: E402


def main(a):
    g = torch.Generator().manual_seed(a.seed)
    rows = []
    for (p, n, N) in [(7, 2, 3), (11, 2, 3), (23, 2, 3), (47, 2, 3), (7, 3, 4), (11, 3, 4), (5, 4, 5)]:
        M1 = 1 - math.cos(2 * math.pi / p)
        kap = 1.0 / math.tan(math.pi / p)
        idx = torch.cartesian_prod(*[torch.arange(p)] * n)
        u = torch.polar(torch.ones(p), torch.arange(p) * 2 * math.pi / p)
        U = u[idx.reshape(-1, n)]
        thD = torch.arange(1, p) * 2 * math.pi / p
        worst = -1e18
        for t in range(a.trials):
            if t % 2:                                   # block + noise (the interesting regime)
                c0, v0 = block(n)
                nz = 10 ** float(torch.empty(1).uniform_(-2.5, -0.2, generator=g))
                c = c0 + nz * torch.complex(torch.randn(N, generator=g), torch.randn(N, generator=g))
                v = torch.polar(torch.ones(N, n), torch.angle(v0) + nz * torch.randn(N, n, generator=g))
            else:                                       # fully random
                c = torch.complex(torch.randn(N, generator=g), torch.randn(N, generator=g))
                v = torch.polar(torch.ones(N, n), torch.rand(N, n, generator=g) * 2 * math.pi)
            H = h_on_points(c, v, U)
            Lam = float(((1 - thD.cos())[:, None] * H.real[None, :]
                         + thD.sin()[:, None] * H.imag[None, :]).min())
            C0 = complex(c.sum())
            J = 0.0
            for m in range(1, 1 << n):
                w = torch.ones(N, dtype=torch.cdouble)
                for j in range(n):
                    if m >> j & 1:
                        w = w * v[:, j]
                J = max(J, abs(complex((c * w).sum())))
            worst = max(worst, Lam / M1 - (C0.real - kap * (abs(C0.imag) + J / 2)))
        # LEMMA C4 (rigidity of realness on the grid):
        #   mean_u |Im h(u) - Im C_0|^2  ==  (1/2) sum_{T != empty} |C_T|^2   exactly,
        # hence Im h == 0 on the grid  <=>  every junk coefficient vanishes and C_0 is real.
        c4 = 0.0
        for _ in range(40):
            c = torch.complex(torch.randn(N, generator=g), torch.randn(N, generator=g))
            v = torch.polar(torch.ones(N, n), torch.rand(N, n, generator=g) * 2 * math.pi)
            H = h_on_points(c, v, U)
            C0 = complex(c.sum())
            ss = 0.0
            for m in range(1, 1 << n):
                w = torch.ones(N, dtype=torch.cdouble)
                for j in range(n):
                    if m >> j & 1:
                        w = w * v[:, j]
                ss += abs(complex((c * w).sum())) ** 2
            lhs = float(((H.imag - C0.imag) ** 2).mean())
            c4 = max(c4, abs(lhs / (0.5 * ss) - 1.0))
        rows.append(dict(p=p, n=n, N=N, max_violation=worst, C4_max_relerr=c4))
        print(f"p={p:3d} n={n} N={N}: max (Lambda/M_1 - RHS of (5.2)) = {worst:.4e}   (must be <= 0)"
              f"   | Lemma C4 identity rel err <= {c4:.2e}", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / "c3.json").write_text(json.dumps(rows, indent=1))
    print("wrote", d / "c3.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    main(ap.parse_args())
