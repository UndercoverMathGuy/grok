"""initaudit_ceiling.py -- ABSOLUTE ceiling for gamma at p = 71 (and 59), x^2 net, nu = 3.

gamma(net) = (minimum margin over all p^2 rows) / ||theta||^3, where ||theta||^2 is the sum of squares of every weight.
It is scale-free: multiplying all weights by c leaves it unchanged, so it is a property of the DIRECTION only.

For each K we take the optimal support S(K) from heracles/kstar.json (the exhaustive/local argmax of the coverage
M_K) and build the exact junk-free x^2 circuit on it (audit_construct's 6-neuron cyclic family per frequency,
phases (phi_1, phi_2) = 2 pi l (1, 4)/6, l = 0..5, equal amplitude on every frequency and balanced across the three
weight tables).  That gives the exact constructed gamma_con(K) -- a LOWER bound on the true optimum over K-sparse
directions (free per-frequency amplitudes buy another 1.5-10%, audit_program.json), and the number the trained nets
should be compared against.
"""
import json, math, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audit_common import rows, logits, margin  # noqa: E402
torch.set_default_dtype(torch.float64)

KS = json.loads((ROOT / "kstar.json").read_text())
PRED = json.loads((ROOT / "rep_pred.json").read_text())


def build_x2(p, S):
    L = 6
    ph = [(2 * math.pi * l / 6, 2 * math.pi * 4 * l / 6) for l in range(L)]
    a = torch.arange(p) * 2 * math.pi / p
    m = len(S) * L
    E = torch.zeros(2, m, p); W = torch.zeros(p, m)
    i = 0
    for k in S:
        for p1, p2 in ph:
            E[0, i] = torch.cos(k * a + p1); E[1, i] = torch.cos(k * a + p2)
            W[:, i] = torch.cos(k * a + p1 + p2)
            i += 1
    return E, W


for p in (59, 71):
    ops, y = rows(p, 2)
    d = KS[str(p)]
    print(f"\n== p = {p}, x^2 (nu = 3): exact constructed ceiling per support size K ==")
    print("  K | support S(K)                         | min margin | ||th||^2 |  gamma_con   | gamma_con/max | ladder r(K)")
    out = {}
    best = 0.0
    rows_ = []
    for K in sorted(d["S"], key=int):
        S = d["S"][K]
        E, W = build_x2(p, S)
        z = logits(E, W.T, ops, "pow")
        mg = float(margin(z, y).min())
        n2 = float(E.square().sum() + W.square().sum())
        g = mg / n2 ** 1.5
        rows_.append((int(K), S, mg, n2, g))
        best = max(best, g)
    r = PRED[str(p)]["2"]["r"]
    for K, S, mg, n2, g in rows_:
        print(f" {K:>2} | {str(S):36s} | {mg:10.4f} | {n2:8.1f} | {g:.6e} | {g/best:12.4f}  | {r.get(str(K),'-')}")
    out = {str(K): g for K, S, mg, n2, g in rows_}
    (ROOT / "FINAL_RUNS" / "INITAUDIT" / f"ceiling_p{p}.json").write_text(json.dumps(out, indent=1))
