"""Per-run rescoring of the Adam-eps dial (runs_eps, 192 runs + runs_sweep/base):
is the committee count K set by WHICH REGIME the frequency-carrying tensors
were in while memorisation ran?

Per run: t_mem = first epoch with train acc >= 0.999 (spectra tap, 100-epoch
grid), t_sw = per-tensor eps-crossing epoch (first snapshot with median |g| <
eps, from notes/eps_dial_scored.json), t_grok. Bins by the E-tensor crossing:
  A  t_sw <= t_mem            (memorisation finished in the GD-like regime)
  B  t_mem < t_sw <= t_grok   (sign regime through memorisation, GD by takeover)
  C  t_sw >  t_grok           (sign regime through takeover)
Prediction under the "sign-phase seed" account (eps_dial_results.md reading
2): mean K in A < B = C. Under the "sign regime at takeover" account (the
Xie-Li l_inf reading): A = B < C. Pooled permutation tests + within-cell
(fixed eps) comparison where both bins are populated.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from score_eps_dial import perm_test

sc = json.loads((ROOT / "notes" / "eps_dial_scored.json").read_text())
rows = []
for r in sc["runs"]:
    rd = ROOT / "runs_eps" / r["cell"] / r["run"]
    z = np.load(rd / "spectra.npz")
    ep = z["epochs"]; ta = z["train_acc"]
    hit = np.flatnonzero(ta >= 0.999)
    t_mem = int(ep[hit[0]]) if len(hit) else None
    r = dict(r, t_mem=t_mem)
    rows.append(r)
print(f"{len(rows)} runs; t_mem missing: {sum(r['t_mem'] is None for r in rows)}")
tens = ["E", "V", "O", "U", "in", "out"]
for T in tens:
    A, B, C = [], [], []
    for r in rows:
        if r["acc"] < 0.99 or r["t_mem"] is None: continue
        sw = r["cross_g"].get(T)
        sw = 10**9 if sw is None else sw
        (A if sw <= r["t_mem"] else B if sw <= r["grok"] else C).append(r)
    def h(rs):
        ks = [r["K"] for r in rs]; return {k: ks.count(k) for k in sorted(set(ks))}, (np.mean(ks) if ks else float("nan"))
    hA, mA = h(A); hB, mB = h(B); hC, mC = h(C)
    print(f"\n[{T}] A(sw<=mem) n={len(A)} K={hA} mean {mA:.2f} | B(mem<sw<=grok) n={len(B)} K={hB} mean {mB:.2f} | C(sw>grok) n={len(C)} K={hC} mean {mC:.2f}")
    if A and B:
        d, p = perm_test([r["K"] for r in B], [r["K"] for r in A]); print(f"   B>A: delta {d:+.2f} p={p:.4f}")
    if B and C:
        d, p = perm_test([r["K"] for r in C], [r["K"] for r in B]); print(f"   C>B: delta {d:+.2f} p={p:.4f}")
    if A and C:
        d, p = perm_test([r["K"] for r in C], [r["K"] for r in A]); print(f"   C>A: delta {d:+.2f} p={p:.4f}")
# within-cell: for each eps cell, split by whether E crossed before t_mem
print("\nWithin-cell (E tensor), split at t_mem:")
for cell in sorted(set(r["cell"] for r in rows)):
    rs = [r for r in rows if r["cell"] == cell and r["acc"] >= 0.99 and r["t_mem"] is not None]
    a = [r["K"] for r in rs if (r["cross_g"]["E"] or 10**9) <= r["t_mem"]]
    b = [r["K"] for r in rs if (r["cross_g"]["E"] or 10**9) > r["t_mem"]]
    tm = [r["t_mem"] for r in rs]; sw = [r["cross_g"]["E"] for r in rs]
    line = f"  {cell:7s} t_mem med {np.median(tm):6.0f} (range {min(tm)}-{max(tm)}) sw_E {sw[0]}: before-mem n={len(a)} meanK {np.mean(a) if a else float('nan'):.2f} | after-mem n={len(b)} meanK {np.mean(b) if b else float('nan'):.2f}"
    if a and b:
        d, p = perm_test(b, a); line += f"  after>before delta {d:+.2f} p={p:.3f}"
    print(line)
# continuous: K vs (t_sw - t_mem) sign, and K vs t_mem within base cell
out = [dict(run=r["run"], cell=r["cell"], eps=r["eps"], K=r["K"], t_mem=r["t_mem"], grok=r["grok"], cross_g=r["cross_g"]) for r in rows]
(ROOT / "notes" / "eps_crossover_test.json").write_text(json.dumps(out, indent=0))
print("wrote notes/eps_crossover_test.json")
