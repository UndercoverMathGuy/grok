"""What is being counted, verified on the 192 cohort-M/MG finals (p=113 MLPs).

Instrument as used everywhere (train_semifinal_torch.project / margin_polish.spectrum):
  logits L(a,b,c) on the full p^2 grid ->
  coeff_k  = <L, cos(k(a+b-c))>   (the clean mod-add term at frequency k; the sin(k(a+b-c))
             phase is NOT stored — checked here)
  energy_k = ||proj of L onto span{cos(k(a+b)), sin(k(a+b))} over (a,b), any c-dependence||^2
  K_gap    = committee_from_coeffs: sort |coeff|, cut at the largest log-gap among the top 13, keep >= 2% of max
  K_thr    = #{k : |coeff_k| >= 0.1 max}
  k_eff    = 1 / sum(share_k^2) over energy shares
Added here (functional definitions tied to the theory's object = frequencies that carry margin):
  K_load90 = smallest K such that the top-K clean truncation sum_{k in topK} coeff_k cos(k(a+b-c))
             reaches >= 90% of the min-margin of the FULL clean table (all 56 k)
  K_acc99  = smallest K such that the top-K clean truncation is >= 99% correct on the grid
"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "nudial"))
from train_nudial import Config, forward, committee_from_coeffs
from safetensors.torch import load_file
torch.set_grad_enabled(False)
p = 113; nf = p // 2
a = torch.arange(p).repeat_interleave(p); b = torch.arange(p).repeat(p); c = torch.arange(p)
n = ((a + b)[:, None] - c[None, :]) % p                     # (p^2, p) = a+b-c
k = torch.arange(1, nf + 1, dtype=torch.float64)
COS = torch.cos(2 * math.pi * k[:, None, None] * n[None].double() / p)   # (nf, p^2, p)
SIN = torch.sin(2 * math.pi * k[:, None, None] * n[None].double() / p)
lab = (a + b) % p
def minmargin(L):
    zy = L[torch.arange(p * p), lab]; o = L.clone(); o[torch.arange(p * p), lab] = -1e30
    return float((zy - o.max(1).values).min())
def acc(L): return float((L.argmax(1) == lab).float().mean())
rows = []
for cell in sorted((ROOT / "runs_nudial" / "M").glob("*")) + sorted((ROOT / "runs_nudial" / "MG").glob("*")):
    for rd in sorted(cell.glob("*_s*")):
        cfg = Config(**{kk: v for kk, v in json.loads((rd / "config.json").read_text()).items() if kk in Config.__dataclass_fields__})
        pr = {kk: v.double()[None] for kk, v in load_file(rd / "final.safetensors").items()}
        if not all(torch.isfinite(v).all() for v in pr.values()): continue
        L = forward(pr, a, b, cfg)[0]                                # (p^2, p)
        L = L - L.mean(1, keepdim=True)
        cc = (COS * L[None]).sum((1, 2)) * 2 / (p * p * p)           # clean cos coefficient per k
        ss = (SIN * L[None]).sum((1, 2)) * 2 / (p * p * p)           # the unstored sin phase
        amp = torch.sqrt(cc ** 2 + ss ** 2)
        order = torch.argsort(amp, descending=True)
        full = sum(cc[j] * COS[j] + ss[j] * SIN[j] for j in range(nf))
        m_full = minmargin(full); m_net = minmargin(L)
        K_load, K_acc, T = None, None, torch.zeros_like(L)
        for i, j in enumerate(order):
            T = T + cc[j] * COS[j] + ss[j] * SIN[j]
            if K_acc is None and acc(T) >= 0.99: K_acc = i + 1
            if K_load is None and minmargin(T) >= 0.9 * m_full: K_load = i + 1
            if K_load and K_acc: break
        share = amp ** 2 / (amp ** 2).sum()
        rows.append(dict(cell=cell.name, run=rd.name, K_gap=len(committee_from_coeffs(cc.numpy())),
                         K_gap_amp=len(committee_from_coeffs(amp.numpy())), K_thr10=int((amp >= 0.1 * amp.max()).sum()),
                         K_thr05=int((amp >= 0.05 * amp.max()).sum()), k_eff=float(1 / (share ** 2).sum()),
                         K_load90=K_load, K_acc99=K_acc, sin_frac=float((ss ** 2).sum() / (amp ** 2).sum()),
                         clean_frac=float((amp ** 2).sum() * p ** 3 / 2 / (L ** 2).sum()), m_net_over_clean=m_net / m_full,
                         top8=[round(float(x), 3) for x in (amp[order][:8] / amp.max())]))
(ROOT / "notes" / "count_definitions.json").write_text(json.dumps(rows, indent=0))
import collections
for cell in dict.fromkeys(r["cell"] for r in rows):
    rs = [r for r in rows if r["cell"] == cell]
    h = lambda key: dict(sorted(collections.Counter(r[key] for r in rs).items()))
    print(f"\n{cell} n={len(rs)}")
    for key in ("K_gap", "K_gap_amp", "K_thr10", "K_thr05", "K_load90", "K_acc99"): print(f"  {key:9s} {h(key)}")
    print(f"  k_eff med {np.median([r['k_eff'] for r in rs]):.2f} | sin-phase energy frac med {np.median([r['sin_frac'] for r in rs]):.4f} | clean/total energy med {np.median([r['clean_frac'] for r in rs]):.3f} | net margin / clean margin med {np.median([r['m_net_over_clean'] for r in rs]):.3f}")
    print(f"  top-8 relative amplitude median {np.median([r['top8'] for r in rs], 0).round(3).tolist()}")
agree = lambda x, y: np.mean([r[x] == r[y] for r in rows]); within1 = lambda x, y: np.mean([abs(r[x] - r[y]) <= 1 for r in rows])
print(f"\nALL n={len(rows)}: K_gap==K_load90 {agree('K_gap','K_load90'):.2f} (within 1: {within1('K_gap','K_load90'):.2f}) | K_thr10==K_load90 {agree('K_thr10','K_load90'):.2f} ({within1('K_thr10','K_load90'):.2f}) | K_gap==K_thr10 {agree('K_gap','K_thr10'):.2f} ({within1('K_gap','K_thr10'):.2f}) | K_acc99==K_load90 {agree('K_acc99','K_load90'):.2f}")
