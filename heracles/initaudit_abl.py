"""initaudit_abl.py -- H1(d): the INDEPENDENT interface-ablation count K_abl on the init-scale cells.
K_abl counts a frequency k only if projecting cos/sin(2 pi k x/p) out of BOTH embeddings and the readout drops the
margin above a null floor (max drop over 32 random low-energy frequencies) AND the cut hurts on each side alone
(read-and-write rule); harmonics are folded into their fundamental.  It never looks at the logit spectrum, so it is
immune to the clean-energy normalisation that K_2pct/K_5pct use."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsum_ablation import Net, ablation_count  # noqa: E402

REP = ROOT / "FINAL_RUNS" / "REP"
OUT = ROOT / "FINAL_RUNS" / "INITAUDIT" / "ablation_init.json"
CELLS = [("n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init1_m500_st40000", 71, 1),
         ("n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init2_m500_st40000", 71, 2),
         ("n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000", 71, 4),
         ("n2_pow_p59_l2q2_lam0.0001_lrmax12.8_init1_m500_st40000", 59, 1),
         ("n2_pow_p59_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000", 59, 4),
         ("n2_pow_p71_l2q2_lam1e-05_lrmax12.8_init1_m500_st60000_from0.0001", 71, 1)]
SEEDS = [100, 101, 102, 103, 104, 105]
done = {r["name"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
for cell, p, init in CELLS:
    for s in SEEDS:
        rd = REP / cell / f"seed{s}"
        name = f"{cell}/seed{s}"
        if name in done or not (rd / "final.safetensors").exists(): continue
        r = ablation_count(Net(rd)); r["init"] = init
        done[name] = r
        OUT.write_text(json.dumps(list(done.values()), indent=1))
print()
print("== K_abl (interface ablation) vs the logit-spectrum counts, medians over seeds 100-105 ==")
print("  cell                                               | K_abl        | K_raw | K_2pct | K_5pct | N_.05 | clean")
g = defaultdict(list)
for r in done.values():
    g[r["name"].split("/")[0]].append(r)
for cell, p, init in CELLS:
    rs = g.get(cell, [])
    if not rs: continue
    m = st.median
    print(f"  {cell[:50]:50s} | {m([r['K_abl'] for r in rs]):4} [{min(r['K_abl'] for r in rs)}-{max(r['K_abl'] for r in rs)}] | "
          f"{m([r['K_raw'] for r in rs]):5} | {m([r['K_2pct'] for r in rs]):6} | {m([r['K_5pct'] for r in rs]):6} | "
          f"{m([r['N_05'] for r in rs]):5} | {m([r['clean_frac'] for r in rs]):.3f}")
    print(f"      per-seed K_abl: " + " ".join(f"s{r['seed']}:{r['K_abl']}" for r in sorted(rs, key=lambda r: r['seed'])))
