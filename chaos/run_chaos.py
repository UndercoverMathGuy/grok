"""Forecast-ceiling experiment (option 3): how chaotic is committee SELECTION
w.r.t. tiny init perturbations?  For each seed: train the baseline init and
K perturbed copies  W -> W + eps*std(W)*N(0,1)  (same data, same everything)
for 2000 epochs (selection completes by ~ep 20-100), then compare committees.
If flip probability is already high at eps << 1, NO function of the init can
forecast the committee better than ~1 - flip/2: an intrinsic ceiling that the
observed weight-readout AUC (0.725) can be compared against.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))
import train_semifinal_torch as T
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404])
ap.add_argument("--eps", type=float, nargs="+", default=[0, 1e-4, 1e-3, 1e-2, 1e-1])
ap.add_argument("--reps", type=int, default=1, help="perturbation draws per eps")
ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--p", type=int, default=113)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="runs_chaos")
ap.add_argument("--compile", default="off")
a = ap.parse_args()
T.RUNS = Path(a.out)
T.P, T.NF = a.p, a.p // 2
torch.set_num_threads(8)

for s in a.seeds:
    cfg = T.Config(p=a.p, data_seed=s, init_seed=s, num_epochs=a.epochs,
                   save_every=a.epochs)
    base = T.init_params(cfg)
    cfgs, dirs, cks = [], [], []
    g = torch.Generator().manual_seed(s * 7919)
    for e in a.eps:
        for r in range(a.reps if e > 0 else 1):
            name = f"seed{s}/eps{e:g}_r{r}"
            d = T.RUNS / name
            if (d / "spectra.npz").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            pert = {k: (v + e * v.std() * torch.randn(v.shape, generator=g)
                        if v.numel() > 1 and v.std() > 0 else v.clone())
                    for k, v in base.items()}
            ck = d / "init.safetensors"
            save_file({k: v.contiguous() for k, v in pert.items()}, str(ck))
            cfgs.append(cfg); dirs.append(d); cks.append(ck)
    if not cfgs:
        continue
    print(f"=== seed {s}: {len(cfgs)} runs ===", flush=True)
    T.train_batched(cfgs, dirs, init_from=cks, loss="f32stable",
                    compile_mode=a.compile, device=a.device, log_every=200,
                    spectra_every=100)
    for d in dirs:
        T.report(d)

# ---- analysis
rows = []
for s in a.seeds:
    bz = np.load(T.RUNS / f"seed{s}/eps0_r0/spectra.npz")
    cb = set(T.committee_from_coeffs(bz["coeffs"][-1]))
    for e in a.eps:
        if e == 0: continue
        for r in range(a.reps):
            d = T.RUNS / f"seed{s}/eps{e:g}_r{r}"
            if not (d / "spectra.npz").exists(): continue
            z = np.load(d / "spectra.npz")
            c = set(T.committee_from_coeffs(z["coeffs"][-1]))
            j = len(c & cb) / max(1, len(c | cb))
            rows.append(dict(seed=s, eps=e, rep=r, base=sorted(cb), pert=sorted(c),
                             jaccard=j, exact=c == cb))
            print(f"seed{s} eps{e:g} r{r}: base {sorted(cb)} pert {sorted(c)} J={j:.2f}")
summ = {}
for e in a.eps:
    if e == 0: continue
    rr = [x for x in rows if x["eps"] == e]
    if rr:
        summ[f"{e:g}"] = dict(n=len(rr), exact_rate=float(np.mean([x["exact"] for x in rr])),
                             mean_jaccard=float(np.mean([x["jaccard"] for x in rr])))
print(json.dumps(summ, indent=1))
Path("notes").mkdir(exist_ok=True)
Path(f"notes/chaos_summary.json").write_text(json.dumps(dict(rows=rows, summary=summ), indent=1))
