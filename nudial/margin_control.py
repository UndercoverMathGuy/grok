"""Controls for the margin redo's "0 accepted polish steps" reading.

(a) f64 polish on 8 AdamW relu4 f0.5 finals (was float32): if the ascent
    still accepts nothing, float32 was not the blocker.
(b) positive control: perturb the same 8 finals with Gaussian noise at a
    relative scale (1% and 3% of each tensor's norm), which lowers gamma,
    then polish (f32, same settings). If the polish recovers most of the
    lost margin, "0 accepted" on the unperturbed nets means stationary, not
    a broken ascent.
Usage: python3 nudial/margin_control.py  -> notes/margin_control.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
import margin_redo as mr

runs = sorted((ROOT / "runs_nudial" / "M" / "relu4_m512_p113_f0.5_train").glob("wd1_s*"))[:8]
out = {}
# (a) f64
mr.DT = torch.float64
cfgs, pr = mr.load_group(runs)
g0, g1, keff, Kg, nacc = mr.polish(pr, cfgs[0], 4, 100, label="f64 AdamW relu4")
out["f64"] = dict(gamma_before=g0.tolist(), gamma_after=g1.tolist(), accepted=nacc.tolist())
print(f"f64: gamma med {np.median(g0):.4e} -> {np.median(g1):.4e}, accepted steps {nacc.tolist()}", flush=True)
# (b) noise positive control, f32
mr.DT = torch.float32
for rel in (0.1, 0.03):
    cfgs, pr = mr.load_group(runs)
    gen = torch.Generator(device=mr.DEV).manual_seed(0)
    for k in pr:
        nrm = pr[k].flatten(1).norm(dim=1).view(-1, *[1] * (pr[k].dim() - 1))
        noise = torch.randn(pr[k].shape, generator=gen, device=mr.DEV, dtype=mr.DT)
        noise = noise / noise.flatten(1).norm(dim=1).view(-1, *[1] * (pr[k].dim() - 1)) * nrm * rel
        pr[k] = pr[k] + noise
    g0, g1, keff, Kg, nacc = mr.polish(pr, cfgs[0], 4, 150, label=f"noise {rel}")
    orig = np.array(out["f64"]["gamma_before"])
    out[f"noise_{rel}"] = dict(gamma_perturbed=g0.tolist(), gamma_polished=g1.tolist(), gamma_original=orig.tolist(),
                              accepted=nacc.tolist(), K_gap=[int(k) for k in Kg])
    print(f"noise {rel}: original med {np.median(orig):.4e} -> perturbed {np.median(g0):.4e} -> polished {np.median(g1):.4e} "
          f"(recovered {np.median((g1 - g0) / (orig - g0)) * 100:.0f}% of the loss); accepted {nacc.tolist()}; K {Kg}", flush=True)
(ROOT / "notes" / "margin_control.json").write_text(json.dumps(out, indent=1))
print("wrote notes/margin_control.json")
