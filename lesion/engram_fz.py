"""Order-flip test for engram formation: dense-checkpoint donors trained
with W_E FROZEN for the first 500 epochs. Natural order (engram_curves.py)
is W_E writes first (0-60), W_U replicates after (60-100). Prediction if
"first scribe" is contingent on trainability: under the freeze, W_U's
record forms FIRST (0-100) while frozen W_E's AUC stays flat at its init
value, rising only after epoch 500.

Run on pod: python3 lesion/engram_fz.py
Output: notes/engram_fz_curves.json
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_engram_fz"))
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "dial"))

import numpy as np  # noqa: E402
from train_semifinal_torch import RUNS, Config, train_batched  # noqa: E402
from train_lesion_seeds import draw_seeds  # noqa: E402


def main():
    data_seeds, init_seeds = draw_seeds()
    jobs = []
    for ds in data_seeds:
        for is_ in init_seeds:
            rd = RUNS / f"ds{ds}_is{is_}"
            if not (rd / "spectra.npz").exists():
                jobs.append((Config(d_mlp=512, weight_decay=3.0, data_seed=ds,
                                    init_seed=is_, num_epochs=2500,
                                    save_every=10), rd))
    if jobs:
        print(f"fz donors: {len(jobs)}", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                      freeze=[("freeze_we", 500)] * len(jobs),
                      spectra_every=10)

    from analyze_dial import (BASIS, NF, P, committee_from_coeffs,
                              freq_energy, mw_auc)
    from safetensors.numpy import load_file
    curves = {}
    for rd in sorted(RUNS.glob("ds*")):
        z = np.load(rd / "spectra.npz")
        comm = committee_from_coeffs(z["coeffs"][-1])
        m = np.zeros(NF, bool)
        m[[k - 1 for k in comm]] = True
        for ck in sorted((rd / "checkpoints").glob("epoch_*.safetensors")):
            ep = int(ck.stem.split("_")[1])
            s = load_file(str(ck))
            we = mw_auc(freq_energy(
                BASIS @ s["embed.W_E"].astype(np.float64)[:, :P].T), m)
            wu = mw_auc(freq_energy(
                BASIS @ s["unembed.W_U"].astype(np.float64)[:, :P].T), m)
            curves.setdefault(ep, []).append((float(we), float(wu)))

    out = {str(ep): {"auc_we": float(np.mean([a for a, _ in v])),
                     "auc_wu": float(np.mean([b for _, b in v])),
                     "n": len(v)}
           for ep, v in sorted(curves.items())}
    o = ROOT / "notes" / "engram_fz_curves.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
    for ep in [0, 30, 60, 100, 200, 400, 500, 550, 600, 700, 1000, 2000]:
        if str(ep) in out:
            r = out[str(ep)]
            print(f"{ep:>5} WE {r['auc_we']:.3f}  WU {r['auc_wu']:.3f}")
    print(f"wrote {o}", flush=True)


if __name__ == "__main__":
    main()
