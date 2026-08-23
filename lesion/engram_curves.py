"""Dense engram-formation curves: retrain 8 mod-add donors (wd 3) with
checkpoints every 10 epochs, then trace committee AUC of W_E and W_U per
checkpoint. Exploratory (mechanism section support): does one copy form
first, or do the two grow in lockstep (balanced bilinear growth)?

Run on pod: python3 lesion/engram_curves.py   (trains if needed, then analyzes)
Output: notes/engram_curves.json
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_engram"))
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
                                    init_seed=is_, num_epochs=1500,
                                    save_every=10), rd))
    if jobs:
        print(f"engram donors: {len(jobs)}", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
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
    o = ROOT / "notes" / "engram_curves.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
    print(f"{'ep':>5} {'WE':>6} {'WU':>6}")
    for ep in sorted(curves, key=int):
        r = out[str(ep)]
        print(f"{ep:>5} {r['auc_we']:>6.3f} {r['auc_wu']:>6.3f}")
    print(f"wrote {o}", flush=True)


if __name__ == "__main__":
    main()
