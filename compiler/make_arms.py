"""Phase A arm generation: compile target committees into base inits, write
the edited epoch-0 checkpoints, and PRE-REGISTER the predicted committees
before any training.

Per base (natural p-113 runs with an epoch-0 checkpoint):
  - N_SETS strictly-feasible K=4 target sets (no additive relations incl.
    harmonics; LP margin percentile >= 40), drawn from a seeded RNG
  - reliable mode: flat substrate, energy route, safety s (default 3.0)
  - 1 control: flat substrate, NO targets (the re-rolled lottery baseline;
    prediction: committee != any arm's target set; weak prediction: members
    come from the substrate's own T_k leaders, recorded here)

Outputs:
  compiler/arms/ckpts/<tag>.safetensors      compiled epoch-0 checkpoints
  compiler/arms/phaseA_manifest.json         one record per arm, with the
                                             pre-registered prediction
Run BEFORE training; train_arms.py consumes the manifest.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler"))

from grok.config import Config
from core import (compile_init, feasibility, load_ckpt, save_ckpt,
                  tk_profile, violations)

RUNS = ROOT / "runs_torch"
ARMS = ROOT / "compiler" / "arms"
OUT_RUNS = ROOT / "runs_compiler"

NATURAL_FAMS = {"p-113", "seed0", "seed1", "seed2", "og_seed0"}


def natural_bases(p=113):
    """Natural runs of prime p with an epoch-0 checkpoint."""
    out = []
    for cj in sorted(RUNS.rglob("config.json")):
        d = cj.parent
        fam = str(d.relative_to(RUNS)).split("/")[0]
        if fam not in NATURAL_FAMS:
            continue
        cfg = Config.load(cj)
        if cfg.p != p or getattr(cfg, "embed_init", "normal") != "normal":
            continue
        e0 = d / "checkpoints" / "epoch_00000.safetensors"
        if e0.exists():
            out.append((d, cfg, e0))
    return out


def draw_feasible_set(rng, p, k, floor_pct):
    nf = p // 2
    for _ in range(10_000):
        S = sorted(rng.choice(np.arange(1, nf + 1), size=k, replace=False).tolist())
        if violations(S, p):
            continue
        f = feasibility(S, p, floor_pct=floor_pct)
        if f["ok"]:
            return S, f
    raise RuntimeError("no feasible set found")


def make_sweep(args):
    """Phase B: dose x K sweep, global renorm (achieved margin == s exactly),
    no probes, no controls — every arm is a weights-only compiled program.
    Pre-registration: predicted committee == targets for every arm; expected
    exact-match rate is non-decreasing in s within each K."""
    bases = natural_bases(args.p)[:args.bases]
    rng = np.random.default_rng(args.rng_seed)
    (ARMS / "ckpts_b").mkdir(parents=True, exist_ok=True)
    safeties = [float(x) for x in args.sweep.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    manifest = []
    for bdir, bcfg, e0 in bases:
        bname = bdir.name
        params = load_ckpt(e0)
        for k in ks:
            for s in safeties:
                for i in range(args.sets_per_cell):
                    S, feas = draw_feasible_set(rng, args.p, k, args.floor_pct)
                    tag = f"s{s:g}_K{k}_set{i}_" + "-".join(map(str, S))
                    compiled, rep = compile_init(params, args.p, S,
                                                 substrate="flat",
                                                 route="energy", safety=s,
                                                 renorm="global")
                    assert rep["spec_met"], rep
                    ck = ARMS / "ckpts_b" / f"{bname}_{tag}.safetensors"
                    save_ckpt(ck, compiled)
                    manifest.append(dict(
                        tag=f"{bname}_{tag}", base=str(bdir.relative_to(RUNS)),
                        run_dir=str(OUT_RUNS / "phaseB" / bname / tag),
                        ckpt=str(ck), targets=S, feasibility=feas, report=rep,
                        predicted_committee=S, cell=dict(s=s, k=k),
                        config=json.loads((bdir / "config.json").read_text())
                        | dict(num_epochs=args.epochs,
                               save_every=args.epochs),
                    ))
                print(f"  {bname} K={k} s={s:g}: {args.sets_per_cell} arms",
                      flush=True)
    out = ARMS / "phaseB_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} arms -> {out}")
    print("PRE-REGISTERED: predicted committee == targets for every arm; "
          "exact-rate non-decreasing in s within each K.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--bases", type=int, default=2, help="number of base inits")
    ap.add_argument("--sets-per-base", type=int, default=3)
    ap.add_argument("--k", type=int, default=4, help="target committee size")
    ap.add_argument("--safety", type=float, default=3.0)
    ap.add_argument("--floor-pct", type=float, default=40.0)
    ap.add_argument("--epochs", type=int, default=50_000)
    ap.add_argument("--rng-seed", type=int, default=20260812)
    ap.add_argument("--sweep", default=None,
                    help="comma list of safeties -> Phase B sweep mode")
    ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--sets-per-cell", type=int, default=4)
    args = ap.parse_args()
    if args.sweep:
        make_sweep(args)
        return

    bases = natural_bases(args.p)
    assert len(bases) >= args.bases, f"only {len(bases)} natural bases with e0"
    bases = bases[:args.bases]
    rng = np.random.default_rng(args.rng_seed)
    (ARMS / "ckpts").mkdir(parents=True, exist_ok=True)

    manifest = []
    for bdir, bcfg, e0 in bases:
        bname = bdir.name
        params = load_ckpt(e0)
        print(f"== base {bdir.relative_to(RUNS)} (data_seed {bcfg.data_seed})")

        arms = []
        for i in range(args.sets_per_base):
            S, feas = draw_feasible_set(rng, args.p, args.k, args.floor_pct)
            arms.append((f"set{i}_" + "-".join(map(str, S)), S, feas))
        arms.append(("control", None, None))

        for tag, S, feas in arms:
            full_tag = f"{bname}_{tag}"
            if S is not None:
                compiled, rep = compile_init(params, args.p, S,
                                             substrate="flat", route="energy",
                                             safety=args.safety)
                assert rep["spec_met"], rep
                predicted = S
            else:
                from core import flatten_WE
                compiled = flatten_WE(params, args.p)
                tk = tk_profile(compiled, args.p)
                rep = dict(substrate="flat", route=None, safety=None,
                           tk_top12=[int(x) for x in np.argsort(tk)[::-1][:12] + 1],
                           margin=None, spec_met=None)
                predicted = None       # control: no dictated committee
            ck = ARMS / "ckpts" / f"{full_tag}.safetensors"
            save_ckpt(ck, compiled)
            cfg = Config.load(bdir / "config.json")
            cfg.num_epochs = args.epochs
            cfg.save_every = 1000
            manifest.append(dict(
                tag=full_tag, base=str(bdir.relative_to(RUNS)),
                run_dir=str(OUT_RUNS / "phaseA" / bname / tag),
                ckpt=str(ck), targets=S, feasibility=feas, report=rep,
                predicted_committee=predicted,
                config=json.loads((bdir / "config.json").read_text())
                | dict(num_epochs=args.epochs, save_every=1000),
            ))
            m = rep.get("margin")
            print(f"   {tag:<24} targets {S}  margin "
                  f"{m:.2f}" if m else f"   {tag:<24} (control)  "
                  f"tk_top {rep['tk_top12'][:6]}")

    out = ARMS / "phaseA_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} arms -> {out}")
    print("PRE-REGISTERED: predicted committee == targets for every compiled "
          "arm (exact set, unified detector), controls excluded.")


if __name__ == "__main__":
    main()
