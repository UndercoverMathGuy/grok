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


def make_crossed(args):
    """Phase C: the S-side content dissociation, built to be testable.

    Three defects of Phase B are fixed by construction:
      1. ONE shared content pool, drawn once, installed at every dose on
         every base. Phase B redrew inside the innermost loop, so all 96
         target sets were distinct and the 'identity' factor was an index
         label carrying no content -- the within-dose permutation was
         vacuous. Here content is a genuine crossed factor.
      2. Natural substrate: the base's own W_E draw is kept, so the init is
         a real natural init with a tilt added, not a QR-flattened field.
      3. Every base with an epoch-0 checkpoint (2 data masks x 4 init
         seeds), not 2 bases off a single mask.
    One no-install control per base anchors the natural t_gen of the cohort.
    """
    bases = natural_bases(args.p)[:args.bases]
    assert len(bases) == args.bases, f"only {len(bases)} bases available"
    rng = np.random.default_rng(args.rng_seed)
    (ARMS / "ckpts_c").mkdir(parents=True, exist_ok=True)
    safeties = [float(x) for x in args.crossed.split(",")]

    pool = []
    while len(pool) < args.n_contents:
        S, feas = draw_feasible_set(rng, args.p, args.k, args.floor_pct)
        if S not in [c[0] for c in pool]:
            pool.append((S, feas))
    print(f"shared content pool (K={args.k}, drawn once, crossed with "
          f"dose x base):")
    for i, (S, _) in enumerate(pool):
        print(f"  content{i}: {S}")

    def rec(tag, bdir, kind, ckpt, run_sub, epochs, cell, **kw):
        """Paths are stored REPO-RELATIVE: the cloud pod resolves them against
        its own clone root, so nothing depends on this machine's layout."""
        cfg = json.loads((bdir / "config.json").read_text()) | dict(
            num_epochs=epochs, save_every=epochs)
        cfg.update(kw.pop("config_over", {}))
        return dict(tag=tag, kind=kind, base=str(bdir.relative_to(RUNS)),
                    ckpt_rel=str(Path(ckpt).relative_to(ROOT)),
                    run_rel=str(Path("runs_compiler/phaseC") / run_sub),
                    run_dir=str(OUT_RUNS / "phaseC" / run_sub),
                    ckpt=str(ckpt), cell=cell, config=cfg, **kw)

    manifest = []
    # ---- S-side: content x dose x base, fully crossed -------------------
    for bdir, bcfg, e0 in bases:
        bname = bdir.name
        params = load_ckpt(e0)
        for ci, (S, feas) in enumerate(pool):
            for s in safeties:
                tag = f"s{s:g}_c{ci}_" + "-".join(map(str, S))
                compiled, rep = compile_init(params, args.p, S,
                                             substrate=args.substrate,
                                             route="energy", safety=s,
                                             renorm="global")
                assert rep["spec_met"], (tag, rep)
                ck = ARMS / "ckpts_c" / f"{bname}_{tag}.safetensors"
                save_ckpt(ck, compiled)
                manifest.append(rec(
                    f"{bname}_{tag}", bdir, "sside", ck, f"{bname}/{tag}",
                    args.epochs, dict(s=s, k=args.k, content=ci, base=bname),
                    targets=S, feasibility=feas, report=rep,
                    predicted_committee=S))
        # no-install control: the base's own epoch-0 checkpoint, untouched.
        # Anchors this cohort's natural t_gen, so the acceleration the
        # install buys is measured inside the cohort instead of across it.
        manifest.append(rec(
            f"{bname}_control", bdir, "sside_control", e0, f"{bname}/control",
            args.epochs, dict(s=0.0, k=None, content=None, base=bname),
            targets=[], feasibility=None, predicted_committee=None,
            report=dict(substrate="none", route=None, safety=None)))
        print(f"  {bname}: {len(pool) * len(safeties)} S-side arms + control",
              flush=True)

    # ---- L-side: late data reveal, post-lock ----------------------------
    # Resumes from the base's OWN natural checkpoints (save_every=1000), so
    # the donors are genuine mainline natural runs at wd=1.0 -- not the
    # wd=3.0 boost donors, 2 of whose 8 never reached the threshold.
    # frac 0.3 -> 0.7 with the same data_seed: the split is a seeded shuffle
    # truncated at frac*N, so the larger training set is an exact SUPERSET.
    # No example changes side, no parameter or optimizer setting is touched.
    jumps = [int(x) for x in args.jump_times.split(",")]
    for bdir, bcfg, e0 in bases:
        bname = bdir.name
        for t in jumps:
            ck = bdir / "checkpoints" / f"epoch_{t:05d}.safetensors"
            assert ck.exists(), f"missing donor checkpoint {ck}"
            for kind, fr in (("lside_jump", args.frac2),
                             ("lside_gate", bcfg.frac_train)):
                sub = f"{bname}/{kind.split('_')[1]}{t}"
                manifest.append(rec(
                    f"{bname}_{kind.split('_')[1]}{t}", bdir, kind, ck, sub,
                    args.lside_epochs,
                    dict(s=None, k=None, content=None, base=bname, t_jump=t),
                    targets=[], feasibility=None, predicted_committee=None,
                    report=dict(substrate="none", route=None, safety=None),
                    config_over=dict(frac_train=fr)))
        print(f"  {bname}: {2 * len(jumps)} L-side arms", flush=True)

    out = ARMS / "phaseC_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    ns = sum(a["kind"].startswith("sside") for a in manifest)
    nl = sum(a["kind"].startswith("lside") for a in manifest)
    print(f"\n{len(manifest)} arms -> {out}   (S-side {ns}, L-side {nl})")
    print(f"S-side: {args.bases} bases x {len(pool)} contents x "
          f"{len(safeties)} doses (fully crossed) + {args.bases} controls")
    print(f"L-side: {args.bases} bases x {len(jumps)} jump times "
          f"x (jump frac {args.frac2} + self-restart gate)")
    print("PRE-REGISTERED analyses, both committed before training: "
          "compiler/analyze_sside.py and compiler/analyze_lside.py.")


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
    ap.add_argument("--crossed", default=None,
                    help="comma list of safeties -> Phase C crossed mode")
    ap.add_argument("--substrate", default="natural",
                    choices=("natural", "flat"))
    ap.add_argument("--n-contents", type=int, default=8,
                    help="size of the shared content pool (Phase C)")
    ap.add_argument("--jump-times", default="500,1000,2000,4000,6000,8000,10000",
                    help="Phase C L-side: donor epochs to reveal data at")
    ap.add_argument("--frac2", type=float, default=0.7,
                    help="Phase C L-side: revealed training fraction")
    ap.add_argument("--lside-epochs", type=int, default=20_000)
    args = ap.parse_args()
    if args.crossed:
        make_crossed(args)
        return
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
