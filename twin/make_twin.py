"""twin/make_twin.py — manifest generation for the twin suite (two stages).

Stage 1 is the BASES: 8 (data_seed, init_seed) pairs trained at frac 0.3
(BASES-A, 30k epochs) and the SAME 8 pairs at frac 0.5 (BASES-B, 15k), with
checkpoint ladders.  Stage 1 needs nothing but seeds.

Stage 2 is every twin cohort, and it CANNOT be written before stage 1 has run:
FORK needs the checkpoint ladder, SWAP and DIAL need each base's final
committee (which frequencies won) and its init W_E per-frequency energy ranking
(which frequencies were loud at t=0), HOLD needs the BASES-B ladder.

Every seed is DRAWN from one META_SEED stream and recorded in the manifest, so
the cohort is reproducible and carries no author preference.  Init/data seeds
come from [40000, 50000), a range no previous cohort in this repo has spent.

Group discipline: every job carries a `group`, and train_twin never splits a
group across vmap batches.  Each group contains its own unedited / eps=0
control, so every comparison the analysis makes is within one batch — the
fallback rule for gate G3.  This is why SWAP and DIAL each carry one extra
`ctrl` arm per base beyond the nominal arm counts.

Usage:
    uv run --group convert python twin/make_twin.py --stage 1 --out twin/manifests
    uv run --group convert python twin/make_twin.py --stage 2 --out twin/manifests --runs runs_twin
"""
import argparse
import random
from pathlib import Path

import numpy as np

from core import (Recipe, committee_from_coeffs, load_ckpt, we_freq_energy,
                  write_json)

META_SEED = 20260816
SEED_RANGE = (40_000, 50_000)      # disjoint from every seed range spent so far


def default_spec():
    return {
        "meta_seed": META_SEED,
        "recipe": Recipe(),
        "n_data_seeds": 2,
        "n_init_seeds": 4,
        "log_every": 10,
        "spectra_every": 25,
        "basesA": {"frac": 0.3, "epochs": 30_000,
                   "ckpts": [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]},
        "basesB": {"frac": 0.5, "epochs": 15_000,
                   "ckpts": [10, 25, 60, 150, 400]},
        "fork": {"eps": [0.01, 0.03, 0.1], "K": 3, "epochs": 30_000},
        "swap": {"n_exchange": 4, "n_sham": 4, "n_pos": 2, "pos_scale": 0.5,
                 "epochs": 30_000},
        "dial": {"doses": [0.25, 0.5, 0.8, 1.25, 2.0, 4.0],
                 "n_member": 2, "n_runner": 2, "n_mid": 2, "epochs": 30_000},
        "hold": {"levels": [0.35, 0.4, 0.7, 0.9], "epochs": 30_000},
        # SWAP/DIAL targets are read off the base's FINAL committee; a base
        # that never grokked has a memorization-phase spectrum and its targets
        # are garbage.  Stage 2 refuses to generate rather than burn 30k-epoch
        # arms on it.  (The smoke sets this to 0.0 — its minis never grok.)
        "min_base_acc": 0.99,
    }


def draw_seeds(spec):
    """All base seeds from one stream.  Same META_SEED -> same cohort."""
    rng = random.Random(spec["meta_seed"])
    lo, hi = SEED_RANGE
    pool = rng.sample(range(lo, hi), spec["n_data_seeds"] + spec["n_init_seeds"])
    data_seeds = sorted(pool[:spec["n_data_seeds"]])
    init_seeds = sorted(pool[spec["n_data_seeds"]:])
    return data_seeds, init_seeds


def base_pairs(spec):
    ds, is_ = draw_seeds(spec)
    return [(d, i) for d in ds for i in is_]


def bname(ds, iseed):
    return f"ds{ds}_is{iseed}"


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------

def stage1_manifest(spec):
    jobs = []
    for tag in ("basesA", "basesB"):
        s = spec[tag]
        for ds, iseed in base_pairs(spec):
            jobs.append({
                "name": f"{tag}/{bname(ds, iseed)}", "cohort": tag,
                "group": tag, "data_seed": ds, "init_seed": iseed,
                "frac": s["frac"], "start_epoch": 0, "end_epoch": s["epochs"],
                "log_every": spec["log_every"], "spectra_every": spec["spectra_every"],
                "ckpt_epochs": sorted(set(s["ckpts"]) | {s["epochs"]}),
                "init": {"kind": "fresh"}, "edit": {"kind": "none"},
            })
    return {"stage": 1, "meta_seed": spec["meta_seed"],
            "recipe": spec["recipe"].to_json(), "jobs": jobs}


# --------------------------------------------------------------------------
# stage 2 — needs stage 1 results
# --------------------------------------------------------------------------

def base_readout(runs_dir, name, p, min_acc=0.99):
    """(final committee, members ranked by |final coeff|, non-members ranked by
    init W_E per-frequency energy) for one BASES-A run."""
    rd = Path(runs_dir) / name
    z = np.load(rd / "spectra.npz")
    acc = float(z["test_acc"][-1])
    assert acc >= min_acc, (
        f"{name}: base never grokked (final test acc {acc:.3f} < {min_acc}) — "
        f"its final committee is a memorization spectrum and SWAP/DIAL targets "
        f"derived from it are undefined. Extend the base budget or drop it.")
    coeffs = z["coeffs"][-1]
    comm = committee_from_coeffs(coeffs)
    ck = load_ckpt(rd / "checkpoints" / "epoch_000000.pt")
    E = we_freq_energy(ck["weights"]["embed.W_E"].numpy(), p)
    nf = p // 2
    members = sorted(comm, key=lambda k: -abs(coeffs[k - 1]))
    nonmem = sorted((k for k in range(1, nf + 1) if k not in comm),
                    key=lambda k: -E[k - 1])
    return {"committee": comm, "members": members, "nonmembers": nonmem,
            "final_test_acc": float(z["test_acc"][-1]),
            "we_energy": E.tolist()}


def fork_jobs(spec, pairs, rng):
    s, A = spec["fork"], spec["basesA"]
    jobs = []
    for ds, iseed in pairs:
        b = bname(ds, iseed)
        base = f"basesA/{b}"
        for T in A["ckpts"]:
            grp = f"fork/{b}/T{T:06d}"
            common = dict(cohort="fork", group=grp, data_seed=ds, init_seed=iseed,
                          frac=A["frac"], start_epoch=T, end_epoch=s["epochs"],
                          log_every=spec["log_every"],
                          spectra_every=spec["spectra_every"], ckpt_epochs=[],
                          init={"kind": "ckpt",
                                "path": f"{base}/checkpoints/epoch_{T:06d}.pt"},
                          base=base, T=T)
            jobs.append({"name": f"{grp}/eps0_ctrl",
                         "edit": {"kind": "none"}, "arm": "ctrl", "eps": 0.0,
                         "draw": -1, **common})
            for eps in s["eps"]:
                for k in range(s["K"]):
                    jobs.append({
                        "name": f"{grp}/eps{eps:g}_k{k}",
                        "edit": {"kind": "perturb", "eps": float(eps),
                                 "seed": rng.randrange(*SEED_RANGE)},
                        "arm": "perturb", "eps": float(eps), "draw": k, **common})
    return jobs


def swap_jobs(spec, pairs, readouts):
    s, A = spec["swap"], spec["basesA"]
    jobs = []
    for ds, iseed in pairs:
        b = bname(ds, iseed)
        base = f"basesA/{b}"
        r = readouts[b]
        grp = f"swap/{b}"
        common = dict(cohort="swap", group=grp, data_seed=ds, init_seed=iseed,
                      frac=A["frac"], start_epoch=0, end_epoch=s["epochs"],
                      log_every=spec["log_every"],
                      spectra_every=spec["spectra_every"], ckpt_epochs=[],
                      init={"kind": "ckpt",
                            "path": f"{base}/checkpoints/epoch_000000.pt"},
                      base=base, T=0)
        mem, non = r["members"], r["nonmembers"]
        # unedited twin, trained in the SAME batch as every edited arm
        jobs.append({"name": f"{grp}/ctrl", "edit": {"kind": "none"},
                     "arm": "ctrl", **common})
        # Group size must be CONSTANT across bases (G3: cohort chunks must all
        # have one width, and mixed group sizes trigger the greedy packer).  A
        # base with a small committee therefore pads its exchange/pos shortfall
        # with extra sham arms: 1 + n_ex + n_sham + n_pos is invariant.
        n_ex = min(s["n_exchange"], len(mem), len(non))
        n_pos = min(s["n_pos"], len(mem))
        n_sham = s["n_sham"] + (s["n_exchange"] - n_ex) + (s["n_pos"] - n_pos)
        # winner <-> runner: swap a committee member's init drive with the
        # loudest non-member's.  If selection is about WHICH frequency holds the
        # drive, the committee follows the exchange.
        for t in range(n_ex):
            i, j = mem[t], non[t]
            jobs.append({"name": f"{grp}/ex{t}_f{i}x{j}",
                         "edit": {"kind": "we_exchange", "pairs": [[i, j]]},
                         "arm": "exchange", "rank": t, "freqs": [i, j], **common})
        # sham: exchange two mid-rank non-members with each other.  Same KIND of
        # surgery, same conservation laws, no committee frequency involved.
        mid = len(non) // 2
        pool = non[max(0, mid - n_sham): max(0, mid - n_sham) + 2 * n_sham]
        for t in range(min(n_sham, len(pool) // 2)):
            i, j = pool[2 * t], pool[2 * t + 1]
            jobs.append({"name": f"{grp}/sham{t}_f{i}x{j}",
                         "edit": {"kind": "we_exchange", "pairs": [[i, j]]},
                         "arm": "sham", "rank": t, "freqs": [i, j], **common})
        # positive control: a NON-conservative edit (halve a winner's drive).
        # This one is expected to move things; it calibrates the null arms.
        for t in range(n_pos):
            i = mem[t]
            jobs.append({"name": f"{grp}/pos{t}_f{i}x{s['pos_scale']:g}",
                         "edit": {"kind": "we_scale",
                                  "scales": {str(i): s["pos_scale"]}},
                         "arm": "pos", "rank": t, "freqs": [i], **common})
    return jobs


def dial_jobs(spec, pairs, readouts):
    s, A = spec["dial"], spec["basesA"]
    jobs = []
    for ds, iseed in pairs:
        b = bname(ds, iseed)
        base = f"basesA/{b}"
        r = readouts[b]
        grp = f"dial/{b}"
        common = dict(cohort="dial", group=grp, data_seed=ds, init_seed=iseed,
                      frac=A["frac"], start_epoch=0, end_epoch=s["epochs"],
                      log_every=spec["log_every"],
                      spectra_every=spec["spectra_every"], ckpt_epochs=[],
                      init={"kind": "ckpt",
                            "path": f"{base}/checkpoints/epoch_000000.pt"},
                      base=base, T=0)
        jobs.append({"name": f"{grp}/ctrl", "edit": {"kind": "none"},
                     "arm": "ctrl", **common})
        mem, non = r["members"], r["nonmembers"]
        mid = len(non) // 2
        # constant group size across bases (see swap_jobs): a member shortfall
        # is backfilled with extra runner targets
        n_mem = min(s["n_member"], len(mem))
        n_run = s["n_runner"] + (s["n_member"] - n_mem)
        targets = ([(k, "member") for k in mem[:n_mem]]
                   + [(k, "runner") for k in non[:n_run]]
                   + [(k, "mid") for k in non[mid:mid + s["n_mid"]]])
        for k, cls in targets:
            for dose in s["doses"]:
                jobs.append({
                    "name": f"{grp}/f{k}_{cls}/dose{dose:g}",
                    "edit": {"kind": "we_scale", "scales": {str(k): float(dose)}},
                    "arm": "dial", "target": k, "target_class": cls,
                    "dose": float(dose), **common})
    return jobs


def hold_jobs(spec, pairs):
    s, B = spec["hold"], spec["basesB"]
    jobs = []
    for ds, iseed in pairs:
        b = bname(ds, iseed)
        base = f"basesB/{b}"
        for T in B["ckpts"]:
            grp = f"hold/{b}/T{T:06d}"
            common = dict(cohort="hold", group=grp, data_seed=ds, init_seed=iseed,
                          start_epoch=T, end_epoch=s["epochs"],
                          log_every=spec["log_every"],
                          spectra_every=spec["spectra_every"], ckpt_epochs=[],
                          init={"kind": "ckpt",
                                "path": f"{base}/checkpoints/epoch_{T:06d}.pt"},
                          base=base, T=T)
            # same-data control: identical mask, no edit -> must reproduce the
            # base's logs bitwise up to the base's own horizon
            jobs.append({"name": f"{grp}/same", "edit": {"kind": "none"},
                         "arm": "ctrl", "frac": B["frac"],
                         "level": B["frac"], **common})
            for lv in s["levels"]:
                kind = "conceal" if lv < B["frac"] else "reveal"
                jobs.append({"name": f"{grp}/{kind}{lv:g}",
                             "edit": {"kind": "none"}, "arm": kind,
                             "frac": float(lv), "level": float(lv), **common})
    return jobs


def stage2_manifests(spec, runs_dir):
    pairs = base_pairs(spec)
    p = spec["recipe"].p
    readouts = {bname(ds, i): base_readout(runs_dir, f"basesA/{bname(ds, i)}", p,
                                           spec.get("min_base_acc", 0.99))
                for ds, i in pairs}
    rng = random.Random(spec["meta_seed"] + 1)
    out = {
        "fork": fork_jobs(spec, pairs, rng),
        "swap": swap_jobs(spec, pairs, readouts),
        "dial": dial_jobs(spec, pairs, readouts),
        "hold": hold_jobs(spec, pairs),
    }
    return out, readouts


def wrap(spec, jobs, stage=2):
    return {"stage": stage, "meta_seed": spec["meta_seed"],
            "recipe": spec["recipe"].to_json(), "jobs": jobs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--out", default="twin/manifests")
    ap.add_argument("--runs", default="runs_twin")
    args = ap.parse_args()

    spec = default_spec()
    out = Path(args.out)
    if args.stage == 1:
        mf = stage1_manifest(spec)
        write_json(out / "stage1.json", mf)
        ds, iss = draw_seeds(spec)
        print(f"META_SEED {spec['meta_seed']}  data_seeds {ds}  init_seeds {iss}")
        print(f"wrote {out / 'stage1.json'}: {len(mf['jobs'])} runs")
    else:
        cohorts, readouts = stage2_manifests(spec, args.runs)
        write_json(out / "base_readouts.json", readouts)
        total = 0
        for k, jobs in cohorts.items():
            write_json(out / f"{k}.json", wrap(spec, jobs))
            print(f"wrote {out / (k + '.json')}: {len(jobs)} runs")
            total += len(jobs)
        for b, r in sorted(readouts.items()):
            print(f"  {b}: committee {r['committee']}  "
                  f"top nonmembers {r['nonmembers'][:4]}  "
                  f"acc {r['final_test_acc']:.4f}")
        print(f"stage 2 total: {total} runs")


if __name__ == "__main__":
    main()
