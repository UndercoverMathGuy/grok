"""twin/smoke_twin.py — CPU smoke test for the whole twin suite.

Tiny override of the recipe (p=23, d_model 32, 2 heads, d_mlp 64) and ~600
epoch budgets: the smoke does NOT need anything to grok, it needs the
MECHANISM to be right.  Five things are checked:

  1. gates       G1 / G2(solo) / G2(batched) / G3 / within-batch.  G3 may fail;
                 if it does the fallback rule is engaged and reported.
  2. certificates SWAP exchange and DIAL scale edits on a real tiny init, plus
                 the perturbation norm match.
  3. masks       reveal/conceal masks are prefixes of ONE permutation and never
                 touch the test set, which is identical across fractions.
  4. end-to-end  stage 1 -> stage 2 manifest generation -> train every cohort
                 (fork incl. eps=0, swap, dial, hold with a mask change) ->
                 analyze, with the eps=0 fork reproducing its base's logs.
  5. analysis    all four summaries produced and structurally sane.

Usage:
    uv run --group convert python twin/smoke_twin.py
"""
import argparse
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import make_twin
from analyze_twin import load_all, analyze_bases, analyze_fork, analyze_swap, \
    analyze_dial, analyze_hold
from core import (Recipe, apply_edit, committee_from_coeffs, init_params,
                  perturb_params, setup_determinism, split_masks, we_exchange,
                  we_freq_energy, we_scale, write_json)
from gates import run_gates
from make_twin import stage1_manifest, stage2_manifests, wrap
from train_twin import run_manifest

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
    except Exception as e:      # noqa: BLE001 — a smoke failure must be visible
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        traceback.print_exc()
    return RESULTS[-1][1]


def smoke_spec():
    s = make_twin.default_spec()
    s["recipe"] = Recipe(p=23, d_model=32, num_heads=2, d_mlp=64)
    s["n_data_seeds"], s["n_init_seeds"] = 1, 2
    s["log_every"], s["spectra_every"] = 10, 25
    s["basesA"] = {"frac": 0.5, "epochs": 600, "ckpts": [0, 10, 40, 120]}
    s["basesB"] = {"frac": 0.5, "epochs": 300, "ckpts": [10, 40]}
    s["fork"] = {"eps": [0.03], "K": 2, "epochs": 600}
    s["swap"] = {"n_exchange": 1, "n_sham": 1, "n_pos": 1, "pos_scale": 0.5,
                 "epochs": 600}
    s["dial"] = {"doses": [0.5, 2.0], "n_member": 1, "n_runner": 1, "n_mid": 1,
                 "epochs": 600}
    s["hold"] = {"levels": [0.4, 0.7], "epochs": 600}
    s["min_base_acc"] = 0.0        # smoke minis never grok; mechanism only
    return s


# --------------------------------------------------------------------------

def check_masks(p):
    fracs = [0.35, 0.4, 0.5, 0.7, 0.9]
    for ds in (41111, 41222):
        perm, _, te0 = split_masks(p, ds, 0.5)
        prev = None
        for f in fracs:
            _, tr, te = split_masks(p, ds, f)
            assert torch.equal(te, te0), "test set moved with the fraction"
            assert not bool((tr & te).any()), "train mask touched the test set"
            idx = torch.nonzero(tr).flatten()
            assert torch.equal(torch.sort(idx).values,
                               torch.sort(perm[:int(f * p * p)]).values), \
                "train mask is not the permutation prefix"
            if prev is not None:
                assert bool((prev <= tr).all()), "masks are not nested"
            prev = tr
    return f"5 fractions x 2 data seeds nested on p={p}"


def check_certificates(recipe):
    p = recipe.p
    params = init_params(recipe, 41337)
    E0 = we_freq_energy(params["embed.W_E"].numpy(), p)
    order = np.argsort(-E0) + 1
    i, j = int(order[0]), int(order[-1])
    out, cert = we_exchange(params, [(i, j)], p)          # asserts internally
    E1 = we_freq_energy(out["embed.W_E"].numpy(), p)
    assert abs(E1[i - 1] - E0[j - 1]) < 1e-5 * E0[j - 1]
    assert abs(E1[j - 1] - E0[i - 1]) < 1e-5 * E0[i - 1]
    for k in range(1, p // 2 + 1):
        if k not in (i, j):
            assert abs(E1[k - 1] - E0[k - 1]) <= 1e-5 * E0[k - 1]
    out2, cert2 = we_scale(params, {i: 0.5}, p)
    E2 = we_freq_energy(out2["embed.W_E"].numpy(), p)
    assert abs(E2[i - 1] - 0.25 * E0[i - 1]) <= 1e-5 * 0.25 * E0[i - 1]
    _, pc = perturb_params(params, 0.03, 99)
    assert abs(pc["rel"] - 0.03) < 1e-7
    z, zc = perturb_params(params, 0.0, 99)
    assert all(torch.equal(z[k], params[k]) for k in params), "eps=0 not a twin"
    # dispatch path used by the trainer
    apply_edit(params, {"kind": "we_exchange", "pairs": [[i, j]]}, p)
    apply_edit(params, {"kind": "we_scale", "scales": {str(i): 2.0}}, p)
    return (f"exchange f{i}<->f{j} energy_rel_err {cert['energy_rel_err']:.1e}, "
            f"norm_rel_err {cert['norm_rel_err']:.1e}, "
            f"other_freq_abs_err {cert['other_freq_abs_err']:.1e}; "
            f"scale err {cert2['energy_rel_err']:.1e}; perturb rel {pc['rel']:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_twin/_smoke")
    ap.add_argument("--out", default="twin/results/smoke")
    ap.add_argument("--manifests", default="twin/manifests/smoke")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    setup_determinism()
    t0 = time.time()
    spec = smoke_spec()
    recipe = spec["recipe"]
    runs = Path(args.runs)
    mdir = Path(args.manifests)
    if runs.exists() and not args.keep:
        shutil.rmtree(runs)

    print(f"twin smoke: p={recipe.p} d_model={recipe.d_model} "
          f"heads={recipe.num_heads} d_mlp={recipe.d_mlp}, cpu, no compile\n")

    # ---- 1. gates ------------------------------------------------------
    gates = {}

    def _gates():
        # 600 epochs, not 40: batch-width differences start at 1 ulp and only
        # become visible once the dynamics have amplified them, so a short gate
        # reports a false PASS on G3.
        gates.update(run_gates(recipe, runs / "_gates", "cpu", "off",
                               n_epochs=600))
        for k in ("G1", "G2_solo", "G2_batched", "within_batch"):
            assert gates[k], f"{k} failed"
        return ("G1/G2solo/G2batched/within_batch PASS; "
                f"G3={'PASS' if gates['G3'] else 'FAIL (fallback engaged)'}")

    check("gates", _gates)

    # ---- 2. edit certificates -----------------------------------------
    check("edit certificates", lambda: check_certificates(recipe))

    # ---- 3. mask nesting ----------------------------------------------
    check("mask nesting", lambda: check_masks(recipe.p))

    # ---- 4. end-to-end -------------------------------------------------
    state = {}

    def _stage1():
        mf = stage1_manifest(spec)
        write_json(mdir / "stage1.json", mf)
        n = run_manifest(mf, runs, "cpu", "off", args.width, verbose=False)
        state["n1"] = n
        return f"{n} base runs trained (BASES-A + BASES-B)"

    ok1 = check("stage 1 (mini bases)", _stage1)

    def _stage2gen():
        cohorts, readouts = stage2_manifests(spec, runs)
        # Smoke-only extra lane: a SECOND eps=0 control inside one fork group.
        # Two identical lanes in one batch must agree bitwise — this is the
        # within-batch comparison path the G3 fallback rule relies on.
        g = cohorts["fork"][0]["group"]
        dup = dict(next(j for j in cohorts["fork"] if j["group"] == g
                        and j["arm"] == "ctrl"))
        dup["name"] = f"{g}/eps0_ctrl_dup"
        cohorts["fork"].append(dup)
        state["cohorts"] = cohorts
        state["dup"] = (f"{g}/eps0_ctrl", dup["name"])
        for k, jobs in cohorts.items():
            write_json(mdir / f"{k}.json", wrap(spec, jobs))
        return "  ".join(f"{k}:{len(v)}" for k, v in cohorts.items()) + \
            f"  committees {[r['committee'] for r in readouts.values()]}"

    ok2 = ok1 and check("stage 2 (manifest generation)", _stage2gen)

    def _stage2train():
        n = 0
        for k in ("fork", "swap", "dial", "hold"):
            n += run_manifest(wrap(spec, state["cohorts"][k]), runs, "cpu",
                              "off", args.width, verbose=False)
        state["n2"] = n
        return f"{n} stage-2 runs trained across 4 cohorts"

    ok3 = ok2 and check("stage 2 (train all cohorts)", _stage2train)

    def _dupcheck():
        a, b = state["dup"]
        za, zb = (np.load(runs / a / "spectra.npz"), np.load(runs / b / "spectra.npz"))
        for k in ("test_acc", "train_loss", "coeffs"):
            assert np.array_equal(za[k], zb[k]), f"within-batch twins differ on {k}"
        return f"{a} == {b} bitwise (within-batch control path)"

    ok3 and check("within-batch twin identity", _dupcheck)

    def _analyze():
        rs = load_all(runs)
        rs = {k: v for k, v in rs.items() if not k.startswith("g")}
        res = {"bases": analyze_bases(rs), "fork": analyze_fork(rs),
               "swap": analyze_swap(rs), "dial": analyze_dial(rs),
               "hold": analyze_hold(rs)}
        for k, v in res.items():
            write_json(Path(args.out) / f"{k}_summary.json", v)
        assert res["fork"]["n_forks"] > 0 and res["swap"]["n_arms"] > 0
        assert res["dial"]["n_arms"] > 0 and res["hold"]["n_arms"] > 0
        assert res["fork"]["chi"] and res["hold"]["table"]
        state["res"] = res
        e = res["fork"]["eps0_bitwise_vs_base"]
        h = res["hold"]["ctrl_bitwise_vs_base"]
        return (f"fork {res['fork']['n_forks']} / swap {res['swap']['n_arms']} / "
                f"dial {res['dial']['n_arms']} / hold {res['hold']['n_arms']} arms; "
                f"eps0-vs-base {e['matched']}/{e['checked']}, "
                f"hold-ctrl-vs-base {h['matched']}/{h['checked']}")

    ok4 = ok3 and check("analysis runs end to end", _analyze)

    def _eps0():
        """Spec: the eps=0 fork must reproduce its base bitwise, "subject to the
        G3 fallback rule: compare within the same batch size".  So: train the
        eps=0 forks at EXACTLY the width the bases were trained at (one lane per
        base, 2 lanes = the BASES-A batch) and require bitwise reproduction
        there.  The cross-width count from the production cohort is reported
        alongside, and is expected to be partial because G3 fails."""
        pairs = make_twin.base_pairs(spec)
        T = spec["basesA"]["ckpts"][2]
        jobs = []
        for ds, iseed in pairs:
            b = make_twin.bname(ds, iseed)
            jobs.append({"name": f"forkmw/{b}/T{T:06d}/eps0", "cohort": "forkmw",
                         "group": f"forkmw/{b}", "data_seed": ds,
                         "init_seed": iseed, "frac": spec["basesA"]["frac"],
                         "start_epoch": T, "end_epoch": spec["basesA"]["epochs"],
                         "log_every": spec["log_every"],
                         "spectra_every": spec["spectra_every"], "ckpt_epochs": [],
                         "init": {"kind": "ckpt", "path":
                                  f"basesA/{b}/checkpoints/epoch_{T:06d}.pt"},
                         "edit": {"kind": "none"}, "base": f"basesA/{b}", "T": T,
                         "arm": "ctrl"})
        run_manifest(wrap(spec, jobs), runs, "cpu", "off", len(pairs), verbose=False)
        from analyze_twin import load_run, logs_equal
        n_ok = 0
        for j in jobs:
            a = load_run(runs / j["name"])
            bse = load_run(runs / j["base"])
            assert logs_equal(a, bse), f"{j['name']} != its base at matched width"
            n_ok += 1
        e = state["res"]["fork"]["eps0_bitwise_vs_base"]
        return (f"{n_ok}/{len(jobs)} eps=0 forks bitwise == base at MATCHED batch "
                f"width ({len(pairs)} lanes); cross-width production cohort: "
                f"{e['matched']}/{e['checked']} "
                f"(G3={'PASS' if gates.get('G3') else 'FAIL'} -> within-batch mode)")

    ok4 and check("eps=0 fork reproduces base (matched width)", _eps0)

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 72)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if detail:
            print(f"          {detail}")
    print("=" * 72)
    allok = all(ok for _, ok, _ in RESULTS)
    print(f"G3 (batch invariance): {'PASS' if gates.get('G3') else 'FAIL'} -> "
          f"comparison mode = "
          f"{'any (batch-invariant)' if gates.get('G3') else 'WITHIN-BATCH ONLY'}")
    print(f"twin smoke {'PASS' if allok else 'FAIL'} in {time.time() - t0:.0f}s")
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
