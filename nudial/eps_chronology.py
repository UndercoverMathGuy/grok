"""Per-tensor eps-crossing chronology on checkpointed runs (E5 instrument).

For every checkpoint of every run matching --glob, compute the FULL-BATCH
training gradient (the exact quantity Adam sees: d(mean train CE)/d theta,
stable-f32 loss) and record, per weight tensor:

    q10/q50/q90   quantiles of |g| over the tensor's coordinates
    frac_lt_eps   fraction of coordinates with |g| < adam_eps of the run
    l1, l2        norms of the tensor itself

Then per run: grok epoch (test acc >= .99), committee LOCK epoch (last
snapshot at which the committee set changed, 100-epoch resolution, rounded
UP to the next checkpoint), final committee, and for each tensor the first
checkpoint at which the MEDIAN |g| is below eps ("crossing epoch").

Forward prediction for an eps sweep (rule A: lock margin is eps-independent):
at the run's lock checkpoint, for each candidate eps' count the signal-
carrying tensors (degree-1 chain members W_E, W_V, W_O, W_in, W_out, W_U;
Q/K/pos/biases carry no margin degree per nudial/nu_decompose) whose
median |g| < eps'  ->  s = count/2  ->  K_pred = argmax_K M_K(p)/K^s.
Rule B (lock = first crossing) predicts s independent of eps -> flat.

Outputs a JSON (--out) with everything, prints the summary tables.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
from train_semifinal_torch import (Config, PARAM_KEYS, batched_final_logits,
                                   make_dataset, train_test_split,
                                   per_ex_stable_f32, committee_from_coeffs,
                                   grok_epoch)
from safetensors.torch import load_file

SIGNAL = ["embed.W_E", "blocks.0.attn.W_V", "blocks.0.attn.W_O",
          "blocks.0.mlp.W_in", "blocks.0.mlp.W_out", "unembed.W_U"]
SHORT = {"embed.W_E": "E", "pos_embed.W_pos": "pos", "blocks.0.attn.W_K": "K",
         "blocks.0.attn.W_Q": "Q", "blocks.0.attn.W_V": "V",
         "blocks.0.attn.W_O": "O", "blocks.0.mlp.W_in": "in",
         "blocks.0.mlp.b_in": "b_in", "blocks.0.mlp.W_out": "out",
         "blocks.0.mlp.b_out": "b_out", "unembed.W_U": "U"}
EPS_GRID = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def mk_table(p):
    d = json.load(open(Path(__file__).resolve().parent.parent
                       / "notes" / "mk_tables_sweep.json"))[str(p)]
    return {int(k): v["M"] for k, v in d.items()}


def k_pred(MK, s, kmax=12):
    ks = [k for k in sorted(MK) if k <= kmax]
    vals = [MK[k] / k ** s for k in ks]
    return ks[int(np.argmax(vals))]


def staircase(MK):
    """s -> argmax K table on a fine grid, and the s-intervals per K."""
    out = {}
    for s in np.arange(0.0, 3.51, 0.01):
        out.setdefault(k_pred(MK, s), []).append(round(float(s), 2))
    return {k: (min(v), max(v)) for k, v in out.items()}


def grad_stats(cfg, ck, tr_tok, tr_lab, eps):
    pr = {k: ck[k].unsqueeze(0).clone().requires_grad_() for k in PARAM_KEYS}
    lg = batched_final_logits(pr, tr_tok, cfg)
    loss = per_ex_stable_f32(lg, tr_lab).mean(1).sum()
    grads = torch.autograd.grad(loss, [pr[k] for k in PARAM_KEYS])
    st = {}
    for k, g in zip(PARAM_KEYS, grads):
        a = g[0].abs().flatten().double()
        w = ck[k].flatten().double()
        q = torch.quantile(a, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
        st[SHORT[k]] = dict(q10=float(q[0]), q50=float(q[1]), q90=float(q[2]),
                            frac_lt_eps=float((a < eps).double().mean()),
                            l1=float(w.abs().sum()), l2=float(w.norm()),
                            n=int(a.numel()))
    return st, float(loss)


def lock_epoch(z):
    comms = [tuple(committee_from_coeffs(z["coeffs"][i]))
             for i in range(len(z["epochs"]))]
    for i in range(len(comms) - 1, -1, -1):
        if comms[i] != comms[-1]:
            return int(z["epochs"][i + 1])
    return int(z["epochs"][0])


def analyze_run(rd, device):
    rd = Path(rd)
    cfg = Config.load(rd / "config.json")
    z = np.load(rd / "spectra.npz")
    tokens, labels = make_dataset(cfg)
    mask = train_test_split(cfg)
    tr_tok = torch.from_numpy(tokens[mask]).unsqueeze(0).to(device)
    tr_lab = torch.from_numpy(labels[mask]).unsqueeze(0).to(device)
    cks = sorted((rd / "checkpoints").glob("epoch_*.safetensors"))
    chron = []
    for c in cks:
        ep = int(c.stem.split("_")[1])
        ck = {k: v.to(device) for k, v in load_file(str(c)).items()
              if k in PARAM_KEYS}
        st, loss = grad_stats(cfg, ck, tr_tok, tr_lab, cfg.adam_eps)
        chron.append(dict(epoch=ep, train_ce=loss, tensors=st))
    ge, le = grok_epoch(z), lock_epoch(z)
    eps_ck = [c["epoch"] for c in chron]
    lock_ck = min((e for e in eps_ck if e >= le), default=eps_ck[-1])
    grok_ck = min((e for e in eps_ck if e >= ge), default=eps_ck[-1]) if ge > 0 else None
    cross = {}
    for t in SIGNAL:
        s = SHORT[t]
        hit = [c["epoch"] for c in chron if c["tensors"][s]["q50"] < cfg.adam_eps]
        cross[s] = hit[0] if hit else None
    at_lock = next(c for c in chron if c["epoch"] == lock_ck)["tensors"]
    return dict(run=str(rd), eps=cfg.adam_eps, p=cfg.p, grok=ge, lock=le,
                lock_ck=lock_ck, grok_ck=grok_ck,
                committee=committee_from_coeffs(z["coeffs"][-1]),
                test_acc=float(z["test_acc"][-1]),
                crossing_epoch=cross, at_lock=at_lock, chron=chron)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs_torch/p-113/*/*")
    ap.add_argument("--out", default="notes/eps_chronology.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    runs = sorted(d for d in glob.glob(args.glob)
                  if (Path(d) / "spectra.npz").exists()
                  and len(list((Path(d) / "checkpoints").glob("epoch_*"))) > 2)
    print(f"{len(runs)} runs with checkpoints", flush=True)
    res = [analyze_run(r, args.device) for r in runs]
    MK = mk_table(res[0]["p"])
    stair = staircase(MK)
    print("\nstaircase p=%d  (s-interval -> K_pred):" % res[0]["p"])
    for k in sorted(stair):
        print(f"  K={k}: s in [{stair[k][0]:.2f}, {stair[k][1]:.2f}]")
    print("\nper run: grok/lock epochs, K, crossing epoch of median|g| per tensor")
    hdr = "run".ljust(40) + "grok   lock   K  " + "  ".join(f"{SHORT[t]:>5}" for t in SIGNAL)
    print(hdr)
    for r in res:
        ce = r["crossing_epoch"]
        print(Path(r["run"]).name.ljust(40)
              + f"{r['grok']:5d}  {r['lock']:5d}  {len(r['committee'])}  "
              + "  ".join(f"{str(ce[SHORT[t]]):>5}" for t in SIGNAL))
    print("\nmedian |g| per signal tensor AT LOCK checkpoint (log10):")
    for r in res:
        al = r["at_lock"]
        print(Path(r["run"]).name.ljust(40)
              + "  ".join(f"{np.log10(al[SHORT[t]]['q50']):6.2f}" for t in SIGNAL))
    # forward prediction, rule A
    print("\nRULE A forward prediction: K_pred per run at each eps' "
          "(tensors crossed at the eps=1e-8 lock checkpoint)")
    pred = {}
    for e in EPS_GRID:
        col = []
        for r in res:
            n_cross = sum(r["at_lock"][SHORT[t]]["q50"] < e for t in SIGNAL)
            col.append(k_pred(MK, n_cross / 2))
        pred[e] = col
        hist = {k: col.count(k) for k in sorted(set(col))}
        print(f"  eps'={e:.0e}: K_pred per run {col}  hist {hist}")
    json.dump(dict(runs=res, staircase={str(k): v for k, v in stair.items()},
                   rule_A_pred={f"{e:.0e}": v for e, v in pred.items()},
                   MK=MK), open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
