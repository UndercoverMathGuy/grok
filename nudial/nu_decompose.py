"""Compute nu for grokked Nanda-setup transformers from the known circuit,
per weight tensor, instead of measuring one lumped slope.

Instrument: per-tensor Euler contribution.  For observable f (correct-class
logit, or per-example margin) and tensor T:

    c_T = d log f / d log s_T   at s_T = 1     (scale ONLY tensor T by s_T)

By the Euler identity these contributions sum over tensors to the total
nu_eff = d log f / d log t (all weights scaled together).  Central finite
differences in float64, h = 1e-4.

Two modes:
  live   — the model as-is (softmax active)
  frozen — attention pattern computed once from the UNSCALED weights and
           held fixed; the model is then an exact sum of homogeneous paths.

Circuit prediction (frozen): c = 1 for each of W_E, W_V, W_O, W_in, W_out,
W_U;  c = 0 for W_Q, W_K (exactly), and ~0 for W_pos, b_in, b_out.
"""
import json, math, glob
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_grad_enabled(False)
P = 113
H, DH = 4, 32
RUNS = sorted(glob.glob(
    "/Users/ruhaanrajadhyaksha/projects/grok/runs_dial/w0512_wd1/ds*_is*"))
TENSORS = ["embed.W_E", "pos_embed.W_pos",
           "blocks.0.attn.W_Q", "blocks.0.attn.W_K",
           "blocks.0.attn.W_V", "blocks.0.attn.W_O",
           "blocks.0.mlp.W_in", "blocks.0.mlp.b_in",
           "blocks.0.mlp.W_out", "blocks.0.mlp.b_out",
           "unembed.W_U"]
SHORT = {t: t.split(".")[-1] for t in TENSORS}

a = torch.arange(P).repeat_interleave(P)
b = torch.arange(P).repeat(P)
tokens = torch.stack([a, b, torch.full_like(a, P)], 1)     # (N,3)
labels = (a + b) % P
rng = np.random.default_rng(0)
sub = torch.tensor(rng.choice(P * P, 3000, replace=False))

def forward(w, scale=None, attn_from=None):
    """w: dict of float64 tensors. scale: {key: factor}. attn_from: weight
    dict to compute the attention pattern from (frozen mode); None = live."""
    s = lambda k: w[k] * (scale.get(k, 1.0) if scale else 1.0)
    def stream(ws):
        E = ws("embed.W_E")                                # (d, v)
        x = E.T[tokens] + ws("pos_embed.W_pos")[None, :3]  # (N,3,d)
        return x
    def pattern(ws):
        x = stream(ws)
        k = (x @ ws("blocks.0.attn.W_K").T).view(-1, 3, H, DH)
        q = (x[:, -1] @ ws("blocks.0.attn.W_Q").T).view(-1, 1, H, DH)
        return torch.softmax((k * q).sum(-1) / math.sqrt(DH), dim=1)  # (N,3,H)
    x = stream(s)
    attn = pattern((lambda k: attn_from[k]) if attn_from is not None else s)
    v = (x @ s("blocks.0.attn.W_V").T).view(-1, 3, H, DH)
    z = (attn.unsqueeze(-1) * v).sum(1).reshape(-1, H * DH)
    xf = x[:, -1] + z @ s("blocks.0.attn.W_O").T
    act = torch.relu(xf @ s("blocks.0.mlp.W_in").T + s("blocks.0.mlp.b_in"))
    xf = xf + act @ s("blocks.0.mlp.W_out").T + s("blocks.0.mlp.b_out")
    return xf @ s("unembed.W_U")                           # (N, v)

def observables(logits):
    ly = labels[sub]
    z = logits[sub]
    zy = z.gather(1, ly[:, None]).squeeze(1)
    others = z.clone(); others[torch.arange(len(ly)), ly] = -torch.inf
    return zy, zy - others.max(1).values                   # logit, margin

H_FD = 1e-4
def elasticity(w, keys, attn_from):
    up = forward(w, {k: 1 + H_FD for k in keys}, attn_from)
    dn = forward(w, {k: 1 - H_FD for k in keys}, attn_from)
    lu, mu = observables(up); ld, md = observables(dn)
    e_log = (lu - ld) / (2 * H_FD * base_logit)
    e_mar = (mu - md) / (2 * H_FD * base_margin)
    return e_log.median().item(), e_mar.median().item()

rows = []
for run in RUNS:
    w = {k: v.double() for k, v in load_file(f"{run}/checkpoints/epoch_50000.safetensors").items()}
    w = {k: (v.reshape(H * DH, -1) if ".attn.W_" in k and v.dim() == 3 else v)
         for k, v in w.items()}
    logits = forward(w)
    acc = (logits[:, :P].argmax(1) == labels).double().mean().item()
    if acc < 0.99:
        print(f"skip {run.split('/')[-1]} acc={acc:.3f}"); continue
    base_logit, base_margin = observables(logits)
    res = {"run": run.split("/")[-1]}
    for mode in ("live", "frozen"):
        af = w if mode == "frozen" else None
        tot_l, tot_m = elasticity(w, TENSORS, af)
        res[mode] = {"TOTAL": (tot_l, tot_m)}
        for t in TENSORS:
            res[mode][SHORT[t]] = elasticity(w, [t], af)
    rows.append(res)
    print(f"{res['run']}  acc={acc:.4f}  "
          f"nu live {res['live']['TOTAL'][0]:.2f}/{res['live']['TOTAL'][1]:.2f}  "
          f"frozen {res['frozen']['TOTAL'][0]:.2f}/{res['frozen']['TOTAL'][1]:.2f}")

names = ["TOTAL"] + [SHORT[t] for t in TENSORS]
for mode in ("live", "frozen"):
    print(f"\n== {mode} attention: median over {len(rows)} grokked runs "
          f"(logit / margin elasticity per tensor) ==")
    for n in names:
        el = float(np.median([r[mode][n][0] for r in rows]))
        em = float(np.median([r[mode][n][1] for r in rows]))
        print(f"  {n:8s}  {el:7.3f}   {em:7.3f}")
