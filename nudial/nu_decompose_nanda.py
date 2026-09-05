"""Per-tensor nu decomposition on Neel Nanda's ORIGINAL mainline grokking run
(full_run_data.pth from the progress-measures paper's own download cell).
Same instrument as nudial/nu_decompose.py: c_T = dlog f / dlog s_T per weight
tensor, central FD in float64, live vs frozen attention. Sanity: full-grid
accuracy must be 1.0 (also proves the reimplemented forward matches his
architecture semantics exactly)."""
import math
import numpy as np
import torch

torch.set_grad_enabled(False)
PTH = ("/private/tmp/claude-502/-Users-ruhaanrajadhyaksha-projects-grok/"
       "7f0dff89-eb6a-4e96-9cbb-a47cc7b2496d/scratchpad/full_run_data.pth")
d = torch.load(PTH, map_location="cpu", weights_only=False)
print("config:", d["config"])
P = d["config"]["p"]; H_, DH = 4, 32
epochs = d["epochs"]

a = torch.arange(P).repeat_interleave(P)
b = torch.arange(P).repeat(P)
tokens = torch.stack([a, b, torch.full_like(a, P)], 1)
labels = (a + b) % P
sub = torch.tensor(np.random.default_rng(0).choice(P * P, 3000, replace=False))

TENSORS = ["embed.W_E", "pos_embed.W_pos",
           "blocks.0.attn.W_Q", "blocks.0.attn.W_K",
           "blocks.0.attn.W_V", "blocks.0.attn.W_O",
           "blocks.0.mlp.W_in", "blocks.0.mlp.b_in",
           "blocks.0.mlp.W_out", "blocks.0.mlp.b_out",
           "unembed.W_U"]
SHORT = {t: t.split(".")[-1] for t in TENSORS}

def prep(sd):
    w = {k: v.double() for k, v in sd.items() if "mask" not in k}
    return {k: (v.reshape(H_ * DH, -1) if ".attn.W_" in k and v.dim() == 3 else v)
            for k, v in w.items()}

def forward(w, scale=None, attn_from=None):
    s = lambda k: w[k] * (scale.get(k, 1.0) if scale else 1.0)
    def stream(ws):
        E = ws("embed.W_E")
        return E.T[tokens] + ws("pos_embed.W_pos")[None, :3]
    def pattern(ws):
        x = stream(ws)
        k = (x @ ws("blocks.0.attn.W_K").T).view(-1, 3, H_, DH)
        q = (x[:, -1] @ ws("blocks.0.attn.W_Q").T).view(-1, 1, H_, DH)
        return torch.softmax((k * q).sum(-1) / math.sqrt(DH), dim=1)
    x = stream(s)
    attn = pattern((lambda k: attn_from[k]) if attn_from is not None else s)
    v = (x @ s("blocks.0.attn.W_V").T).view(-1, 3, H_, DH)
    z = (attn.unsqueeze(-1) * v).sum(1).reshape(-1, H_ * DH)
    xf = x[:, -1] + z @ s("blocks.0.attn.W_O").T
    act = torch.relu(xf @ s("blocks.0.mlp.W_in").T + s("blocks.0.mlp.b_in"))
    xf = xf + act @ s("blocks.0.mlp.W_out").T + s("blocks.0.mlp.b_out")
    return xf @ s("unembed.W_U")

def observables(logits):
    ly = labels[sub]; z = logits[sub]
    zy = z.gather(1, ly[:, None]).squeeze(1)
    others = z.clone(); others[torch.arange(len(ly)), ly] = -torch.inf
    return zy, zy - others.max(1).values

H_FD = 1e-4
def report(sd, tag):
    w = prep(sd)
    logits = forward(w)
    acc = (logits[:, :P].argmax(1) == labels).double().mean().item()
    base_logit, base_margin = observables(logits)
    print(f"\n### {tag}: full-grid acc = {acc:.4f}, "
          f"median margin = {base_margin.median():.2f} nats")
    def elast(keys, attn_from):
        lu, mu = observables(forward(w, {k: 1 + H_FD for k in keys}, attn_from))
        ld, md = observables(forward(w, {k: 1 - H_FD for k in keys}, attn_from))
        return (((lu - ld) / (2 * H_FD * base_logit)).median().item(),
                ((mu - md) / (2 * H_FD * base_margin)).median().item())
    for mode in ("live", "frozen"):
        af = w if mode == "frozen" else None
        tl, tm = elast(TENSORS, af)
        print(f"  {mode:6s} TOTAL   logit {tl:6.3f}  margin {tm:6.3f}")
        for t in TENSORS:
            el, em = elast([t], af)
            print(f"         {SHORT[t]:8s} {el:7.3f}   {em:7.3f}")

report(d["model"], f"final model (epoch {epochs[-1]})")
report(d["state_dicts"][400], f"checkpoint epoch {epochs[400]}")
