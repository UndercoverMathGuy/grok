"""Shared discovery + utilities for the SEMIFINAL claim-backing analyses.

Every script in this directory is ANALYSIS-ONLY (reads runs/, never trains)
and general-purpose: it discovers every compatible run under runs/ by
config + artifacts, so new runs are picked up automatically.

Cohorts (by config, not by folder name):
  natural-normal   embed_init=normal,     attn_init=normal, natural family
  surgical         embed_init=normal but init-edited (surgery/dose/... family)
  orth-flat        embed_init=orthogonal, attn_init=normal
  double-flat      embed_init=orthogonal, attn_init=isometric
Excluded: warm-start / distill / non-selection families, runs without
spectra.npz, runs that never grokked (test acc < 0.99).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# NOTE (2026-08-03): the pre-v2 dataset and its lab notebook live under
# legacy/ (legacy/runs, legacy/findings) and are deliberately NOT
# discovered — runs/ holds only the v2 dataset built by
# training/train_semifinal_v2.py. Legacy-only sections (claim 2A dynamics
# variants, claim 3E transplants, the legacy steering arms) report from
# the archived captures in results/legacy-2026-08-03/.

from grok.config import Config
from grok.data import make_dataset
from grok.fourier import Fourier

# Unified committee detector (2026-08-03, replaces mask_lottery's + the old
# per-run FIX dict in claim4): largest-log-gap detector, then drop stragglers
# below COMMITTEE_FLOOR x the run's max |coeff|. The floor sits in the widest
# empty band of the pooled member/background amplitude-ratio distribution
# across all 122 runs (members >= 0.027, background <= 0.0141 except the four
# straggler entries this rule reclassifies); any value in [0.015, 0.026]
# yields identical committees. The same constant is the claim-3 adoption
# criterion.
COMMITTEE_FLOOR = 0.02

def committee_from_coeffs(coeffs, floor=COMMITTEE_FLOOR):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())

NATURAL_FAMS = {"og_seed0", "seed0", "seed1", "seed2", "p-113", "p-127",
                "p-157"}
SURGICAL_FAMS = {"surgery", "surgery2", "transplant", "dosefarm", "gkrotate",
                 "collisionfarm", "suppress", "chaospair"}
EXCLUDED_FAMS = {"eviction", "naive_distill", "linear_interp",
                 "ce_from_onehot_t80_e390", "ce_from_onehot_t80_e390_lr1e4"}

_FOURIER = {}
def fourier(p):
    if p not in _FOURIER:
        _FOURIER[p] = Fourier(p)
    return _FOURIER[p]

def discover(require_e0=False):
    """Yield dicts for every compatible grokked run under runs/."""
    for cj in sorted(ROOT.joinpath("runs").rglob("config.json")):
        d = cj.parent
        fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
        if fam in EXCLUDED_FAMS or not (d / "spectra.npz").exists():
            continue
        cfg = Config.load(cj)
        e0 = d / "checkpoints" / "epoch_00000.safetensors"
        if require_e0 and not e0.exists():
            continue
        z = np.load(d / "spectra.npz")
        if float(z["test_acc"][-1]) < 0.99:
            continue
        emb = getattr(cfg, "embed_init", "normal")
        att = getattr(cfg, "attn_init", "normal")
        if emb == "orthogonal" and att == "isometric":
            cohort = "double-flat"
        elif emb == "orthogonal":
            cohort = "orth-flat"
        elif fam in SURGICAL_FAMS:
            cohort = "surgical"
        elif fam in NATURAL_FAMS:
            cohort = "natural-normal"
        else:
            cohort = "other-normal"
        yield dict(dir=d, rel=str(d.relative_to(ROOT / "runs")), fam=fam,
                   cfg=cfg, spectra=z, e0=e0 if e0.exists() else None,
                   cohort=cohort,
                   committee=committee_from_coeffs(z["coeffs"][-1]))

_TOKENS = {}
def tokens_and_fidx(cfg):
    """Full-batch tokens + per-frequency 2D index stack. Cached by p alone:
    make_dataset's tokens are all p^2 inputs (data_seed only affects the
    train/test split, which no analysis here uses)."""
    p = cfg.p
    if p not in _TOKENS:
        tokens, _ = make_dataset(cfg)
        f = fourier(p)
        fidx = np.stack([f.freq_indices_2d(k) for k in range(1, p // 2 + 1)])
        _TOKENS[p] = (tokens, fidx)
    return _TOKENS[p]

def mlp_freq_energy(m, tokens, fidx, p):
    """Per-(frequency, neuron) energy of the final-position MLP activations
    — the single forward-pass committee readout used by claims 1a and 1b."""
    _, cache = m.run_with_cache(tokens)
    acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier(p).fft2d(centered) ** 2
    nf = p // 2
    return fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool)
    lab[np.array(members) - 1] = True
    # midranks (1-based): ties share their average rank, so heavily tied
    # score vectors (e.g. per-neuron winner counts, mostly 0) score 0.5
    # instead of an index-order artifact.
    r = rankdata(scores)
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def freq_energy(M, p):
    F = M @ fourier(p).basis.T
    E = (F ** 2).sum(0)
    nf = p // 2
    return E[1::2][:nf] + E[2::2][:nf]

def fold(x, p):
    x %= p
    return min(x, p - x)

def violations(S, p):
    S = sorted(S)
    ss = set(S)
    out = []
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            i, j = S[a], S[b]
            if fold(i + j, p) in ss or fold(i - j, p) in ss:
                out.append((i, j))
    return out

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0

def grok_epoch(z):
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    return int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1

def menu_at(z, epoch=3000, top=8):
    i = int(np.argmin(np.abs(z["epochs"] - epoch)))
    return (np.argsort(np.abs(z["coeffs"][i]))[::-1][:top] + 1).tolist()

def cluster_key(r):
    """Independent-init cluster for a discovered run: runs sharing an epoch-0
    lottery draw (same init trained under many dynamics, or surgical arms
    carved from one base) are ONE unit of evidence, not several."""
    parts = r["rel"].split("/")
    c = r["cohort"]
    if c in ("orth-flat", "double-flat"):
        return ("init", r["cfg"].p, r["cfg"].data_seed, r["cfg"].init_seed)
    if c == "surgical":
        if parts[0] in ("surgery", "surgery2"):
            return ("base", "seed27058")
        if parts[0] == "gkrotate":
            # gkrotate/gain_* is the legacy seed27058 suite; the v2 layout is
            # gkrotate/<base>/gain_*.
            return ("base", parts[1] if len(parts) > 2 else "seed27058")
        if parts[0] in ("dosefarm", "suppress", "chaospair"):
            # suppress/<base>; chaospair/<base>/arm[AB] — the two arms share
            # one base init (they differ by a sub-threshold swap).
            return ("base", parts[1])
        if parts[0] in ("collisionfarm", "transplant"):
            return ("base", parts[1].split("_")[0])
    if c == "natural-normal":
        # natural inits are independent; cluster by data mask (the one shared
        # ingredient — mask popularity correlates outcomes within a cell)
        return ("mask", r["cfg"].p, r["cfg"].data_seed)
    return ("run", r["rel"])


def find_base_run(name):
    """Locate a natural base run by directory basename (e.g. 'seed27058').

    Raises on an ambiguous basename (present under more than one natural
    family): silently picking one would attribute arms to the wrong base.
    """
    hits = [cj.parent
            for cj in ROOT.joinpath("runs").rglob(f"{name}/config.json")
            if str(cj.parent.relative_to(ROOT / "runs")).split("/")[0]
            in NATURAL_FAMS]
    if len(hits) > 1:
        raise ValueError(f"base name {name!r} is ambiguous across natural "
                         f"families: {sorted(str(h) for h in hits)}")
    return hits[0] if hits else None
