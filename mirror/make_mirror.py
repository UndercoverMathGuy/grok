"""Mirror cohort arm generation (prereg: mirror/README.md, DRAFT).

Draws the shared content pool, picks the cohort g, compiles the installed
inits (natural substrate, renorm=global, dose s), builds the exact-twin
checkpoints and masks, and writes mirror/arms/mirror_manifest.json.

Run BEFORE training. Smoke: python3 mirror/make_mirror.py --smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mirror"))

from compile_torch import (compile_init, draw_feasible_set, feasibility,  # noqa: E402
                           load_ckpt, natural_bases, save_ckpt, violations)

ARMS = ROOT / "mirror" / "arms"
MIRROR_SEED = 20260819
P = 113
NF = P // 2
DOSE = 3.0
N_CONTENTS = 3
K = 4


def fold(k):
    return int(min(k % P, (-k) % P))


def freq_image(g, k):
    return fold(g * k)


def vocab_perm(g):
    """pi[x] = g*x mod p for tokens 0..p-1; the '=' token p is fixed."""
    pi = np.arange(P + 1)
    pi[:P] = (g * np.arange(P)) % P
    return pi


def permute_params(params, g):
    """Exact vocab relabeling: column x of W_E / W_U moves to column g*x.
    Touches nothing else — positions, attention, MLP are vocab-free."""
    pi = vocab_perm(g)
    out = dict(params)
    for key in ("embed.W_E", "unembed.W_U"):
        W = np.asarray(params[key])
        W2 = W.copy()
        W2[:, pi] = W
        out[key] = W2
    return out


def permute_mask(m, g):
    """Train-mask image under (a,b) -> (ga, gb); lexicographic pair index."""
    a = np.repeat(np.arange(P), P)
    b = np.tile(np.arange(P), P)
    idx2 = ((g * a) % P) * P + (g * b) % P
    m2 = np.zeros_like(m)
    m2[idx2] = m
    return m2


def pick_g(pool, rng):
    """Smallest g >= 2 whose mirrors are disjoint from every original and
    violations-free (symmetry guarantees the latter; asserted anyway)."""
    for g in range(2, P):
        ok = True
        for S, _ in pool:
            Sg = sorted(freq_image(g, k) for k in S)
            if len(set(Sg)) != K or set(Sg) & set(S) or violations(Sg, P):
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("no g found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(MIRROR_SEED)
    pool = []
    while len(pool) < N_CONTENTS:
        S, feas = draw_feasible_set(rng, P, K, 40.0)
        if S not in [c[0] for c in pool]:
            pool.append((S, feas))
    g = pick_g(pool, rng)
    print(f"g = {g}; contents (K={K}, drawn once):")
    mirrors = []
    for i, (S, feas) in enumerate(pool):
        Sg = sorted(freq_image(g, k) for k in S)
        fg = feasibility(Sg, P, floor_pct=0.0)   # recorded; symmetry-matched
        assert not violations(Sg, P)
        mirrors.append((Sg, fg))
        print(f"  content{i}: {S}  mirror: {Sg}  "
              f"lp_pct {feas['lp_pct']:.0f}/{fg['lp_pct']:.0f}")

    if args.smoke:
        pi = vocab_perm(g)
        assert sorted(pi.tolist()) == list(range(P + 1)) and pi[P] == P
        d = np.random.default_rng(0).normal(size=(16, P + 1))
        pp = permute_params({"embed.W_E": d, "unembed.W_U": d.copy()}, g)
        assert np.allclose(np.linalg.norm(pp["embed.W_E"]), np.linalg.norm(d))
        assert np.allclose(pp["embed.W_E"][:, pi], d)          # round trip
        m = np.zeros(P * P, bool)
        m[:100] = True
        m2 = permute_mask(m, g)
        assert m2.sum() == m.sum()
        a, b = 7, 31
        assert m2[((g * a) % P) * P + (g * b) % P] == m[a * P + b]
        # frequency-image bijection on folded frequencies
        img = sorted(freq_image(g, k) for k in range(1, NF + 1))
        assert img == list(range(1, NF + 1))
        print("SMOKE OK: permutation algebra, mask image, freq bijection")
        return

    bases = natural_bases(P)[:8]
    assert len(bases) == 8, f"need 8 natural bases, found {len(bases)}"
    names = [b[0].name for b in bases]
    assert len(set(names)) == 8, f"base leaf names collide: {names}"
    (ARMS / "ckpts").mkdir(parents=True, exist_ok=True)
    (ARMS / "masks").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "cloud"))
    from train_semifinal_torch import Config as TC, train_test_split

    manifest = []

    def rec(tag, bdir, kind, ckpt, cell, **kw):
        cfg = json.loads((bdir / "config.json").read_text()) | dict(
            num_epochs=30_000, save_every=30_000)
        return dict(tag=tag, kind=kind, base=bdir.name,
                    ckpt=str(Path(ckpt).relative_to(ROOT)) if ckpt else None,
                    run_rel=f"runs_mirror/{tag}", cell=cell, config=cfg, **kw)

    for bdir, bcfg, e0 in bases:
        bname = bdir.name
        params = load_ckpt(e0)
        installed0 = None                       # content0 C-arm, for the twin
        for ci in range(N_CONTENTS):
            for side, (S, feas) in (("orig", pool[ci]), ("mirr", mirrors[ci])):
                tag = f"{bname}_c{ci}_{side}"
                compiled, rep = compile_init(params, P, S, safety=DOSE)
                assert rep["spec_met"], (tag, rep)
                ck = ARMS / "ckpts" / f"{tag}.safetensors"
                save_ckpt(ck, compiled)
                if ci == 0 and side == "orig":
                    installed0 = compiled
                manifest.append(rec(tag, bdir, "install", ck,
                                    dict(base=bname, content=ci, side=side),
                                    targets=S, feasibility=feas, report=rep))
        # control
        manifest.append(rec(f"{bname}_control", bdir, "control", e0,
                            dict(base=bname, content=None, side=None),
                            targets=[], feasibility=None, report=None))
        # twin of the content0 orig install: permuted params + permuted mask
        twin = permute_params(installed0, g)
        ck = ARMS / "ckpts" / f"{bname}_twin.safetensors"
        save_ckpt(ck, twin)
        m = train_test_split(TC(**{k: v for k, v in
                                   (json.loads((bdir / "config.json")
                                    .read_text())).items()
                                   if k in TC.__dataclass_fields__}))
        mp = ARMS / "masks" / f"{bname}_twin.npy"
        np.save(mp, permute_mask(m, g))
        # token perm x -> g*x maps FREQUENCY k -> fold(g^-1 * k)
        manifest.append(rec(f"{bname}_twin", bdir, "twin", ck,
                            dict(base=bname, content=0, side="twin"),
                            targets=sorted(freq_image(pow(g, -1, P), k)
                                           for k in pool[0][0]),
                            feasibility=None, report=None,
                            mask=str(mp.relative_to(ROOT)),
                            twin_of=f"{bname}_c0_orig"))
        print(f"  {bname}: {2 * N_CONTENTS} installs + control + twin",
              flush=True)

    out = ARMS / "mirror_manifest.json"
    out.write_text(json.dumps(dict(g=g, dose=DOSE, seed=MIRROR_SEED,
                                   contents=[c[0] for c in pool],
                                   mirrors=[m_[0] for m_ in mirrors],
                                   arms=manifest), indent=1))
    print(f"{len(manifest)} arms -> {out}")


if __name__ == "__main__":
    main()
