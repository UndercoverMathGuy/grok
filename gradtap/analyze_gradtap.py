"""Analyze the gradient-tap trajectory cohorts (runs_gradtap/).

D11 (32 K=5 installs): when does the gradient identify the evictee?
  For each arm: evicted = installed - realized. Per installed member the
  tap gives dot_k(t) (>0 = loss decays the freq) and logit energy E_k(t).
  Readouts: is the evictee under distinctive decay pressure from epoch ~0,
  and how early does argmax-dot over installed members point at it?

C8 (64 mirror arms): where is the C-vs-gC timing dispersion generated?
  Per install run, own-committee scalars: on-committee grad share s(t) and
  committee-relative growth rate r(t) = -sum dot_k / sum E_k. Paired
  |orig - mirr| trajectories vs the (control, twin) float-divergence
  envelope, in phase coordinates (t / own t_gen).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
from train_semifinal_torch import committee_from_coeffs, grok_epoch  # noqa: E402

GT = ROOT / "runs_gradtap"


def load(rd):
    return dict(np.load(rd / "spectra.npz"))


def d11():
    man = json.loads((ROOT / "compiler" / "arms" / "phaseB_manifest.json")
                     .read_text())
    arms = {a["tag"]: a for a in man if "_K5_" in a["tag"]}
    rows = []
    for rd in sorted((GT / "d11").iterdir()):
        a = arms[rd.name]
        z = load(rd)
        inst = a["targets"]
        fin = committee_from_coeffs(z["coeffs"][-1])
        ev = sorted(set(inst) - set(fin))
        ge = grok_epoch(z)
        if not ev:
            rows.append(dict(tag=rd.name, grok=ge, installed=inst,
                             final=fin, evicted=[], ident_ep=None))
            continue
        ep = z["epochs"]
        dot = z["grad_dot"][:, [k - 1 for k in inst]]        # (S, 5)
        is_ev = np.isin(inst, ev)
        # integrated loss pressure on W_E energy: dE_k/dt|loss = -2*lr*dot_k,
        # so cum_k(t) = cumsum(dot_k) is the cumulative decay each member has
        # received (one space, no mixing). Identification epoch = first
        # snapshot from which the max-cum member is an evictee and stays one
        # for 5 consecutive snapshots.
        cum = np.cumsum(dot, 0)
        hit = is_ev[cum.argmax(1)]
        ident = None
        for i in range(len(hit) - 5):
            if hit[i:i + 5].all():
                ident = int(ep[i])
                break
        # was the evictee simply the weakest injection? (veto-law check)
        E0 = z["energy"][0, [k - 1 for k in inst]]
        weakest = inst[int(np.argmin(E0))]
        w = ep <= 200
        rows.append(dict(
            tag=rd.name, grok=ge, installed=inst, final=fin, evicted=ev,
            ident_ep=ident, weakest_is_evicted=bool(weakest in ev),
            early_cumdot_evicted=float(cum[w][-1, is_ev].mean()),
            early_cumdot_kept=float(cum[w][-1, ~is_ev].mean())))
    ne = [r for r in rows if r["evicted"]]
    idents = [r["ident_ep"] for r in ne if r["ident_ep"] is not None]
    early_flag = sum(r["early_cumdot_evicted"] > r["early_cumdot_kept"]
                     for r in ne)
    wk = sum(r["weakest_is_evicted"] for r in ne)
    print(f"d11: {len(rows)} arms, {len(ne)} with evictions "
          f"({sum(len(r['evicted']) for r in ne)} members evicted)")
    print(f"d11: weakest-injection member evicted in {wk}/{len(ne)} arms")
    print(f"d11: evictee has larger cumulative decay pressure by ep 200 in "
          f"{early_flag}/{len(ne)} arms")
    print(f"d11: sustained max-cum-pressure identification in "
          f"{len(idents)}/{len(ne)} arms, median epoch "
          f"{np.median(idents) if idents else float('nan'):.0f}")
    return rows


def c8_scalars(z, comm):
    idx = [k - 1 for k in comm]
    gE = z["grad_energy"]
    s = gE[:, idx].sum(1) / np.maximum(gE.sum(1), 1e-30)
    r = -z["grad_dot"][:, idx].sum(1) \
        / np.maximum(z["energy"][:, idx].sum(1), 1e-12)
    return s, r


def c8():
    man = json.loads((ROOT / "mirror" / "arms" / "mirror_manifest.json")
                     .read_text())
    arms = {a["tag"]: a for a in man["arms"]}
    # twin envelope: (control, twin) per base, same scalar construction
    sys.path.insert(0, str(ROOT / "mirror"))
    from make_mirror import freq_image
    gi = pow(man["g"], -1, 113)
    perm = np.array([freq_image(gi, k) - 1 for k in range(1, 57)])
    bases = sorted({a["base"] for a in man["arms"]})
    env = []
    for b in bases:
        zc = load(GT / "c8" / f"{b}_control")
        zt = load(GT / "c8" / f"{b}_twin")
        # committee label from the ORIGINAL 30k control run (the 5k rerun
        # ends pre-grok, so its final-coeff committee readout is noise)
        cc = committee_from_coeffs(
            np.load(ROOT / "runs_mirror" / f"{b}_control" / "spectra.npz")
            ["coeffs"][-1])
        sc, rc = c8_scalars(zc, cc)
        # twin lives in mapped frequency space; map committee across
        zt_m = dict(zt)
        zt_m["grad_energy"] = zt["grad_energy"][:, perm]
        zt_m["grad_dot"] = zt["grad_dot"][:, perm]
        zt_m["energy"] = zt["energy"][:, perm]
        st, rt = c8_scalars(zt_m, cc)
        env.append(np.abs(sc - st))
    env = np.stack(env)                                       # (8, S)
    env_max = env.max(0)
    rows = []
    for b in bases:
        for c in range(3):
            zo = load(GT / "c8" / f"{b}_c{c}_orig")
            zm = load(GT / "c8" / f"{b}_c{c}_mirr")
            co = arms[f"{b}_c{c}_orig"]["targets"]
            cm = arms[f"{b}_c{c}_mirr"]["targets"]
            so, ro = c8_scalars(zo, co)
            sm, rm = c8_scalars(zm, cm)
            go, gm = grok_epoch(zo), grok_epoch(zm)
            d = np.abs(so - sm)
            ep = zo["epochs"]
            exc = ep[np.argmax(d > env_max)] if (d > env_max).any() else None
            # phase profile: mean divergence in quarters of the pair's own
            # (min) grok time, plus the post-grok band
            tg = max(min(go, gm), 1) if go > 0 and gm > 0 else None
            phase = None
            if tg:
                q = np.clip((ep / tg * 4).astype(int), 0, 4)
                phase = [float(d[q == i].mean()) if (q == i).any() else None
                         for i in range(5)]
            # time-warp test: resample each run's share curve on its OWN
            # phase axis u = t/t_gen. If pairs collapse under the warp,
            # identity acts as a clock-speed multiplier, not a shape change.
            warp = None
            if go > 0 and gm > 0:
                u = np.linspace(0.05, 1.0, 40)
                wo = np.interp(u * go, ep, so)
                wm = np.interp(u * gm, ep, sm)
                warp = float(np.median(np.abs(wo - wm)))
            rows.append(dict(base=b, content=c, tgen=(go, gm),
                             d_warp=warp,
                             dt_rel=abs(go - gm) / max(min(go, gm), 1)
                             if go > 0 and gm > 0 else None,
                             first_exceed=None if exc is None else int(exc),
                             phase_prof=phase,
                             d_med_pregrok=float(np.median(
                                 d[ep < max(min(go, gm), 1)])),
                             env_med_pregrok=float(np.median(
                                 env_max[ep < max(min(go, gm), 1)]))))
    ex = [r["first_exceed"] for r in rows if r["first_exceed"] is not None]
    print(f"c8: {len(rows)} install pairs; {len(ex)}/{len(rows)} exceed the "
          f"twin envelope, median first-exceed epoch "
          f"{np.median(ex) if ex else float('nan'):.0f}")
    dm = [r["d_med_pregrok"] for r in rows]
    em = [r["env_med_pregrok"] for r in rows]
    print(f"c8: median pre-grok |Δ on-comm grad share| pair {np.median(dm):.4f}"
          f" vs twin envelope {np.median(em):.4f} "
          f"(ratio {np.median(dm) / max(np.median(em), 1e-12):.1f}x)")
    dts = [r["dt_rel"] for r in rows if r["dt_rel"] is not None]
    print(f"c8: paired |dt_gen| rel median {np.median(dts):.3f} (n={len(dts)})")
    dw = [r["d_warp"] for r in rows if r["d_warp"] is not None]
    du = [r["d_med_pregrok"] for r in rows if r["d_warp"] is not None]
    print(f"c8: time-warp test — median |Δ share| warped {np.median(dw):.4f} "
          f"vs unwarped {np.median(du):.4f} "
          f"(ratio {np.median(dw) / max(np.median(du), 1e-12):.2f})")
    profs = np.array([r["phase_prof"] for r in rows
                      if r["phase_prof"] and None not in r["phase_prof"]],
                     dtype=float)
    if len(profs):
        print(f"c8: mean |Δ share| by phase quarter of t_gen "
              f"{profs.mean(0).round(4).tolist()} (q1..q4, post)")
    return rows, dict(env_max=env_max.tolist())


def main():
    out = {}
    print("=== d11 eviction ===")
    out["d11"] = d11()
    print("=== c8 mirror decoherence ===")
    rows, envd = c8()
    out["c8"] = rows
    out["c8_env"] = envd
    p = ROOT / "notes" / "gradtap_summary.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
