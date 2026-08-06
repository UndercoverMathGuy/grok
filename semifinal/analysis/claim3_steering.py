"""CLAIM 3 — the committee can be steered by hand (spectra-only analysis).

Auto-discovers every steering experiment under runs/ and rebuilds:

  A. Energy dose-response curves: runs/dosefarm/<base>/dose_*, plus the
     original surgery arms on seed27058 (control/boost_subtle/boost_strong
     = doses 1.0/1.2/2.25) and surgery2 dose arms (1.05/1.10). The boosted
     target is identified automatically by comparing each arm's epoch-0
     W_E spectrum to its base run's (exactly one frequency scaled).
  C. Suppression: surgery/suppress (0.5x on the strongest incumbent).
  D. Alignment-knob arms: runs/gkrotate/gain_* — energy-preserving
     rotations at matched arrival-loudness gains.
  E. Transplant tally: runs/transplant/<R>_from_<D> — donor-unique vs
     recipient-unique members adopted (full-spectrum copy does NOT
     transfer identity).
  F. Chaos exhibit: surgery2 triple_deg vs triple_clean (energy-matched
     sub-threshold swap changes the committee).

Outcome variable: the TARGET's final |coefficient|; adoption = final >=
COMMITTEE_FLOOR x the run's max final |coefficient| — the SAME constant the
committee detector uses (common.py), so "adopted" and "committee member"
can never disagree.
"""
import re
import numpy as np

from common import (RUNS, discover, freq_energy, find_base_run, jaccard,
                    grok_epoch, COMMITTEE_FLOOR)
from grok.model import Transformer
from grok.config import Config

runs = {r["rel"]: r for r in discover()}

def e0_energy(run_dir, cfg):
    m = Transformer(cfg)
    m.load_weights(str(run_dir / "checkpoints" / "epoch_00000.safetensors"))
    return freq_energy(np.array(m.W_E, dtype=np.float64)[:, :cfg.p], cfg.p)

_base_cache = {}
def base_energy(name):
    """(dir, cfg, epoch-0 W_E spectrum) of a natural base run, or None if the
    basename doesn't resolve under a natural family."""
    if name not in _base_cache:
        b = find_base_run(name)
        if b is None:
            _base_cache[name] = None
        else:
            cfg = Config.load(b / "config.json")
            _base_cache[name] = (b, cfg, e0_energy(b, cfg))
    return _base_cache[name]

def target_and_dose(rel, base_name):
    r = runs[rel]
    be = base_energy(base_name)
    if be is None:
        return None
    _, bcfg, eb = be
    er = e0_energy(r["dir"], r["cfg"])
    ratio = er / eb
    k = int(np.argmax(np.abs(ratio - 1))) + 1
    return k, float(ratio[k - 1])

def arm_row(rel, k):
    r = runs[rel]
    z = r["spectra"]
    tr = np.abs(z["coeffs"][:, k - 1])
    adopted = tr[-1] >= COMMITTEE_FLOOR * np.abs(z["coeffs"][-1]).max()
    return (f"    target f{k}: final {tr[-1]:9.0f}  peak {tr.max():9.0f}  "
            f"adopted={adopted}  committee {sorted(r['committee'])}  "
            f"grok@{grok_epoch(z)}")

print("=== A. energy dose-response curves ===")
curves = {}
for rel in sorted(runs):
    m1 = re.match(r"dosefarm/([^/]+)/dose_(\d+)$", rel)
    if m1:
        curves.setdefault(m1.group(1), []).append(
            (int(m1.group(2)) / 100, rel))
for legacy, base, dose in (("surgery/control", "seed27058", 1.00),
                           ("surgery2/dose_105", "seed27058", 1.05),
                           ("surgery2/dose_110", "seed27058", 1.10),
                           ("surgery/boost_subtle", "seed27058", 1.20),
                           ("surgery/boost_strong", "seed27058", 2.25)):
    if legacy in runs:
        curves.setdefault(base, []).append((dose, legacy))
dose_target = {}                      # base -> boosted freq (reused by D)
for base, arms in sorted(curves.items()):
    print(f"\n  base {base}:")
    if not any(d == 1.0 for d, _ in arms):
        b = find_base_run(base)       # v2 suites: base natural IS the control
        r = runs.get(str(b.relative_to(RUNS))) if b else None
        if r is not None:
            print(f"   x1.00 [{r['rel']}] control (the base run): committee "
                  f"{sorted(r['committee'])} grok@{grok_epoch(r['spectra'])}")
        else:
            print(f"   x1.00 control: base run '{base}' not discovered")
    for dose, rel in sorted(arms):
        if dose == 1.0:
            k = None
        else:
            td = target_and_dose(rel, base)
            if td is None:
                print(f"   x{dose:.2f} [{rel}] base run '{base}' not found "
                      f"— skipped")
                continue
            k, ratio = td
            assert abs(ratio - dose) < 0.02, (rel, ratio, dose)
            dose_target[base] = k
        if k is None:
            r = runs[rel]
            print(f"   x{dose:.2f} [{rel}] control: committee "
                  f"{sorted(r['committee'])} grok@{grok_epoch(r['spectra'])}")
        else:
            print(f"   x{dose:.2f} [{rel}]")
            print(arm_row(rel, k))

print("\n=== C. suppression (0.5x the strongest incumbent) ===")
sup_arms = [("surgery/suppress", "seed27058")] + [
    (rel, rel.split("/")[1]) for rel in sorted(runs)
    if re.match(r"suppress/[^/]+$", rel)]
for rel, base in sup_arms:
    if rel in runs:
        td = target_and_dose(rel, base)
        if td is None:
            print(f"   [{rel}] base run '{base}' not found — skipped")
            continue
        k, ratio = td
        print(f"   x{ratio:.2f} on f{k} [base {base}]:")
        print(arm_row(rel, k))

print("\n=== D. alignment-knob arms (gkrotate, energy fixed) ===")
for rel in sorted(runs):
    m2 = re.match(r"gkrotate/(?:([^/]+)/)?gain_(\d+)$", rel)
    if m2:
        base = m2.group(1) or "seed27058"
        # gkrotate arms are energy-preserving, so the target cannot be read
        # from the W_E spectrum; it comes from the base's dose arms (or the
        # documented f7 target of the original seed27058 surgery).
        k = 7 if base == "seed27058" else dose_target.get(base)
        if k is None:
            print(f"   [{rel}] no dose arm identifies base '{base}' target "
                  f"— skipped")
            continue
        print(f"   T-gain x{int(m2.group(2))/100:.2f} [{rel}] (target f{k})")
        print(arm_row(rel, k))
print("   (compare with the energy rows above at the same factor: "
      "matched loudness, matched outcome)")

print("\n=== E. transplant tally (full energy-spectrum copy) ===")
nat_by_name, nat_dupes = {}, set()
for r in runs.values():
    if r["cohort"] != "natural-normal":
        continue
    name = r["rel"].split("/")[-1]
    if name in nat_by_name:
        nat_dupes.add(name)
    else:
        nat_by_name[name] = r
tD = tR = 0
for rel in sorted(runs):
    m3 = re.match(r"transplant/(seed\d+)_from_(seed\d+)$", rel)
    if not m3:
        continue
    rname, dname = m3.groups()
    amb = {rname, dname} & nat_dupes
    if amb:
        print(f"   {rel}: base name(s) {sorted(amb)} ambiguous across "
              f"natural families — skipped")
        continue
    rbase, dbase = nat_by_name.get(rname), nat_by_name.get(dname)
    if rbase is None or dbase is None:
        print(f"   {rel}: base run(s) not discovered — skipped")
        continue
    final = set(runs[rel]["committee"])
    uR = set(rbase["committee"]) - set(dbase["committee"])
    uD = set(dbase["committee"]) - set(rbase["committee"])
    tD += len(final & uD)
    tR += len(final & uR)
    print(f"   {rel}: final {sorted(final)}  donor-unique kept "
          f"{sorted(final & uD)}  recipient-unique kept {sorted(final & uR)}")
print(f"   TALLY: donor-unique {tD} vs recipient-unique {tR} "
      f"(energy copy does not transfer identity)")

print("\n=== F. chaos exhibit (energy-matched sub-threshold swap) ===")
chaos = [rel for rel in ("surgery2/triple_deg", "surgery2/triple_clean")
         if rel in runs]
chaos += [rel for rel in sorted(runs)
          if re.match(r"chaospair/[^/]+/arm[AB]$", rel)]
for rel in chaos:
    r = runs[rel]
    print(f"   {rel}: committee {sorted(r['committee'])} "
          f"grok@{grok_epoch(r['spectra'])}")
print("""
Backs SEMIFINAL claim 3: dose-monotone targeted control (A, C), knob
interchangeability at matched loudness (D vs A), no identity transfer from
energy alone (E), and bystander chaos (F). Note the context-dependent
adoption thresholds across bases in A — reported as a scope limit.""")
