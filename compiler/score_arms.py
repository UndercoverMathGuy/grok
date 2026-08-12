"""Score trained compiler arms against their pre-registered predictions.

Per arm: final committee (unified detector) vs predicted target set —
exact match, Jaccard, per-target adoption (final |coeff| and the 10x-
background criterion), unpredicted recruits with their compiled-T_k rank
(the P-C4 recruit prediction), grok epoch, final test CE.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler"))

from core import committee_from_coeffs, load_ckpt, tk_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "compiler" / "arms" /
                                              "phaseA_manifest.json"))
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    arms = json.loads(Path(args.manifest).read_text())

    rows, exact, total = [], 0, 0
    for a in arms:
        rd = Path(a["run_dir"])
        if args.out_root:
            rd = Path(args.out_root) / rd.relative_to(ROOT / "runs_compiler")
        z = rd / "spectra.npz"
        if not z.exists():
            print(f"pending: {a['tag']}")
            continue
        z = np.load(z)
        p = a["config"]["p"]
        coeffs = z["coeffs"][-1]
        comm = committee_from_coeffs(coeffs)
        acc = float(z["test_acc"][-1])
        gi = int(np.argmax(z["test_acc"] >= 0.99))
        grok = int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1
        S = a["predicted_committee"]
        row = dict(tag=a["tag"], committee=comm, predicted=S, grok=grok,
                   test_acc=acc)
        if S is not None:
            total += 1
            row["exact"] = comm == sorted(S)
            exact += row["exact"]
            inter = set(comm) & set(S)
            row["jaccard"] = len(inter) / len(set(comm) | set(S))
            amps = np.abs(coeffs)
            bg = np.median(amps)
            row["target_amps"] = {t: float(amps[t - 1]) for t in S}
            row["adopted"] = {t: bool(amps[t - 1] >= 10 * bg) for t in S}
            recruits = [c for c in comm if c not in S]
            if recruits:
                tk = tk_profile(load_ckpt(a["ckpt"]), p)
                order = list(np.argsort(tk)[::-1] + 1)
                row["recruits"] = {r: int(order.index(r) + 1) for r in recruits}
        rows.append(row)
        mark = ("EXACT" if row.get("exact") else
                "MISS " if S is not None else "ctrl ")
        print(f"{mark} {a['tag']:<32} final {comm}  pred {S}  "
              f"grok {grok}  acc {acc:.3f}"
              + (f"  recruits(tk-rank) {row.get('recruits')}"
                 if row.get("recruits") else ""))

    if total:
        print(f"\n=== exact-set match {exact}/{total} ===")
    by_cell = {}
    arm_by_tag = {a["tag"]: a for a in arms}
    for row in rows:
        a = arm_by_tag[row["tag"]]
        if "cell" in a and "exact" in row:
            key = (a["cell"]["k"], a["cell"]["s"])
            by_cell.setdefault(key, []).append(row["exact"])
    if by_cell:
        print("\n=== exact rate by (K, s) cell ===")
        for (k, s) in sorted(by_cell):
            v = by_cell[(k, s)]
            print(f"  K={k} s={s:<5g} {sum(v)}/{len(v)}")
    out = Path(args.manifest).with_name(
        Path(args.manifest).stem.replace("_manifest", "_scores") + ".json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"scores -> {out}")


if __name__ == "__main__":
    main()
