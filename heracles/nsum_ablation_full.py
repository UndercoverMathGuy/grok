"""Full per-frequency ablation profile (every frequency, both/in/out, min- and mean-margin) for every nsum net."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsum_ablation import Net, ROOT
out = ROOT / "nsum_ablation_full.json"
done = {r["name"]: r for r in json.loads(out.read_text())} if out.exists() else {}
runs = sorted((ROOT / "runs" / "NS").glob("*/seed*"), key=lambda r: (int(r.parent.name[1]), r.parent.name))
for rd in runs:
    name = f"{rd.parent.name}/{rd.name}"
    if name in done or not (rd / "final.safetensors").exists(): continue
    try:
        net = Net(rd); b = net.stats(net.E, net.W); prof = {}
        for k in range(1, net.p // 2 + 1):
            prof[k] = {s: net.stats(*net.cut([k], sd)) for s, sd in (("both", ("in", "out")), ("in", ("in",)), ("out", ("out",)))}
        mt = net.metrics
        done[name] = dict(name=name, p=net.p, n=net.n, act=net.act, base=b, prof=prof,
                          **{k: mt.get(k) for k in ("N_05", "K_5pct", "K_2pct", "clean_frac", "q", "penalty", "lam", "seed", "margin", "acc")})
        print(name, b, flush=True)
    except Exception as ex:
        print("FAIL", name, repr(ex), flush=True); continue
    out.write_text(json.dumps(list(done.values())))
