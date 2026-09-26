"""GPU step-time benchmark (E0, R3 planning only; no training runs, no run directories).
Times one full-batch GD step (forward + backward of mean CE + penalty) for B lockstep runs per shape,
float32 with TF32 off (core.train.set_precision_flags). Usage: python experiments/E0_tuning/bench_gpu.py OUT.json"""
import json, math, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from core import models as M, train as T

flags = T.set_precision_flags()
dev = torch.device("cuda")
shapes = [("product", 2, 59), ("product", 2, 83), ("product", 2, 113), ("product", 3, 23), ("product", 3, 37),
          ("product", 3, 43), ("product", 3, 59), ("product", 4, 17), ("product", 4, 23), ("product", 4, 31),
          ("power", 2, 59), ("power", 2, 113), ("power", 3, 23), ("power", 3, 37)]
out = []
for model, n, p in shapes:
    nu = n + 1; h = (p - 1) // 2; m = 2 * nu * h
    for B in (1, 4, 8, 16):
        mem_tab = B * p ** (n + 1) * 4
        if mem_tab > 6e9:
            continue
        try:
            U = torch.randn(B, n, m, p, device=dev, requires_grad=True)
            W = torch.randn(B, m, p, device=dev, requires_grad=True)
            y = M.labels(p, n, device=dev)
            q = torch.full((B,), 2.0, device=dev)
            for it in range(3):
                F = M.logits(model, U, W); L = M.ce_margin_form(F, y) + 1e-4 * M.penalty(U, W, q)
                g = torch.autograd.grad(L.sum(), (U, W))
            torch.cuda.synchronize(); t0 = time.time(); k = 10
            for it in range(k):
                F = M.logits(model, U, W); L = M.ce_margin_form(F, y) + 1e-4 * M.penalty(U, W, q)
                g = torch.autograd.grad(L.sum(), (U, W))
            torch.cuda.synchronize(); dt = (time.time() - t0) / k
            r = {"model": model, "n": n, "p": p, "m": m, "B": B, "s_per_step": dt, "s_per_step_per_run": dt / B,
                 "max_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
        except Exception as e:
            r = {"model": model, "n": n, "p": p, "m": m, "B": B, "error": repr(e)[:200]}
        print(json.dumps(r), flush=True); out.append(r)
        del U, W; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
json.dump({"flags": flags, "device": torch.cuda.get_device_name(0), "results": out}, open(sys.argv[1], "w"), indent=1)
