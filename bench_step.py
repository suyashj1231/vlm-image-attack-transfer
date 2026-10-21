"""Time PGD steps for the reference model: python bench_step.py [float32|bfloat16]"""
import sys
import time

import numpy as np
import torch

from attack.optimizer import AttackConfig, QwenVLAttack
from attack.targets import benign_target

dtype = sys.argv[1] if len(sys.argv) > 1 else "float32"
cfg = AttackConfig(steps=6, dtype=dtype)
t0 = time.time()
atk = QwenVLAttack(cfg)
print(f"[{dtype}] load {time.time() - t0:.1f}s")
tgt = benign_target("HIJACKED-BY-TEAM-X")
img01 = torch.full((3, cfg.canvas, cfg.canvas), 0.5, device=atk.device, requires_grad=True)
times = []
for i in range(cfg.steps):
    torch.mps.synchronize()
    t = time.time()
    loss = atk.loss_for(img01, tgt.prompts_train[i % len(tgt.prompts_train)], tgt.target_text)
    loss.backward()
    img01.grad = None
    torch.mps.synchronize()
    times.append(time.time() - t)
    print(f"  step {i}: {times[-1]:.2f}s  loss={loss.item():.4f}")
print(f"[{dtype}] median step {np.median(times[1:]):.2f}s  "
      f"driver mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GB")
