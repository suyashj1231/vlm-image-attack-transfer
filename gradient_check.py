"""Verification step #1: prove the attack path is differentiable w.r.t. pixels.

Run in the PyTorch env:
    source ~/venvs/vlm-torch/bin/activate
    python gradient_check.py

It (a) prints the processor's image params, (b) checks our torch patchify
matches the HF processor numerically, and (c) confirms a non-zero gradient
reaches the raw pixels through one teacher-forced forward/backward pass.
If all three pass, the optimiser in attack/optimizer.py is sound.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from attack.optimizer import AttackConfig, QwenVLAttack


def main():
    cfg = AttackConfig(steps=1)
    print(f"loading {cfg.model_id} on {cfg.device} ({cfg.dtype}) ...")
    atk = QwenVLAttack(cfg)
    ip = atk.processor.image_processor
    print("image params:",
          f"patch={atk.patch_size} temporal={atk.temporal_patch_size} "
          f"merge={atk.merge_size} factor={atk.factor}")
    print("mean/std:", ip.image_mean, ip.image_std)

    print("\n[1/2] verifying torch patchify vs HF processor ...")
    diff = atk.verify_patchify()
    print(f"      OK  max|diff| = {diff:.3e}")

    print("\n[2/2] gradient check through one forward/backward ...")
    arr = (np.random.rand(cfg.canvas, cfg.canvas, 3) * 255).astype("uint8")
    img01 = torch.from_numpy(arr).float().permute(2, 0, 1).to(atk.device) / 255.0
    img01.requires_grad_(True)
    loss = atk.loss_for(img01, "What is in this image?", "HIJACKED-BY-TEAM-X")
    loss.backward()
    g = img01.grad
    gnorm = g.abs().sum().item()
    nonzero = (g.abs() > 0).float().mean().item()
    print(f"      loss={loss.item():.4f}  grad L1={gnorm:.4f}  "
          f"frac-nonzero-pixels={nonzero:.3f}")
    assert gnorm > 0, "ZERO gradient -- pixels are not in the graph!"
    print("\nALL CHECKS PASSED -- differentiable path is sound.")


if __name__ == "__main__":
    main()
