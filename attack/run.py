"""CLI: optimise one benign attack image against the PyTorch reference model(s).

    source ~/venvs/vlm-torch/bin/activate
    python -m attack.run --out images/attack.png --steps 300

Transferability options (see RESULTS.md / optimizer.py). Examples:

    # white-box ceiling on a bigger model (Experiment 0): does it even work
    # when we differentiate through 8B directly?
    python -m attack.run --model Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 \
        --out images/wb8b.png --steps 1000

    # ensemble of surrogates (2B + 4B) -- the main transfer lever (B1)
    python -m attack.run --ensemble Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct \
        --dtype bfloat16 --out images/ens.png --steps 1000 --eot --eot-samples 2

    # momentum + translation-invariance + input-diversity (B2/B3/B4)
    python -m attack.run --momentum 1.0 --ti 5 --di 0.5 --out images/mitidi.png

    # feature-space loss + natural base + TV penalty (B5 / C)
    python -m attack.run --loss embed --base photo.jpg --tv 0.02 --out images/embed.png

Then evaluate robustness with eval/run_grid.py (MLX env).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from attack.optimizer import AttackConfig, EOTConfig, QwenVLAttack, run_pgd
from attack.targets import benign_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="images/attack.png")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--ensemble", default=None,
                    help="comma-separated model ids to attack jointly (overrides --model)")
    ap.add_argument("--target", default="HIJACKED-BY-TEAM-X")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=16 / 255,
                    help="L-inf budget; pass a negative value for unconstrained")
    ap.add_argument("--lr", type=float, default=1 / 255, help="PGD / Adam step size")
    ap.add_argument("--canvas", type=int, default=448)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="reference-model weight dtype; perturbation stays fp32")
    ap.add_argument("--eot", action="store_true",
                    help="optimise over random resize/JPEG (see EOTConfig)")
    ap.add_argument("--eot-preset", default="base", choices=["base", "wide"],
                    help="transform distribution for --eot (see EOTConfig.preset)")
    ap.add_argument("--eot-samples", type=int, default=1,
                    help="transform draws averaged per step (cost scales linearly)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (drives the EOT/DI transform draws)")

    # -- transferability knobs (B1-B8, C) -------------------------------------
    ap.add_argument("--loss", default="lm", choices=["lm", "embed"],
                    help="lm = teacher-forced cross-entropy; embed = feature-space (B5)")
    ap.add_argument("--optimizer", default="sign", choices=["sign", "adam", "apgd"],
                    help="sign = PGD (default); adam / apgd are stronger optimisers (B6)")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="MI-FGSM decay on the L1-normalised gradient (B2); e.g. 1.0")
    ap.add_argument("--ti", type=int, default=0, metavar="K",
                    help="TI-FGSM Gaussian kernel side, odd, 0=off (B3); e.g. 5")
    ap.add_argument("--ti-sigma", type=float, default=1.0)
    ap.add_argument("--di", type=float, default=0.0, metavar="P",
                    help="DI-FGSM apply probability, 0=off (B4); e.g. 0.5")
    ap.add_argument("--di-frac", type=float, default=0.15,
                    help="max relative shrink for DI")
    ap.add_argument("--tv", type=float, default=0.0,
                    help="total-variation penalty weight, 0=off (C); e.g. 0.02")
    ap.add_argument("--patch-frac", type=float, default=0.0, metavar="F",
                    help="adversarial PATCH: confine perturbation to a centred "
                         "square of fraction F of the canvas (unbounded within; "
                         "ignores --epsilon). Image-Hijacks robustness control.")
    ap.add_argument("--prompt-batch", action="store_true",
                    help="average loss over ALL train prompts each step (B8)")

    ap.add_argument("--check", default=None, metavar="IMAGE",
                    help="skip optimisation; only run the white-box held-out check on IMAGE")
    ap.add_argument("--base", default=None,
                    help="optional base image to perturb; default = grey canvas (C: a "
                         "natural photo keeps it a perturbation, not a painted glyph)")
    args = ap.parse_args()

    model_ids = ([m.strip() for m in args.ensemble.split(",") if m.strip()]
                 if args.ensemble else [args.model])

    def make_cfg(model_id: str) -> AttackConfig:
        return AttackConfig(
            model_id=model_id, steps=args.steps, canvas=args.canvas, lr=args.lr,
            epsilon=None if args.epsilon < 0 else args.epsilon, dtype=args.dtype,
            eot=EOTConfig.preset(args.eot_preset) if args.eot else None,
            eot_samples=args.eot_samples, seed=args.seed,
            loss_mode=args.loss, optimizer=args.optimizer, momentum=args.momentum,
            ti_kernel=args.ti, ti_sigma=args.ti_sigma, di_prob=args.di,
            di_frac=args.di_frac, tv_weight=args.tv, prompt_batch=args.prompt_batch,
            patch_frac=args.patch_frac,
        )

    members = [QwenVLAttack(make_cfg(mid)) for mid in model_ids]
    lead = members[0]
    cfg = lead.cfg
    for m in members:
        print(f"[{m.cfg.model_id}] verifying patchify ...", m.verify_patchify())

    if args.base:
        base = Image.open(args.base).convert("RGB")
    else:
        base = Image.fromarray(
            np.full((cfg.canvas, cfg.canvas, 3), 128, dtype="uint8"))

    tgt = benign_target(args.target)
    if args.check:
        img = Image.open(args.check).convert("RGB")
    else:
        img = run_pgd(members, base, tgt.prompts_train, tgt.target_text, cfg)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"saved -> {out}")
    # white-box check against every surrogate in the ensemble
    for m in members:
        print(f"\n== white-box held-out check: {m.cfg.model_id} ==")
        white_box_check(m, img, tgt)


def white_box_check(atk, img, tgt):
    """Does the reference itself emit the target on held-out prompts?"""
    import torch
    hits = 0
    img01 = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).to(atk.device) / 255.0
    for p in tgt.prompts_heldout:
        inputs, tgt_start, _ = atk._assemble(img01, p, tgt.target_text)
        # _assemble appends the target for teacher forcing; generate from the
        # prompt alone, or the model just continues an already-given target
        for k in ("input_ids", "mm_token_type_ids"):
            inputs[k] = inputs[k][:, :tgt_start]
        with torch.no_grad():
            gen = atk.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = atk.processor.tokenizer.decode(
            gen[0][tgt_start:], skip_special_tokens=True)
        ok = tgt.target_text.lower() in text.lower()
        hits += ok
        print(f"  heldout '{p[:28]}' -> {'HIT' if ok else 'miss'}: {text[:80]!r}")
    print(f"white-box held-out success: {hits}/{len(tgt.prompts_heldout)}")


if __name__ == "__main__":
    main()
