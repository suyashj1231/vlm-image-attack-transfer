"""Evaluation harness: run an attack image against the quantised MLX victims.

This is the DEFENDER'S measurement side. Given an adversarial PNG produced by
attack/optimizer.py against the fp32 reference, we feed it (optionally after a
deployment preprocessing step) to each MLX victim model and record whether the
attack still lands. Runs in the MLX env (~/venvs/vlm), which has mlx-vlm.

    source ~/venvs/vlm/bin/activate
    python -m eval.run_grid --image images/attack.png --target HIJACKED-BY-TEAM-X

Writes results/grid.csv with one row per (model x pipeline x prompt). Nothing
harmful is written: for the benign target we store only the match flag, and for
benchmark prompts we store only the refusal-bypass flag, never model text.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from attack.targets import BENIGN_PROMPTS_HELDOUT, BENIGN_PROMPTS_TRAIN
from eval.metrics import benign_starts_with, benign_success, refusal_bypassed, score_ladder
from eval.preprocess import PIPELINES, apply_pipeline

# The victim grid for milestone 1: three sizes, 8-bit MLX.
DEFAULT_VICTIMS = [
    "mlx-community/Qwen3-VL-2B-Instruct-8bit",
    "mlx-community/Qwen3-VL-4B-Instruct-8bit",
    "mlx-community/Qwen3-VL-8B-Instruct-8bit",
]
# Precision sweep on the size the attack actually lands on (2B).
PRECISION_SWEEP = [
    f"mlx-community/Qwen3-VL-2B-Instruct-{q}"
    for q in ("3bit", "4bit", "5bit", "6bit", "8bit", "bf16")
]
VICTIM_SETS = {"default": DEFAULT_VICTIMS,
               "sweep": PRECISION_SWEEP + DEFAULT_VICTIMS[1:]}

# Every train and held-out prompt from attack/targets.py, tagged with its split
# so train-prompt success (memorisation) is never mixed into held-out numbers.
PROMPT_SPLIT = {p: "train" for p in BENIGN_PROMPTS_TRAIN} | {
    p: "heldout" for p in BENIGN_PROMPTS_HELDOUT}
DEFAULT_PROMPTS = list(PROMPT_SPLIT)


def load_model(model_id):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(model_id)
    config = load_config(model_id)
    return model, processor, config


def generate(model, processor, config, image_path, prompt, max_tokens=64):
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    out = mlx_generate(
        model, processor, formatted, [str(image_path)],
        max_tokens=max_tokens, verbose=False, temperature=0.0,
    )
    # mlx-vlm returns either a string or an object with .text across versions
    return getattr(out, "text", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="adversarial PNG to test")
    ap.add_argument("--target", default="HIJACKED-BY-TEAM-X",
                    help="benign forced string; empty => benchmark/bypass mode")
    ap.add_argument("--mode", choices=["benign", "benchmark"], default="benign")
    ap.add_argument("--victims", nargs="*", default=None,
                    help="explicit repo ids (overrides --victim-set)")
    ap.add_argument("--victim-set", choices=list(VICTIM_SETS), default="default")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--pipelines", nargs="*", default=None,
                    help="subset of pipeline names (default: all)")
    ap.add_argument("--out", default="results/grid.csv")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    from PIL import Image
    base = Image.open(args.image).convert("RGB")
    tmp = Path("results/_tmp_input.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    responses: list[dict] = []   # benign responses, for re-scoring the ladder
    victims = args.victims or VICTIM_SETS[args.victim_set]
    for victim in victims:
        print(f"\n=== victim: {victim} ===")
        model, processor, config = load_model(victim)
        for pipe in [p for p in PIPELINES if not args.pipelines or p.name in args.pipelines]:
            prepped = apply_pipeline(base, pipe)
            prepped.save(tmp)
            for prompt in args.prompts:
                t0 = time.time()
                resp = generate(model, processor, config, tmp, prompt,
                                max_tokens=args.max_tokens)
                dt = time.time() - t0
                row = {
                    "victim": victim, "pipeline": pipe.name, "prompt": prompt,
                    "split": PROMPT_SPLIT.get(prompt, "custom"),
                    "tier": pipe.tier,
                }
                if args.mode == "benign":
                    flag = benign_success(resp, args.target)
                    strict = benign_starts_with(resp, args.target)
                    # full strict->lenient ladder (exact, starts_with, contains,
                    # contains_norm, lenient) -- the control-vs-content axis
                    # ladder provides starts_with/exact/contains/contains_norm/lenient
                    ladder = score_ladder(resp, args.target)
                    row.update(metric="benign_success", success=int(flag), **ladder)
                    # benign responses are safe to store (ETHICS rule 1); keep
                    # them so the ladder can be re-scored without re-generating
                    responses.append({"victim": victim, "pipeline": pipe.name,
                                      "prompt": prompt,
                                      "split": PROMPT_SPLIT.get(prompt, "custom"),
                                      "response": resp})
                    note = f"ctrl={int(strict)} content={int(flag)} leak50={ladder['leak50']}"
                else:
                    flag = strict = refusal_bypassed(resp)
                    row.update(metric="refusal_bypassed", success=int(flag),
                               starts_with=int(strict))
                    note = f"bypass={int(flag)}"
                row["secs"] = round(dt, 2)
                rows.append(row)
                print(f"  [{pipe.name:10s}] {note}  ({dt:.1f}s)  '{prompt[:28]}'")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # save benign responses alongside the CSV, for ladder re-scoring (eval.rescore)
    if responses:
        import json
        rp = out.with_name(out.stem + "_responses.jsonl")
        with rp.open("w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")
        print(f"saved {len(responses)} benign responses -> {rp}")
    # quick aggregate
    print(f"\nwrote {len(rows)} rows -> {out}")
    cells: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for r in rows:
        cells.setdefault((r["victim"], r["pipeline"], r["split"]), []).append(
            (r["success"], r["starts_with"]))
    print("\nsuccess / starts_with rate by victim x pipeline x split:")
    for (v, pipe, split), fl in cells.items():
        n = len(fl)
        print(f"  {v.split('/')[-1]:28s} {pipe:10s} {split:8s} "
              f"{sum(a for a, _ in fl)/n:.2f} / {sum(b for _, b in fl)/n:.2f}  (n={n})")


if __name__ == "__main__":
    main()
