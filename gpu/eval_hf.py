"""CUDA / transformers victim grid -- the Kaggle twin of eval/run_grid.py.

eval/run_grid.py runs MLX victims on the laptop. This runs full-precision
Hugging Face victims on a CUDA box (Kaggle 2x T4: fp16 only, no bf16) and
writes the SAME columns, so CSVs from either runtime go straight into
eval/summarize.py and eval/summarize_transfer.py.

Benign forced-string mode only (ETHICS.md rule 1).

    PYTHONPATH=. python -m gpu.eval_hf --image images/mitidi.png \
        --victims Qwen/Qwen3-VL-2B-Instruct --out results/grid6_mitidi_hf.csv

Rows are appended to the CSV as they are produced, and a re-run with the same
--out skips (victim, pipeline, prompt) cells that are already there, so a job
killed by the 12-hour session limit can be resumed.

Thinking models: any <think>...</think> block is split off. The ladder is
scored on the final answer; `think_contains` records whether the target
appeared only in the reasoning trace.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from attack.targets import BENIGN_PROMPTS_HELDOUT, BENIGN_PROMPTS_TRAIN, BENIGN_TARGET
from eval.metrics import benign_success, score_ladder
from eval.preprocess import PIPELINES, apply_pipeline

PROMPT_SPLIT = {p: "train" for p in BENIGN_PROMPTS_TRAIN} | {
    p: "heldout" for p in BENIGN_PROMPTS_HELDOUT}
THINK_RE = re.compile(r"<think>(.*?)(</think>|$)", re.S)


def split_thinking(text: str) -> tuple[str, str]:
    """Return (reasoning, final_answer). Handles a missing opening tag, which
    Qwen3 thinking templates sometimes put in the prompt instead of the output."""
    if "</think>" in text and "<think>" not in text:
        think, _, answer = text.partition("</think>")
        return think.strip(), answer.strip()
    m = THINK_RE.search(text)
    if not m:
        return "", text.strip()
    return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()


def load_victim(model_id: str, dtype: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch_dtype = {"float16": torch.float16, "float32": torch.float32,
                   "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate(model, processor, image, prompt: str, max_new_tokens: int) -> str:
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new, skip_special_tokens=True)


def done_cells(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {(r["victim"], r["pipeline"], r["prompt"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--target", default=BENIGN_TARGET)
    ap.add_argument("--victims", nargs="+", required=True, help="HF repo ids")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPT_SPLIT))
    ap.add_argument("--pipelines", nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="raise (e.g. 1024) for thinking models")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="results/grid_hf.csv")
    args = ap.parse_args()

    from PIL import Image
    base = Image.open(args.image).convert("RGB")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    resp_path = out.with_name(out.stem + "_responses.jsonl")
    skip = done_cells(out)
    fields = ["victim", "pipeline", "prompt", "split", "tier", "metric", "success",
              "exact", "starts_with", "contains", "contains_norm", "leak50",
              "think_contains", "secs"]
    new_file = not out.exists()
    fcsv = out.open("a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        writer.writeheader()
    fresp = resp_path.open("a")
    pipes = [p for p in PIPELINES if not args.pipelines or p.name in args.pipelines]

    for victim in args.victims:
        todo = [(p, q) for p in pipes for q in args.prompts
                if (victim, p.name, q) not in skip]
        if not todo:
            print(f"=== {victim}: all cells done, skipping")
            continue
        print(f"\n=== victim: {victim} ({len(todo)} cells) ===", flush=True)
        t_load = time.time()
        model, processor = load_victim(victim, args.dtype)
        print(f"loaded in {time.time() - t_load:.0f}s", flush=True)
        for pipe, prompt in todo:
            img = apply_pipeline(base, pipe)
            t0 = time.time()
            raw = generate(model, processor, img, prompt, args.max_tokens)
            dt = time.time() - t0
            think, answer = split_thinking(raw)
            ladder = score_ladder(answer, args.target)
            row = {"victim": victim, "pipeline": pipe.name, "prompt": prompt,
                   "split": PROMPT_SPLIT.get(prompt, "custom"), "tier": pipe.tier,
                   "metric": "benign_success",
                   "success": int(benign_success(answer, args.target)),
                   **{k: ladder[k] for k in ("exact", "starts_with", "contains",
                                             "contains_norm", "leak50")},
                   "think_contains": int(benign_success(think, args.target)),
                   "secs": round(dt, 2)}
            writer.writerow(row)
            fcsv.flush()
            # benign responses are safe to keep (ETHICS rule 1) -> offline rescoring
            fresp.write(json.dumps({"victim": victim, "pipeline": pipe.name,
                                    "prompt": prompt, "split": row["split"],
                                    "response": answer, "reasoning": think}) + "\n")
            fresp.flush()
            print(f"  [{pipe.name:10s}] ctrl={row['starts_with']} "
                  f"content={row['contains']} ({dt:.1f}s) '{prompt[:28]}'", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    fcsv.close()
    fresp.close()
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
