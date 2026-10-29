# VLM image attacks under real deployment conditions

Two well-known papers on attacking vision-language models (VLMs) seem to disagree. One shows you can build a single doctored image that hijacks many models. Another runs large experiments and finds those images almost never work on a model they were not built against. Both are from top venues.

This project tests who is right, and the short answer is that both are, because they quietly measure two different things. The distinction that settles it is control versus content:

- *Control* means the image actually seized the model: it forced the exact target string at the start of the answer, which is a real hijack.
- *Content* means the string only appears because it was painted into the image as pixels, and the model read it off the picture. The model was never steered; it described what it saw.

Score an attack strictly (did it take control?) and transfer is about 0%. Score it loosely (did the string show up as readable pixels?) and it is 40% to 100%. Same images, both results true.

The only thing any attack is ever asked to force is the benign placeholder `HIJACKED-BY-TEAM-X`. Nothing harmful is generated. See `ETHICS.md`.

## What we found

Eleven experiments (E0 through E10), run across 3 model scales (2B, 4B, 8B), 6 quantization levels (3, 4, 5, 6, 8-bit and bf16), and 11 deployment preprocessing pipelines (resize, JPEG, WebP, blur, crop). Held-out prompts throughout.

- Control does not cross architectures. The strongest attack we could build (an ensemble of 2B and 4B surrogates with APGD, MI/TI/DI boosters, EOT, and 1000 steps) hit the target on 7 of 8 white-box checks, so the optimizer genuinely works. On the never-seen 8B it still reached 0% control and roughly 0% verbatim content.
- Within a known architecture, attacks survive deployment. The same strong image kept 82% control and 84% content on 2B, and 61% / 73% on 4B, through the quantization and preprocessing gap.
- Content and control come apart. On the 4B victim, one attack reproduced the target string (content) up to 70% of the time while taking control 0% of the time. The model was reading the text in the image, not following it.
- Quantization is not a defense. Control stays high from 4-bit up to bf16. The only level that suppresses the attack is 3-bit, and it does so by breaking the model's usefulness entirely. The cheap real defense is aggressive downscaling (resize to 256 px), not compression.
- A loose metric overstated the result. An early draft reported 45% content on the 8B using a loose leakage metric (`leak50`). That metric fires on the single word "hijacked," which is 53% of the target and which models use to describe any glitchy screen. The honest number by verbatim matching is near 0%. The correction is kept in the log on purpose.

Full detail, every run and every number, is in `RESEARCH_LOG.md`. Figures are in `figures/`.

## How it works

The study has two sides that run in separate Python environments, which is also what keeps the surrogate and victim models cleanly apart.

| venv | Python | holds | used by |
|---|---|---|---|
| `~/venvs/vlm-torch` | 3.12 | torch, transformers | `attack/` (crafting on MPS) |
| `~/venvs/vlm` | 3.15 | mlx, mlx-vlm | `eval/` (victim measurement) |

The attacker side optimizes pixels with PGD against one or more PyTorch surrogate models and saves a PNG. The victim side feeds that PNG to quantized MLX models, optionally after a preprocessing step, and scores the answer on a strict-to-lenient metric ladder (exact, starts_with = control, contains = content, contains_norm, leak50).

## Reproduce

```bash
# 1. craft one benign attack image (attacker env)
source ~/venvs/vlm-torch/bin/activate
python -m attack.run --out images/attack.png

# the strongest transfer attack from E9:
python -m attack.run \
  --ensemble Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct \
  --optimizer apgd --momentum 1.0 --ti 5 --di 0.5 \
  --prompt-batch --eot --eot-samples 1 --steps 1000 \
  --out images/stress_ens_apgd_s0.png

# 2. measure it across the MLX victims (victim env)
source ~/venvs/vlm/bin/activate
python -m eval.run_grid --image images/attack.png --target HIJACKED-BY-TEAM-X

# 3. rebuild the figures
python repro/make_figures.py
```

The overnight experiment chains live in `repro/` (`stress_chain.sh`, `agg_stress.py`).

## Repository layout

```
attack/optimizer.py   PGD, EOT, ensemble, MI/TI/DI, APGD, TV, patch mode
attack/ultrabreak.py  faithful UltraBreak (semantic attention loss)
attack/typo.py        explicit typographic content attack
attack/run.py         one CLI for every optimizer.py attack
eval/run_grid.py      victim grid, records control and content metrics
eval/preprocess.py    the 11 deployment pipelines
eval/metrics.py       the strict-to-lenient metric ladder
eval/rescore.py       re-score saved responses offline, no GPU
repro/                experiment chains, figures, aggregation
figures/              f1 through f5, the paper figure set
RESEARCH_LOG.md       canonical record of every experiment
ETHICS.md             scope, benign-target rule, disclosure
```

## Ethics

The target is always benign. Only locally downloaded models are used. For any harmful-prompt benchmark, the repository stores aggregate bypass rates only, never model text, and the prompt files themselves are never committed (they are download-local). The findings are written for defenders. Details are in `ETHICS.md`.

## Status

Complete through E10. The open threads, mostly a semantic content metric to replace `leak50`, a non-Qwen victim to widen the cross-architecture claim, and a heavy-EOT patch test, are tracked at the end of `RESEARCH_LOG.md`.
