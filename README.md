# Robustness of VLM Image Attacks Under Real Deployment Conditions

CS 205 research project. **Question:** given an adversarial image optimised once
against a reference VLM, how much of its effect survives the things real
deployments actually do — quantisation (3–8 bit), a different runtime (MLX /
Ollama / llama.cpp), and input preprocessing (resize, JPEG re-encode)?

- **Attacker side** (`attack/`) — white-box PGD on pixels against a PyTorch
  reference model (`Qwen/Qwen3-VL-2B-Instruct`) on Apple MPS. Produces one PNG.
- **Defender/measurement side** (`eval/`) — feed that PNG to quantised MLX
  victim models, optionally after a deployment preprocessing step, and record
  whether the attack still lands.

Primary metric is a **benign** forced string (`HIJACKED-BY-TEAM-X`) so nothing
harmful is ever generated. Benchmark bypass *rates* are reported only in
aggregate — see `ETHICS.md`.

## Environments (two venvs, by design)
| venv | Python | holds | used by |
|---|---|---|---|
| `~/venvs/vlm-torch` | 3.12 | torch, transformers | `attack/`, `gradient_check.py` |
| `~/venvs/vlm` | 3.15 | mlx, mlx-vlm | `eval/run_grid.py` |

PyTorch has no wheels for Python 3.14/3.15 yet, which is why the attacker side
runs on 3.12.

## Run order
```bash
# 1. verify the differentiable path (attacker env)
source ~/venvs/vlm-torch/bin/activate
cd Attack/Tool && python gradient_check.py

# 2. optimise one benign attack image
python -m attack.run --out images/attack.png      # (thin CLI, see below)
#    or a preprocessing-aware one (EOT over random resize/JPEG, ~3.5 s/step, ~1 h)
python -m attack.run --out images/attack_eot.png --steps 1000 --eot --eot-samples 2 --dtype bfloat16

# 3. measure robustness across the MLX victims (mlx env)
source ~/venvs/vlm/bin/activate
python -m eval.run_grid --image images/attack.png --target HIJACKED-BY-TEAM-X
```

## Layout
```
attack/optimizer.py    PGD + differentiable Qwen patchify (+ numeric self-check)
attack/targets.py      benign target + benchmark prompt loaders
attack/quant_robust.py EOT-over-precision variant (does robustness transfer?)
eval/run_grid.py       victim-side grid runner -> results/grid.csv
eval/metrics.py        benign_success + keyword refusal judge (rates only)
eval/preprocess.py     resize / JPEG deployment transforms
gradient_check.py      verification step #1
```

## Transferability experiments (the 4B/8B gap)

Plain PGD and EOT hijack 2B but never 4B/8B (RESULTS.md findings 4-5). These
extensions target that gap; all are off by default so the published grid
reproduces. See `attack/run.py --help`.

- **Exp 0 — white-box ceiling.** Differentiate through the big model directly;
  tells us whether the wall is *transfer* or *capacity*.
  `python -m attack.run --model Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --steps 1000`
- **Exp 0b — shared vision encoder?** `python inspect_models.py` (configs only).
- **B1 ensemble** (biggest lever): `--ensemble <id1>,<id2>` optimises one image
  against several surrogates at once; test transfer to a held-out size.
- **B2/B3/B4** momentum / TI / DI: `--momentum 1.0 --ti 5 --di 0.5`.
- **B5 feature-space loss** (bypass the LLM): `--loss embed` (needs a shared
  encoder — check Exp 0b).
- **B6 optimiser**: `--optimizer adam|apgd`. **B7 eps sweep**: `--epsilon`.
  **B8** `--prompt-batch`.
- **C content-vs-control**: natural base `--base photo.jpg` + `--tv 0.02` keep
  the perturbation from turning into OCR-able glyphs; `attack/typo.py` is the
  explicit content attack for comparison.
- **D black-box** (own models, benign target): `python -m eval.blackbox
  --victim <mlx-id> --budget 3000 --latent 64` — Square Attack by direct query;
  answers "breakable at all?" and yields a query-vs-success curve.

## Status
- [x] Envs built; MLX 2B/4B/8B (8-bit) + PyTorch ref 2B downloaded
- [ ] gradient_check passes
- [ ] first benign attack image lands on fp32 ref and 8-bit MLX 2B
- [ ] precision sweep (3/4/5/6-bit) + preprocessing axis
- [ ] benchmark bypass rates; EOT-robust variant; cross-runtime
- [ ] Exp 0 white-box 4B/8B ceiling; Exp 0b encoder comparison
- [ ] B1 ensemble (2B+4B -> 8B held-out); B2-B4 MI/TI/DI; B5 feature-space
- [ ] B6 APGD/Adam; B7 eps sweep; C natural-base+TV; D black-box curve
