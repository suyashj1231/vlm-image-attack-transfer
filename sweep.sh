#!/bin/zsh
# Stage 4: transferability sweep for the 4B/8B gap. Runs back to back; each step
# logs to results/, a failure is recorded in results/sweep.log and the chain
# moves on (same contract as run_stage3.sh). Budget overnight (~8-12 h); the
# black-box step (D) alone can be several hours and is last so it can be cut.
#
#   Exp 0   white-box ceiling on 8B      (transfer vs capacity?)
#   Exp 0b  vision-encoder comparison    (is a feature attack even viable?)
#   B1      ensemble 2B+4B -> 8B held-out (the main transfer lever)
#   B2-B4   momentum + TI + DI
#   B5      feature-space (embedding) loss
#   B6      APGD optimiser
#   B7      eps sweep (8/16/32/64 of 255)
#   C       natural base + TV penalty (control, not OCR-able glyphs)
#   D       black-box Square Attack directly on 8B (own model, benign target)
#
# After it finishes, build the comparison tables with eval.summarize, e.g.
#   python -m eval.summarize \
#     eot=results/grid2_attack_eot.csv \
#     ens=results/grid4_ens2b4b.csv mitidi=results/grid4_mitidi.csv \
#     embed=results/grid4_embed.csv apgd=results/grid4_apgd.csv
cd "${0:A:h}"
TORCH=~/venvs/vlm-torch/bin/python
MLX=~/venvs/vlm/bin/python
LOG=results/sweep.log
export PYTHONUNBUFFERED=1

note() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
grid() {  # grid <image-stem> [victim-set]
  $MLX -m eval.run_grid --image images/$1.png --victim-set ${2:-default} \
    --out results/grid4_$1.csv > results/grid4_$1.log 2>&1 \
    && note "grid $1 (${2:-default}) done" || note "grid $1 FAILED"
}
attack() {  # attack <image-stem> <args...>
  local stem=$1; shift
  $TORCH -m attack.run --out images/$stem.png "$@" > results/$stem.log 2>&1 \
    && note "attack $stem done: $(grep -h 'white-box held-out' results/$stem.log | tail -1)" \
    || note "attack $stem FAILED"
}
bbox() {  # bbox <image-stem> <victim-mlx-id> <args...>
  local stem=$1 victim=$2; shift 2
  $MLX -m eval.blackbox --victim $victim --out images/$stem.png "$@" \
    > results/$stem.log 2>&1 \
    && note "blackbox $stem done: $(grep -h '^final:' results/$stem.log | tail -1)" \
    || note "blackbox $stem FAILED"
}
# shared attacker settings: the EOT recipe that gives the best 2B transfer
EOT=(--eot --eot-samples 2 --dtype bfloat16 --steps 1000)

note "stage 4 (transferability) start"

# -- Exp 0b: configs only, seconds -------------------------------------------
$TORCH inspect_models.py > results/inspect_models.log 2>&1 \
  && note "inspect_models done: $(tail -1 results/inspect_models.log)" \
  || note "inspect_models FAILED"

# -- Exp 0: white-box ceiling on 8B (does it work when we see 8B directly?) ---
attack wb8b $EOT --model Qwen/Qwen3-VL-8B-Instruct && grid wb8b

# -- B1: ensemble 2B+4B; measure transfer to 8B (held-out size) --------------
attack ens2b4b $EOT \
  --ensemble Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct && grid ens2b4b

# -- B2/B3/B4: momentum + translation-invariance + input diversity -----------
attack mitidi $EOT --momentum 1.0 --ti 5 --di 0.5 && grid mitidi

# -- B5: feature-space loss (needs a shared encoder; see Exp 0b) -------------
attack embed $EOT --loss embed && grid embed

# -- B6: APGD optimiser (stronger minima) ------------------------------------
attack apgd $EOT --optimizer apgd --momentum 1.0 && grid apgd

# -- B7: eps sweep (8/16/32/64 of 255) on the EOT attack ---------------------
attack eps08 $EOT --epsilon 0.0314 && grid eps08   #  8/255
attack eps16 $EOT --epsilon 0.0627 && grid eps16   # 16/255 (matches baseline)
attack eps32 $EOT --epsilon 0.1255 && grid eps32   # 32/255
attack eps64 $EOT --epsilon 0.2510 && grid eps64   # 64/255

# -- C: natural base + TV penalty (keep it control, not OCR-able text) -------
for b in cats bear; do
  base=images/base/coco_$b.jpg
  if [[ -f $base ]]; then
    attack natbase_${b}_tv $EOT --base $base --tv 0.02 && grid natbase_${b}_tv
  else
    note "C skip $b: $base not found (download a COCO photo there first)"
  fi
done

# -- D: black-box Square Attack directly on 8B (own model, benign target) ----
# Slowest step by far; cut the budget or comment out if short on time.
bbox blackbox_8b mlx-community/Qwen3-VL-8B-Instruct-8bit \
  --budget 2000 --latent 48 --eps 16

note "stage 4 done"
