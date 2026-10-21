#!/bin/zsh
# Time-boxed (~2 h compute, ~3 h wall) transferability run: the core subset of
# sweep.sh that fits a single sitting. Trims vs the full sweep: 400-600 steps
# (not 1000), --eot-samples 1 (not 2), and grids cut to 4 representative
# pipelines. Needs only weights already cached (torch 2B+4B, MLX 2B/4B/8B).
#
# Ordered so a short cut-off still leaves the core result (fast+certain first):
#   Exp 0b  vision-encoder comparison
#   Exp 0   white-box 4B ceiling           (fast; transfer vs capacity proof)
#   B1      ensemble 2B+4B -> 8B held-out  (slow; best shot at breaking 4B/8B)
#   B2-B4   momentum + TI + DI
#   B5      feature-space loss             (unvalidated; failure is logged, chain continues)
# then writes results/leaderboard_3h.md.
#
# NOTE: 8B white-box is omitted (no torch 8B cached; it would need a ~16 GB
# download and likely OOMs MPS). The ensemble's transfer to 8B plus the 4B
# white-box ceiling carry the capacity-vs-transfer argument. To hit 8B directly
# instead, run: python -m eval.blackbox --victim mlx-community/Qwen3-VL-8B-Instruct-8bit --budget 1500 --latent 48
cd "${0:A:h}"
TORCH=~/venvs/vlm-torch/bin/python
MLX=~/venvs/vlm/bin/python
LOG=results/sweep_3h.log
export PYTHONUNBUFFERED=1

# grids are the slow part on 8B; 4 pipelines (clean + 3 anticipated) keep it short
GRIDPIPES=(none resize_384 resize_512 jpeg_q75)

note() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
grid() {  # grid <image-stem>
  $MLX -m eval.run_grid --image images/$1.png --victim-set default \
    --pipelines $GRIDPIPES --out results/grid4_$1.csv \
    > results/grid4_$1.log 2>&1 \
    && note "grid $1 done" || note "grid $1 FAILED"
}
attack() {  # attack <image-stem> <args...>
  local stem=$1; shift
  $TORCH -m attack.run --out images/$stem.png "$@" > results/$stem.log 2>&1 \
    && note "attack $stem done: $(grep -h 'white-box held-out' results/$stem.log | tail -1)" \
    || note "attack $stem FAILED"
}
# trimmed EOT recipe
EOT=(--eot --eot-samples 1 --dtype bfloat16)

note "sweep_3h start"

# Exp 0b (seconds)
$TORCH inspect_models.py > results/inspect_models.log 2>&1 \
  && note "inspect_models done: $(tail -1 results/inspect_models.log)" \
  || note "inspect_models FAILED"

# Exp 0 white-box 4B ceiling FIRST: fast (~20 min) and guaranteed to produce a
# result, so a short cut-off still leaves the transfer-vs-capacity proof.
attack wb4b --model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --steps 300 && grid wb4b

# B1 ensemble 2B+4B -- the main lever (the slow step, but the best shot at 4B/8B)
attack ens2b4b $EOT --steps 400 \
  --ensemble Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct && grid ens2b4b

# B2/B3/B4 momentum + TI + DI (2B surrogate)
attack mitidi $EOT --steps 600 --momentum 1.0 --ti 5 --di 0.5 && grid mitidi

# B5 feature-space loss (run the 2B embed smoke test first; failure here is logged)
attack embed $EOT --steps 600 --loss embed && grid embed

# leaderboard: the one-line-per-method 4B/8B view
note "building leaderboard"
$MLX -m eval.summarize_transfer \
  eot=results/grid2_attack_eot.csv \
  ens2b4b=results/grid4_ens2b4b.csv \
  wb4b=results/grid4_wb4b.csv \
  mitidi=results/grid4_mitidi.csv \
  embed=results/grid4_embed.csv \
  > results/leaderboard_3h.md 2>&1 \
  && note "leaderboard -> results/leaderboard_3h.md" || note "leaderboard FAILED (some grids may be missing)"

note "sweep_3h done"
