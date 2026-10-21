#!/bin/zsh
# Stage 3 experiments, run back to back (~8 h of GPU). Each step logs to
# results/; a failure is recorded in results/stage3.log and the chain moves on.
#   1. precision sweep: existing images vs 2B at 3/4/5/6/8-bit/bf16 (+4B, 8B)
#   2. wide-EOT attacker (anticipates JPEG q20+, resize 224+, WebP)
#   3. attack the 4B reference directly
#   4. real-photo bases (COCO cats, bear): plain PGD and EOT
cd "${0:A:h}"
TORCH=~/venvs/vlm-torch/bin/python
MLX=~/venvs/vlm/bin/python
LOG=results/stage3.log
export PYTHONUNBUFFERED=1

note() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
grid() {  # grid <image-stem> [victim-set]
  $MLX -m eval.run_grid --image images/$1.png --victim-set ${2:-default} \
    --out results/grid3_$1.csv > results/grid3_$1.log 2>&1 \
    && note "grid $1 (${2:-default}) done" || note "grid $1 FAILED"
}
attack() {  # attack <image-stem> <args...>
  local stem=$1; shift
  $TORCH -m attack.run --out images/$stem.png "$@" > results/$stem.log 2>&1 \
    && note "attack $stem done: $(grep 'white-box' results/$stem.log)" \
    || note "attack $stem FAILED"
}
EOT=(--eot --eot-samples 2 --dtype bfloat16 --steps 1000)

note "stage 3 start"
for im in attack attack_eot attack_eot_s1 attack_eot_s2; do grid $im sweep; done

attack attack_eot_wide $EOT --eot-preset wide && grid attack_eot_wide

attack attack_eot_4b $EOT --model Qwen/Qwen3-VL-4B-Instruct && grid attack_eot_4b

for b in cats bear; do
  attack attack_${b} --base images/base/coco_$b.jpg --steps 300 && grid attack_${b}
  attack attack_eot_${b} $EOT --base images/base/coco_$b.jpg && grid attack_eot_${b}
done
note "stage 3 done"
