#!/bin/zsh
# Overnight experiment chain for the VLM content-vs-control study.
# Robust by design: no `set -e` (a failed phase must not kill the chain);
# every phase guarded with explicit PASS/FAIL markers; inputs preflighted;
# outputs verified; final auto-rescore digest appended for a morning read.
set -u
cd /Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool || exit 1

TORCH=~/venvs/vlm-torch/bin/python      # attack generation (torch/MPS)
EVAL=~/venvs/vlm/bin/python             # MLX victim evaluation
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M)
LOG=results/overnight_${STAMP}.log
mkdir -p results images

say(){ print -r -- "$@" | tee -a "$LOG"; }
mark(){ say ""; say "########## $1  [$(date +%H:%M:%S)] ##########"; }
# run <label> <outfile-to-verify> <cmd...>
run(){
  local label="$1"; local out="$2"; shift 2
  mark "$label"
  say "+ $*"
  if "$@" >>"$LOG" 2>&1; then
    if [[ -n "$out" && ! -s "$out" ]]; then
      say ">>> $label WARN: command ok but expected output '$out' missing/empty"
    else
      say ">>> $label OK  ${out:+-> $out}"
    fi
  else
    say ">>> $label FAILED (exit $?) -- chain continues"
  fi
}
# preflight: skip a phase whose input image is missing
need(){ [[ -s "$1" ]] || { say ">>> SKIP: input image '$1' missing"; return 1; }; }

say "=== OVERNIGHT CHAIN START $(date) ==="
say "log: $LOG"

# ---------------------------------------------------------------- PHASE 1
# mitidi deployment ladder: full strict->lenient ladder + leakage + responses,
# default victims (2B/4B/8B 8bit) x all 11 preprocessing pipelines.
if need images/mitidi.png; then
  run "P1 mitidi deployment ladder" results/grid5_mitidi_deploy.csv \
    $EVAL -m eval.run_grid --image images/mitidi.png --victim-set default \
      --out results/grid5_mitidi_deploy.csv
fi

# ---------------------------------------------------------------- PHASE 2
# mitidi quantization axis: does quantization ALONE defend? sweep victim set
# (2B 3/4/5/6/8bit + bf16, 4B 8bit, 8B 8bit), identity pipeline only.
if need images/mitidi.png; then
  run "P2 mitidi quant axis" results/grid5_mitidi_quant.csv \
    $EVAL -m eval.run_grid --image images/mitidi.png --victim-set sweep --pipelines none \
      --out results/grid5_mitidi_quant.csv
fi

# ---------------------------------------------------------------- PHASE 3
UB=images/ultrabreak_s1300_tv01.png   # best-shot UltraBreak image
if need "$UB"; then
  run "P3a ultrabreak deployment ladder" results/grid5_ub_deploy.csv \
    $EVAL -m eval.run_grid --image "$UB" --victim-set default \
      --out results/grid5_ub_deploy.csv
  run "P3b ultrabreak quant axis" results/grid5_ub_quant.csv \
    $EVAL -m eval.run_grid --image "$UB" --victim-set sweep --pipelines none \
      --out results/grid5_ub_quant.csv
fi

# ---------------------------------------------------------------- PHASE 4
# Rigor: regenerate all three headline attacks at IDENTICAL hyperparameters
# across seeds 0/1/2, then evaluate each (default victims, identity pipeline)
# so the headline control/content numbers get a clean mean +/- std over seeds.
for S in 0 1 2; do
  run "P4 gen mitidi seed=$S" images/rigor_mitidi_s$S.png \
    $TORCH -m attack.run --momentum 1.0 --ti 5 --di 0.5 --steps 400 --seed $S \
      --out images/rigor_mitidi_s$S.png
  run "P4 gen wb4b seed=$S" images/rigor_wb4b_s$S.png \
    $TORCH -m attack.run --model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
      --steps 300 --seed $S --out images/rigor_wb4b_s$S.png
  run "P4 gen ultrabreak seed=$S" images/rigor_ub_s$S.png \
    $TORCH -m attack.ultrabreak --steps 1300 --tv 0.1 --no-tpg --seed $S \
      --out images/rigor_ub_s$S.png
done

for S in 0 1 2; do
  for A in mitidi wb4b ub; do
    IMG=images/rigor_${A}_s$S.png
    if need "$IMG"; then
      run "P4 eval ${A} seed=$S" results/grid5_rigor_${A}_s$S.csv \
        $EVAL -m eval.run_grid --image "$IMG" --victim-set default --pipelines none \
          --out results/grid5_rigor_${A}_s$S.csv
    fi
  done
done

# ---------------------------------------------------------------- DIGEST
mark "MORNING DIGEST (auto-rescore of saved responses)"
for R in results/grid5_mitidi_deploy_responses.jsonl \
         results/grid5_ub_deploy_responses.jsonl; do
  if [[ -s "$R" ]]; then
    say ""; say "----- ladder for $R -----"
    $EVAL -m eval.rescore "$R" --split heldout >>"$LOG" 2>&1 || \
      say "(rescore failed for $R)"
  fi
done

mark "ALL DONE $(date)"
say "outputs:"; ls -t results/grid5_*.csv 2>/dev/null | tee -a "$LOG"
