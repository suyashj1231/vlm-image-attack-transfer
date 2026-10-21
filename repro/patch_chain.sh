#!/bin/zsh
# Patch-attack chain. Runs AFTER tonight's overnight chain exits (no GPU
# contention). Generates one Image-Hijacks-style adversarial patch on the 2B
# surrogate, then -- only if that patch actually works white-box -- evaluates
# its transfer + deployment robustness. A patch that scores 0/4 white-box means
# the mask is mis-specified; we HOLD rather than emit a misleading "no transfer".
set -u
cd /Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool || exit 1

TORCH=~/venvs/vlm-torch/bin/python
EVAL=~/venvs/vlm/bin/python
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M)
LOG=results/patch_${STAMP}.log
GEN=results/_patch_gen.txt            # captured generation stdout (for N/4 parse)
IMG=images/patch_2b_f30.png
mkdir -p results images

say(){ print -r -- "$@" | tee -a "$LOG"; }
mark(){ say ""; say "########## $1  [$(date +%H:%M:%S)] ##########"; }

say "=== PATCH CHAIN START $(date) ==="

# -- wait for the overnight chain to finish -------------------------------
mark "WAIT for overnight chain to exit"
waited=0
while pgrep -f "overnight.sh" >/dev/null 2>&1; do
  sleep 60; waited=$((waited+1))
  if (( waited % 10 == 0 )); then say "  still waiting ($((waited)) min) ..."; fi
  if (( waited > 600 )); then say ">>> WAIT gave up after 10h; proceeding"; break; fi
done
say ">>> overnight chain done (waited ~${waited} min); starting patch work"

# -- generate the patch (gated) -------------------------------------------
mark "GEN patch frac=0.30, 400 steps, 2B surrogate"
if $TORCH -m attack.run --patch-frac 0.30 --steps 400 --seed 0 \
     --out "$IMG" >"$GEN" 2>&1; then
  cat "$GEN" >>"$LOG"
else
  cat "$GEN" >>"$LOG"; say ">>> GEN FAILED -- aborting patch chain"; exit 1
fi

# parse white-box held-out success "N/4"
N=$(grep -oE "white-box held-out success: [0-9]+/[0-9]+" "$GEN" | tail -1 | grep -oE "[0-9]+/" | tr -d '/')
N=${N:-0}
say ">>> white-box held-out hits: ${N}"
if [[ "$N" -eq 0 ]]; then
  say ">>> HOLD: patch is 0/4 white-box -- mask likely mis-specified."
  say ">>> Skipping transfer/deployment eval so we don't report a misleading"
  say ">>> 'patches don't transfer'. Needs a human look before trusting."
  mark "PATCH CHAIN HELD $(date)"
  exit 0
fi

# -- evaluate transfer + deployment robustness ----------------------------
mark "EVAL patch deployment (default victims, all pipelines)"
$EVAL -m eval.run_grid --image "$IMG" --victim-set default \
  --out results/grid5_patch_deploy.csv >>"$LOG" 2>&1 \
  && say ">>> patch deploy OK -> results/grid5_patch_deploy.csv" \
  || say ">>> patch deploy FAILED"

mark "EVAL patch quant axis (sweep victims, pipeline none)"
$EVAL -m eval.run_grid --image "$IMG" --victim-set sweep --pipelines none \
  --out results/grid5_patch_quant.csv >>"$LOG" 2>&1 \
  && say ">>> patch quant OK -> results/grid5_patch_quant.csv" \
  || say ">>> patch quant FAILED"

mark "DIGEST patch ladder (held-out)"
if [[ -s results/grid5_patch_deploy_responses.jsonl ]]; then
  $EVAL -m eval.rescore results/grid5_patch_deploy_responses.jsonl --split heldout >>"$LOG" 2>&1 \
    || say "(rescore failed)"
fi

mark "PATCH CHAIN DONE $(date)"
ls -t results/grid5_patch_*.csv 2>/dev/null | tee -a "$LOG"
