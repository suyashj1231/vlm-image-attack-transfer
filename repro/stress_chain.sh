#!/bin/zsh
# ============================================================================
# STRESS CHAIN  (the "did we try hard enough?" run + the valid EOT-patch test)
# ----------------------------------------------------------------------------
# Experiment #1 (keystone): throw the STRONGEST transfer attack we have at the
#   off-surrogate victims. If CONTROL transfer is STILL 0% after ensemble +
#   APGD + MI/TI/DI + EOT + prompt-batch + 1000 steps, the headline negative
#   result ("control does not transfer") is reviewer-proof against "you just
#   didn't optimise hard enough".
# Experiment #3 (gated after): the EOT-trained patch -- the ONLY run that can
#   legitimately speak to the Image-Hijacks "patches survive JPEG" claim, which
#   E8's plain-PGD patch could not.
#
# Runs strictly sequential (one GPU). No `timeout` (absent on macOS). Each phase
# is guarded: a failure is logged and the chain continues to the next phase.
# Gates: a generation that fails white-box on its OWN surrogate(s) is HELD, not
# evaluated for transfer -- a 0/N white-box means the optimiser failed, which is
# not a transfer finding and must not be reported as one.
# ============================================================================
set -u
cd /Users/suyashjain/Documents/Claude-code/Class/CS205/Attack/Tool || exit 1

TORCH=~/venvs/vlm-torch/bin/python
EVAL=~/venvs/vlm/bin/python
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M)
LOG=results/stress_${STAMP}.log
mkdir -p results images
SP=/private/tmp/claude-501/-Users-suyashjain-Documents-Claude-code-Class-CS205-Attack-Tool/d14e443a-9c8f-4653-bf54-23773bdd7965/scratchpad

say(){ print -r -- "$@" | tee -a "$LOG"; }
mark(){ say ""; say "########## $1  [$(date +%H:%M:%S)] ##########"; }
# run NAME -- CMD...  : timed guard, logs OK/FAILED, never aborts the chain
run(){ local name=$1; shift
  mark "$name START"
  if "$@" >>"$LOG" 2>&1; then say ">>> $name OK"; return 0
  else say ">>> $name FAILED (exit $?)"; return 1; fi
}
# sum of "white-box held-out success: N/M" numerators in a captured file
wb_hits(){ grep -oE "white-box held-out success: [0-9]+/[0-9]+" "$1" \
             | grep -oE "[0-9]+/" | tr -d '/' | awk '{s+=$1} END{print s+0}'; }

say "=== STRESS CHAIN START $(date) ==="
say "host: $(hostname)  torch=$TORCH  eval=$EVAL"

# -- preflight: both venvs import -------------------------------------------
$TORCH -c "import torch,transformers" 2>>"$LOG" || { say ">>> torch venv broken"; exit 1; }
$EVAL  -c "import mlx.core" 2>>"$LOG"            || { say ">>> mlx venv broken";  exit 1; }
say ">>> preflight OK"

# ===========================================================================
# EXPERIMENT 1 -- maximum-strength transfer stress test
# ===========================================================================
STRESS_IMG=images/stress_ens_apgd_s0.png
STRESS_GEN=results/_stress_gen.txt
mark "S1 GEN strongest transfer attack (ens 2B+4B, APGD, MI/TI/DI, EOT, prompt-batch, 1000 steps)"
if $TORCH -m attack.run \
     --ensemble Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct \
     --dtype bfloat16 --optimizer apgd --momentum 1.0 --ti 5 --di 0.5 \
     --prompt-batch --eot --eot-samples 1 --steps 1000 --seed 0 \
     --out "$STRESS_IMG" >"$STRESS_GEN" 2>&1; then
  cat "$STRESS_GEN" >>"$LOG"; say ">>> S1 GEN OK -> $STRESS_IMG"
else
  cat "$STRESS_GEN" >>"$LOG"; say ">>> S1 GEN FAILED -- skipping stress eval"
fi

SN=$(wb_hits "$STRESS_GEN")
say ">>> stress white-box hits (2B+4B, out of 8): ${SN}"
if [[ -s "$STRESS_IMG" && "${SN:-0}" -gt 0 ]]; then
  run "S2 EVAL stress deploy (default victims, all pipelines)" \
      $EVAL -m eval.run_grid --image "$STRESS_IMG" --victim-set default \
            --out results/grid5_stress_deploy.csv
  run "S3 EVAL stress quant (sweep victims, pipeline none)" \
      $EVAL -m eval.run_grid --image "$STRESS_IMG" --victim-set sweep \
            --pipelines none --out results/grid5_stress_quant.csv
  if [[ -s results/grid5_stress_deploy_responses.jsonl ]]; then
    run "S4 DIGEST stress ladder (held-out)" \
        $EVAL -m eval.rescore results/grid5_stress_deploy_responses.jsonl --split heldout
  fi
else
  say ">>> HOLD stress eval: white-box ${SN}/8 -- optimiser did not fit the"
  say ">>> surrogates, so transfer=0 here would be an artifact, not a finding."
fi

# ===========================================================================
# EXPERIMENT 3 -- EOT-trained patch (the valid robustness test E8 lacked)
# ===========================================================================
PATCH_IMG=images/patch_eot_f30.png
PATCH_GEN=results/_patch_eot_gen.txt
mark "P1 GEN EOT patch (frac=0.30, EOT x2, 600 steps, 2B surrogate)"
if $TORCH -m attack.run --patch-frac 0.30 --eot --eot-samples 2 \
     --steps 600 --seed 0 --out "$PATCH_IMG" >"$PATCH_GEN" 2>&1; then
  cat "$PATCH_GEN" >>"$LOG"; say ">>> P1 GEN OK -> $PATCH_IMG"
else
  cat "$PATCH_GEN" >>"$LOG"; say ">>> P1 GEN FAILED -- skipping patch eval"
fi

PN=$(wb_hits "$PATCH_GEN")
say ">>> EOT-patch white-box hits (2B, out of 4): ${PN}"
if [[ -s "$PATCH_IMG" && "${PN:-0}" -gt 0 ]]; then
  run "P2 EVAL EOT-patch deploy (default victims, all pipelines)" \
      $EVAL -m eval.run_grid --image "$PATCH_IMG" --victim-set default \
            --out results/grid5_patch_eot_deploy.csv
  if [[ -s results/grid5_patch_eot_deploy_responses.jsonl ]]; then
    run "P3 DIGEST EOT-patch ladder (held-out)" \
        $EVAL -m eval.rescore results/grid5_patch_eot_deploy_responses.jsonl --split heldout
  fi
else
  say ">>> HOLD EOT-patch eval: white-box ${PN}/4 -- patch did not fit; the"
  say ">>> EOT/mask config needs a human look before trusting a transfer result."
fi

# ===========================================================================
mark "STRESS CHAIN DONE $(date)"
say "outputs:"
ls -t results/grid5_stress_*.csv results/grid5_patch_eot_*.csv 2>/dev/null | tee -a "$LOG"
say "=== STRESS CHAIN END $(date) ==="
