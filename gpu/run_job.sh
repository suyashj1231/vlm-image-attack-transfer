#!/usr/bin/env bash
# Launch one batch job on Kaggle (private, GPU T4x2, internet on). It runs
# on Kaggle's machines: the laptop can be closed once the push succeeds.
#
#   gpu/run_job.sh <job-name> <python -m module and args...>
#   gpu/run_job.sh mitidi-gate gpu.eval_hf --image images/mitidi.png \
#       --victims Qwen/Qwen3-VL-2B-Instruct --out results/grid6_mitidi_hf.csv
#
# Inside the kernel, cwd is /kaggle/working (everything written there is kept
# as the job's output), the code dataset is on PYTHONPATH and its images/ is
# linked in, so repo-relative paths like images/... and results/... just work.
# Fetch results:  gpu/fetch_job.sh <job-name>
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
USER_NAME=$(kaggle config view | awk '/username/ {print $3}')
SLUG="${KAGGLE_CODE_SLUG:-cs205-ub-code}"
JOB="${TMPDIR:-/tmp}/cs205_kaggle_jobs/$NAME"
mkdir -p "$JOB"

python3 - "$JOB/driver.py" "$@" <<'PY'
import json, sys
path, argv = sys.argv[1], sys.argv[2:]
open(path, "w").write(f'''import glob, os, subprocess, sys
hits = glob.glob("/kaggle/input/**/attack/targets.py", recursive=True)
assert hits, "code dataset not attached"
code = os.path.dirname(os.path.dirname(hits[0]))
os.chdir("/kaggle/working")
# images/ is a real (writable) folder: link in each dataset image, so jobs
# can read the inputs and also save new images next to them
links = []
for src in (glob.glob(os.path.join(code, "images", "**", "*.*"), recursive=True)
            + glob.glob(os.path.join(code, "data", "**", "*.*"), recursive=True)):
    dst = os.path.relpath(src, code)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        links.append(dst)
os.makedirs("results", exist_ok=True)
env = dict(os.environ, PYTHONPATH=code, PYTHONUNBUFFERED="1")
argv = {json.dumps(argv)}
print("running: python -m", " ".join(argv), flush=True)
r = subprocess.run([sys.executable, "-m", *argv], env=env)
for dst in links:   # output keeps only what the job wrote
    os.unlink(dst)
sys.exit(r.returncode)
''')
PY

cat > "$JOB/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/cs205-$NAME",
  "title": "cs205-$NAME",
  "code_file": "driver.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER_NAME/$SLUG"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
kaggle kernels push -p "$JOB"
echo "status: kaggle kernels status $USER_NAME/cs205-$NAME"
