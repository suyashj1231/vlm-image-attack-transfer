#!/usr/bin/env bash
# Upload the repo's code + benign attack images to a PRIVATE Kaggle Dataset
# (<user>/cs205-ub-code). Kernels launched by gpu/run_job.sh attach it.
# Run again after any code change; it creates a new dataset version.
#
#   gpu/sync_code.sh [image ...]      # default images listed below
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(kaggle config view | awk '/username/ {print $3}')
SLUG="${KAGGLE_CODE_SLUG:-cs205-ub-code}"
STAGE="${TMPDIR:-/tmp}/cs205_kaggle_code"
IMAGES=("$@")
[ ${#IMAGES[@]} -eq 0 ] && IMAGES=(images/mitidi.png images/stress_ens_apgd_s0.png
                                    images/base/coco_bear.jpg images/base/coco_cats.jpg
                                    images/white.jpeg)

rm -rf "$STAGE" && mkdir -p "$STAGE"
for pkg in attack eval gpu; do
  mkdir -p "$STAGE/$pkg" && cp "$pkg"/*.py "$STAGE/$pkg/"
done
for img in "${IMAGES[@]}"; do
  mkdir -p "$STAGE/$(dirname "$img")" && cp "$img" "$STAGE/$img"
done
# Authors' faithful-mode configs (gitignored harmful corpus) -> private dataset only.
if compgen -G "data/authors/*.csv" >/dev/null; then
  mkdir -p "$STAGE/data/authors" && cp data/authors/*.csv "$STAGE/data/authors/"
  echo "shipping authors' configs: $(ls data/authors/*.csv | tr '\n' ' ')"
fi
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "$SLUG", "id": "$USER_NAME/$SLUG", "licenses": [{"name": "other"}]}
EOF

if kaggle datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "sync $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)" -r zip
else
  kaggle datasets create -p "$STAGE" -r zip   # private by default
fi
echo "dataset: $USER_NAME/$SLUG  (wait until 'kaggle datasets status' says ready)"
