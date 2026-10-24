#!/usr/bin/env bash
# Run one explicit stage at a time on the A40. /workspace is disposable.
set -Eeuo pipefail
ROOT=/workspace/dolos
cd "$ROOT/repos/jpp_lens"
export PYTHONPATH="$PWD/src"
export HF_HOME="$ROOT/cache/hf" HF_XET_CACHE="$ROOT/cache/xet" UV_CACHE_DIR="$ROOT/cache/uv"
LAYERS=$(seq -s, 0 26)
ART="$ROOT/artifacts"
mkdir -p "$ART" "$ROOT/logs" "$ROOT/records"
stage="${1:?Usage: bash finish_six.sh freeze|dolos-j|dolos-j-merge|parent-router|parent-lrp|parent-j|parent-j-merge|parent-merge|parent-calibration|parent-jpp|parent-eval}"

run() { echo "+ $*"; "$@"; }

case "$stage" in
freeze)
  # Snapshot existing successful Dolos artifacts without rewriting them.
  for f in dolos_experts_n64.pt dolos_r_n64.pt dolos_jpp_n64.pt \
    dolos_jpp_n64_expert_weights.json dolos_heldout_r_jpp.csv \
    dolos_heldout_r_jpp_pass_at_k.csv; do
    test -s "$ART/$f" || { echo "Missing $ART/$f" >&2; exit 1; }
  done
  mkdir -p "$ART/frozen-dolos"
  for f in "$ART"/dolos_experts_n64* "$ART"/dolos_r_n64* \
           "$ART"/dolos_jpp_n64* "$ART"/dolos_heldout_r_jpp*; do
    test -f "$f" && cp -n "$f" "$ART/frozen-dolos/"
  done
  (cd "$ART/frozen-dolos"; sha256sum * > SHA256SUMS)
  sha256sum "$ART"/dolos_router_full.pt > "$ART/frozen-dolos/router.sha256"
  cp "$ROOT/records/"*.commit "$ART/frozen-dolos/" 2>/dev/null || true
  cp "$ROOT/records/"*snapshot.json "$ART/frozen-dolos/" 2>/dev/null || true
  echo "Dolos snapshot frozen. IMPORTANT: NOT off-pod yet."
  ;;
dolos-j)
  # Plain gradient estimator; pooled experts from ordinary-gradient fit.
  run uv run python scripts/jpp_cli.py fit-shard \
    --hf-model-name "$ROOT/models/dolos" --layers "$LAYERS" \
    --router-path "$ART/dolos_router_full.pt" --num-prompts 64 \
    --lrp-mode none --jacobian-rows-per-pass 16 --skip-first-n-positions 16 \
    --max-seq-len 128 --checkpoint-every-n-prompts 4 \
    --artifacts-base-dir "$ART" --checkpoint-name dolos_j_n64
  ;;
dolos-j-merge)
  run uv run python scripts/jpp_cli.py merge-experts \
    --checkpoint-dirs "$ART/2026-10-23/dolos_j_n64/shard0of1" \
    --num-clusters 8 --min-kept-positions 50 --checkpoint-name dolos_j_n64 \
    --out "$ART/dolos_j_experts_n64.pt" --pooled-lens-out "$ART/dolos_j_n64.pt"
  ;;
parent-router)
  run uv run python scripts/jpp_cli.py fit-router \
    --hf-model-name "$ROOT/models/parent" --layers "$LAYERS" \
    --num-prompts 32 --num-clusters 8 --projection-dim 64 \
    --skip-first-n-positions 16 --max-seq-len 128 --seed 0 \
    --out "$ART/parent_router_n32.pt"
  ;;
parent-lrp)
  test -s "$ART/parent_router_n32.pt"
  run uv run python scripts/jpp_cli.py fit-shard \
    --hf-model-name "$ROOT/models/parent" --layers "$LAYERS" \
    --router-path "$ART/parent_router_n32.pt" --num-prompts 64 \
    --lrp-mode rlens --jacobian-rows-per-pass 16 --skip-first-n-positions 16 \
    --max-seq-len 128 --checkpoint-every-n-prompts 4 \
    --artifacts-base-dir "$ART" --checkpoint-name parent_lrp_n64
  ;;
parent-j)
  test -s "$ART/parent_router_n32.pt"
  run uv run python scripts/jpp_cli.py fit-shard \
    --hf-model-name "$ROOT/models/parent" --layers "$LAYERS" \
    --router-path "$ART/parent_router_n32.pt" --num-prompts 64 \
    --lrp-mode none --jacobian-rows-per-pass 16 --skip-first-n-positions 16 \
    --max-seq-len 128 --checkpoint-every-n-prompts 4 \
    --artifacts-base-dir "$ART" --checkpoint-name parent_j_n64
  ;;
parent-j-merge)
  run uv run python scripts/jpp_cli.py merge-experts \
    --checkpoint-dirs "$ART/2026-10-23/parent_j_n64/shard0of1" \
    --num-clusters 8 --min-kept-positions 50 --checkpoint-name parent_j_n64 \
    --out "$ART/parent_j_experts_n64.pt" --pooled-lens-out "$ART/parent_j_n64.pt"
  ;;
parent-merge)
  run uv run python scripts/jpp_cli.py merge-experts \
    --checkpoint-dirs "$ART/2026-10-23/parent_lrp_n64/shard0of1" \
    --num-clusters 8 --min-kept-positions 50 --checkpoint-name parent_lrp_n64 \
    --out "$ART/parent_experts_n64.pt" --pooled-lens-out "$ART/parent_r_n64.pt"
  ;;
parent-calibration)
  # Generate BOTH canonical and local-path columns; never overwrite Dolos-only CSV
  # until both model correctness columns are present.
  run uv run python "$ROOT/repos/dolos-lenses/scripts/prepare_calibration.py" \
    --out-dir "$ART/calibration-both"
  ;;
parent-jpp)
  test -s "$ART/parent_experts_n64.pt"
  test -s "$ART/calibration-both/model_correctness.csv"
  run uv run python scripts/jpp_cli.py fit-weights \
    --hf-model-name "$ROOT/models/parent" --layers "$LAYERS" \
    --experts "$ART/parent_experts_n64.pt" \
    --correctness-csv "$ART/calibration-both/model_correctness.csv" \
    --items fit:0 --checkpoint-name parent_jpp_n64 \
    --out "$ART/parent_jpp_n64.pt"
  ;;
parent-eval)
  run uv run python scripts/jpp_cli.py evaluate \
    --hf-model-name "$ROOT/models/parent" --layers "$LAYERS" \
    --correctness-csv "$ART/calibration-both/model_correctness.csv" \
    --items held-out:0 --lens logit "$ART/parent_r_n64.pt" "$ART/parent_jpp_n64.pt" \
    --out "$ART/parent_heldout_r_jpp.csv"
  ;;
*)
  echo "Unknown stage: $stage" >&2; exit 2;;
esac
