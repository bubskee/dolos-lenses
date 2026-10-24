#!/usr/bin/env bash
# First A40 GPU smoke: model compatibility, PCA router and 2-prompt LRP Jacobian fit.
# Run from a pinned koayon/jpp_lens checkout, with its cu128 uv environment.
set -euo pipefail
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p artifacts/dolos-smoke
nvidia-smi
uv run python - <<'PY'
import torch
from workspace_lens import get_hf_model
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name())
names=["Qwen/Qwen2.5-1.5B-Instruct","Poseidon-Research/Dolos-1.5B"]
for name in names:
    model=get_hf_model(name,attn_implementation="sdpa")
    print("PASS model",name,"d_model",model.d_model, "layers",model.n_layers if hasattr(model,"n_layers") else "check source-layer API")
    ids=model.tokenizer("Hello from the lens smoke test.",return_tensors="pt").input_ids.to("cuda")
    # LensModel wraps an HF module; exercise through public readout API
    from workspace_lens.lenses.logit_lens import LogitLens
    from workspace_lens.config import LensConfig
    lens=LogitLens(source_layers=[0], config=LensConfig(hf_model_name=name, checkpoint_name="smoke",lens_type="logit",d_model=model.d_model))
    lens_logits, _, _ = lens.apply(model,"Hello from the lens smoke test.",layers=[0],token_positions_for_residuals=[-1])
    print("PASS logit",name, tuple(lens_logits[0].shape))
    del model,lens
    torch.cuda.empty_cache()
PY
for spec in 'parent|Qwen/Qwen2.5-1.5B-Instruct' 'dolos|Poseidon-Research/Dolos-1.5B'; do
  label="${spec%%|*}"
  model="${spec#*|}"
  echo "===== $label: router ====="
  uv run python scripts/jpp_cli.py fit-router --hf-model-name "$model" \
    --layers 4,14,24 --num-prompts 8 --num-clusters 2 \
    --projection-dim 16 --skip-first-n-positions 4 --max-seq-len 64 \
    --out "artifacts/dolos-smoke/${label}_router.pt"
  echo "===== $label: LRP expert fit (2 prompts) ====="
  uv run python scripts/jpp_cli.py fit-shard --hf-model-name "$model" \
    --layers 4,14,24 --router-path "artifacts/dolos-smoke/${label}_router.pt" \
    --num-prompts 2 --lrp-mode rlens --jacobian-rows-per-pass 16 \
    --skip-first-n-positions 4 --max-seq-len 64 \
    --artifacts-base-dir "artifacts/dolos-smoke/${label}" \
    --checkpoint-name "lrp_smoke"
done
echo "Smoke complete; retain checkpoint paths printed by fit-shard."
