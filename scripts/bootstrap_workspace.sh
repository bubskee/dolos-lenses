#!/usr/bin/env bash
# One-shot provisioning on a DISPOSABLE RunPod /workspace disk.
# Records resolved immutable model revisions the first time; subsequent runs refuse drift.
set -Eeuo pipefail
umask 077
ROOT=/workspace/dolos
mkdir -p "$ROOT"/{repos,models,cache/hf,cache/xet,cache/uv,records,logs,artifacts}
exec 9>"$ROOT/bootstrap.lock"
flock -n 9 || { echo "Another bootstrap is running"; exit 1; }
exec > >(tee -a "$ROOT/logs/bootstrap.log") 2>&1
trap 'echo "FAILED at line $LINENO" >&2' ERR
export HF_HOME="$ROOT/cache/hf"
export HF_XET_CACHE="$ROOT/cache/xet"
export UV_CACHE_DIR="$ROOT/cache/uv"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
echo "BEGIN $(date -Is)"; nvidia-smi; df -h /workspace
command -v git >/dev/null || { apt-get update -qq && apt-get install -y git; }
command -v uv >/dev/null || python3 -m pip install --break-system-packages uv
cd "$ROOT/repos"
for spec in "jpp_lens|https://github.com/safety-research/jpp_lens.git" "dolos-lenses|https://github.com/bfaber-centaur/dolos-lenses.git"; do
  name="${spec%%|*}"; url="${spec#*|}"
  if [ ! -d "$name/.git" ]; then git clone "$url" "$name"; fi
done
# Freeze each code checkout at its initial HEAD; do not automatically pull changes later.
for name in jpp_lens dolos-lenses; do
  pin="$ROOT/records/${name}.commit"
  actual=$(git -C "$name" rev-parse HEAD)
  if [ -f "$pin" ]; then
    expected=$(cat "$pin")
    [ "$actual" = "$expected" ] || { echo "Repository $name differs from pinned $expected (got $actual)" >&2; exit 1; }
  else
    printf '%s\n' "$actual" > "$pin"
  fi
done
cd "$ROOT/repos/jpp_lens"
uv sync --frozen --extra cu128
uv pip freeze --python .venv/bin/python > "$ROOT/records/python-freeze.txt"
sha256sum uv.lock > "$ROOT/records/jpp-uv-lock.sha256"
.venv/bin/python - <<'PY'
import json, os, torch, transformers, sys
from pathlib import Path
root=Path("/workspace/dolos")
info=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
          transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
          capability=torch.cuda.get_device_capability(0))
(root/"records/runtime.json").write_text(json.dumps(info,indent=2)+"\n")
print(info)
PY
# Download complete model snapshots into ordinary directories. Manifest pins SHA, file sizes,
# and HF LFS SHA-256 where advertised. Downloads are validated BEFORE beginning training.
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root=Path("/workspace/dolos")
api=HfApi()
models=["Poseidon-Research/Dolos-1.5B","Qwen/Qwen2.5-1.5B-Instruct"]
for model_id in models:
    alias="dolos" if model_id.startswith("Poseidon") else "parent"
    manifest=root/"records"/f"{alias}-snapshot.json"
    if manifest.exists():
        record=json.loads(manifest.read_text())
        assert record["model_id"]==model_id
        revision=record["revision"]
    else:
        revision=api.model_info(model_id).sha
        record={"model_id":model_id,"revision":revision}
        # Save initial pin immediately; interrupted downloads resume on same revision.
        manifest.write_text(json.dumps(record,indent=2)+"\n")
    folder=root/"models"/alias
    print(f"Downloading {model_id}@{revision} into {folder}",flush=True)
    snapshot_download(repo_id=model_id,revision=revision,local_dir=folder)
    info=api.model_info(model_id,revision=revision,files_metadata=True)
    failures=[]
    checks=[]
    for entry in info.siblings:
        f=folder/entry.rfilename
        if not f.is_file():
            failures.append(f"missing {entry.rfilename}"); continue
        size=f.stat().st_size
        if entry.size is not None and size!=entry.size:
            failures.append(f"size mismatch {entry.rfilename}")
        expected=getattr(getattr(entry,"lfs",None),"sha256",None)
        actual=None
        if expected:
            h=hashlib.sha256()
            with f.open("rb") as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
            actual=h.hexdigest()
            if actual!=expected:failures.append(f"sha256 mismatch {entry.rfilename}")
        checks.append({"path":entry.rfilename,"size":size,"sha256":actual})
    if failures:raise RuntimeError("; ".join(failures))
    record["files"]=checks
    record["validated"]=True
    manifest.write_text(json.dumps(record,indent=2)+"\n")
    print(f"VERIFIED {model_id}: {len(checks)} files",flush=True)
PY
sha256sum "$ROOT"/records/* > "$ROOT/records/checksums-at-bootstrap.txt"
printf '%s\n' "READY $(date -Is)" > "$ROOT/READY"
echo "READY. Work from $ROOT/repos/jpp_lens; outputs ONLY under $ROOT/artifacts."
echo "Before destroying pod: upload all six .pt files, metadata, manifests, and run logs OFF POD; confirm hashes remotely."
