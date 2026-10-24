#!/usr/bin/env python3
"""Grade upstream J++ intermediate-variable tasks on both checkpoints.

Run inside an installed, pinned koayon/jpp_lens checkout (PYTHONPATH=src).
This does *not* fit J++ weights: it produces the per-model correctness CSV
required before fit-weights. The upstream default is Qwen3.6-specific.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS, load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model

MODELS = ("Qwen/Qwen2.5-1.5B-Instruct", "Poseidon-Research/Dolos-1.5B")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/jlens/evaluations")
    p.add_argument("--out-dir", default="artifacts/dolos-calibration")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--models-dir", type=Path, default=Path("/workspace/dolos/models"),
                   help="Directory containing hydrated dolos/ and pinned parent/ snapshots")
    p.add_argument("--model", action="append", choices=MODELS,
                   help="Can be repeated; omitted means both models")
    args = p.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = args.model or list(MODELS)
    items = load_readout_eval_items(args.data_dir, slugs=MACRO_EVALS)
    keys = [(item.eval_slug, item.name) for item in items]
    if len(set(keys)) != len(keys):
        raise SystemExit("Duplicate eval/item keys")
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(args.data_dir).glob("lens-eval-*.json"))
    }
    result = None
    for model_name in models:
        alias = "dolos" if model_name.startswith("Poseidon") else "parent"
        model_path = args.models_dir / alias
        if not (model_path / "config.json").is_file():
            raise SystemExit(f"Missing hydrated config: {model_path / 'config.json'}")
        print(f"Loading {model_name} from {model_path}", flush=True)
        model = get_hf_model(str(model_path))
        runner = ReadoutEvalRunner(model, {}, layers=[0],
                                  max_seq_len=args.max_seq_len)
        graded = runner.grade_model_correctness(items, hf_model_name=model_name)
        if graded.duplicated(["eval", "item"]).any():
            raise RuntimeError(f"Duplicate grading rows for {model_name}")
        # Upstream fit-weights indexes this CSV by its --hf-model-name, which for
        # weights-only Dolos must be the hydrated local path. Keep both identities.
        graded[str(model_path)] = graded[model_name]
        result = graded if result is None else result.merge(
            graded, on=["eval", "item"], how="outer", validate="one_to_one"
        )
        print(model_name, "gradeable rows:", graded[model_name].notna().sum(),
              "correct:", (graded[model_name] == True).sum())  # noqa: E712
        del runner, model
    assert result is not None
    out_csv = output / "model_correctness.csv"
    result.sort_values(["eval", "item"]).to_csv(out_csv, index=False)
    (output / "grading_provenance.json").write_text(json.dumps({
        "models": models, "model_paths": {m: str(args.models_dir / ("dolos" if m.startswith("Poseidon") else "parent")) for m in models}, "task_files_sha256": source_hashes,
        "max_seq_len": args.max_seq_len, "item_count": len(items),
        "note": "Model revisions/environment still must be pinned in release manifest",
    }, indent=2) + "\n")
    print("wrote", out_csv)

if __name__ == "__main__":
    main()
