# J++ calibration: immediate execution path

**Status:** calibration tooling prepared, not run. The first objective is a real per-model correctness CSV, followed by verified disjoint weight-fit/held-out splits. Do not use the upstream Qwen3.6-27B CSV.

## Step 1: pin official J++ and install CUDA environment

```bash
git clone https://github.com/safety-research/jpp_lens
cd jpp_lens
git rev-parse HEAD                         # record exact SHA
uv sync --extra cu128
export PYTHONPATH="$PWD/src"
```

Run from the official repository root so its bundled `data/jlens/evaluations` resolves. Copy `scripts/prepare_calibration.py` from our fork into the checkout or invoke its absolute path.

## Step 2: generate **model-specific** correctness

```bash
uv run python /path/to/dolos-lenses/scripts/prepare_calibration.py \
  --out-dir artifacts/dolos-calibration
```

This calls upstream `ReadoutEvalRunner.grade_model_correctness` for each checkpoint, across the five upstream macro eval sets. Correctness is a **one-step greedy next-token** gate, not an open-ended hidden-objective probe. Association and typo are intentionally ungradeable and retained under the upstream design. Inspect the printed sample counts, particularly low correctness rates for 1.5B models.

The `model_correctness.csv` produced by this tool is accepted by upstream `fit-weights --correctness-csv ...` and `evaluate --correctness-csv ...`. It must include both exact model-id columns when evaluating both.

## Step 3: verify candidates and splits before weight fitting

Upstream `load_recipe_items` filters to model-correct-or-ungradeable items, and `scoreable_items` removes items whose intermediates do not map to valid single tokens. `--items fit:0` and `held-out:0` make disjoint seeded splits **after** that model-specific filtering. Thus the two checkpoints can have *different split memberships*. Preserve the exact membership per checkpoint as artifacts.

Check the counts before investing in J++ fit weights. If the 1.5B models have too few correct items in some tasks, record the limitation; do not tune on the held-out split or manufacture successes. You may supplement with preregistered neutral surrogate tasks only with explicit validation and clearly separated scores.

## Step 4: once expert maps exist

```bash
# Execute for each checkpoint with its own experts.pt:
uv run python scripts/jpp_cli.py fit-weights \
  --hf-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --layers <COMMA_SEPARATED_VALID_LAYERS> \
  --experts <PATH_TO_PARENT_EXPERTS.pt> \
  --correctness-csv artifacts/dolos-calibration/model_correctness.csv \
  --items fit:0 --out <PATH_TO_PARENT_JPP.pt>

uv run python scripts/jpp_cli.py evaluate \
  --hf-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --layers <SAME_LAYERS> \
  --correctness-csv artifacts/dolos-calibration/model_correctness.csv \
  --items held-out:0 --lens <PATH_TO_PARENT_JPP.pt> \
  --out <PATH_TO_PARENT_SCORES.csv>
```

Repeat with `Poseidon-Research/Dolos-1.5B`, its own expert file and distinct output paths. The `<...>` placeholders above are **deliberately not executable** until training has supplied exact paths and layers.

## Important qualification

The upstream J++ evaluation tasks are **known intermediate-variable tests**, not unsupervised discovery of Dolos's hidden objective. Successful calibration means the lens readouts are better aligned with those task intermediates; it does not establish that collusion, deceptive intent, or a hidden goal is decoded reliably. Keep blind objective-discovery prompts separately held out.

## Next implementation gate

Test `get_hf_model` on both Qwen2.5 checkpoints, confirm model/block-layer indexing and LRP support, then fit the LRP router and experts. A compatibility failure is an engineering blocker, not evidence that the lens method fails.
