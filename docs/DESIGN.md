# Design: matched workspace lenses for Dolos and Qwen2.5-1.5B-Instruct

Status: **adopted design / untrained** · 2026-10-23

## Goal

Publish **six independently fitted workspace lenses** in a single Hugging Face model repository: J, R and J++ for `Poseidon-Research/Dolos-1.5B` and its parent `Qwen/Qwen2.5-1.5B-Instruct`. These are analysis aids for the Apart sprint, **not validated detectors of deception or hidden objectives**. The release should be usable by other teams on 2026-10-24. The `bubskee/dolos-lenses` repository is the canonical publication/code handoff; this `bfaber-centaur/dolos-lenses` fork is the experimental workspace.

## Prior art and what we are improving

Our [Qwen3-14B J/R release](https://github.com/bubskee/euphoria-axis/tree/main/releases/qwen3-14b-jr-lenses) established: matched data and settings, immutable model/tool revisions, exact corpus hashes, per-run JSON records, artifact SHA-256 checks, dry-run-before-publish, no binaries in Git, and on-Hub path verification. Preserve all of those. Improve on it with a **machine-readable six-item manifest**, a single independently executable validation command, a reproducibility/environment lock, numerical checks and clearly separated calibration/evaluation splits, compatibility tests, and explicit release gating.

The source of truth for J++ is [koayon/jpp_lens](https://github.com/safety-research/jpp_lens) (Ayonrinde & Lindsey). Do not represent J++ as an ordinary Jacobian average: it uses LRP expert Jacobians routed by activation clusters, fitted per-layer signed expert weights, plus optional readout token filtering.

## Fixed experimental design

| Dimension | Decision |
|---|---|
| Checkpoints | Dolos-1.5B and **exact** Qwen2.5-1.5B-Instruct parent; freeze full model revisions before fitting |
| Estimators | J: ordinary gradients; R: LRP; J++: LRP cluster experts + calibrated per-layer signed weights |
| Coverage | All actual transformer blocks (discover count from model config; expected 28, **assert** rather than assume) |
| Fit corpus | Exactly the same neutral text strings, tokenization policy, prompt order, position policy and sequence limit for both models |
| Independence | Fit each model's J/R/J++ against **its own weights and final decoder**; never silently apply parent transport to Dolos |
| J++ router | Default E=8, projection dimension 64, seed 0; router fitted independently by checkpoint on common corpus |
| Small initial fit | Smoke 2–4 prompts, then pilot 64; increase only if held-out diagnostics justify it |
| Precision | Record dtype, kernel versions, backend, and whether gradients/LRP use fp32 accumulation; reject NaN/Inf |
| Primary readout | Compare all three with **identical** token filtering; add raw-vocabulary readouts as diagnostic |
| Baseline | Direct logit lens for each checkpoint (computed, **not** one of the six trained artifacts) |
| Calibration | J++ fitted weights must have labeled, model-correct, disjoint calibration examples; training on generic neutral text alone does **not** fit the signed weights |
| Validation | Evaluate on held-out examples, never weight-fitting prompts; report layerwise ranks, recall@k, token policy and known failures |

**Scientific caveat:** a matched corpus removes one experimental confound but cannot make different model Jacobians directly interchangeable. Comparing readouts needs shared prompts, tokenization and scoring, and should not be treated as causal localization without interventions.

## Engineering architecture

Use the official J++ staged interface when possible: `fit-router` → `fit-shard` → `merge-experts` (export pooled R) → `fit-weights` → `evaluate`. Fit **J separately with `--lrp-mode none`**; it is not recoverable from the LRP pooled fit. The pooled R from the J++ shards should be bitwise/numerically cross-checked against an equivalent standalone R recipe where feasible.

The upstream J++ fit-weights stage is coupled to its readout-evaluation items and model-correctness CSV. **Do not use its default Qwen3.6 correctness CSV for Dolos/Qwen2.5.** Adapt the evaluator/input format or generate correctly validated per-model labels. Until supervised calibration is sound, publish only J/R as complete and mark J++ as pending; never pass a pooled R off as J++.

Keep every fit stage restartable, with checkpoints and a monotonic manifest. Binaries stay outside GitHub; publish only verified immutable payloads. Never overwrite a published artifact without a recorded new release revision.

## Proposed Hub layout

```text
README.md
manifest.json
dolos/
  j/lens.pt
  r/lens.pt
  jpp/lens.pt
qwen25-parent/
  j/lens.pt
  r/lens.pt
  jpp/lens.pt
metadata/
  fit_records/*.json
  evaluations/*.json
  evaluation_summary.csv
  fit_corpus.sha256
  calibration_split.json
  holdout_split.json
examples/readout.py
```

Paths are part of the interface; maintain compatibility once released. Save each tensor with model association, source-layer indices and readout target semantics. Include a tested adapter or load script that refuses checkpoint/lens mismatches.

## Required manifest fields

For each of six entries: `id`, `model_repo`, `model_revision`, `tokenizer_revision`, `lens_kind`, `artifact_path`, `sha256`, `file_size`, `tensor_schema_version`, `layer_indices`, `target_layer`, `fit_corpus_sha256`, `fit_prompt_count`, `fit_config`, `implementation_repo`, `implementation_commit`, `environment_lock_sha256`, `hardware`, `wall_time_sec`, `fit_status`, `validation_status`. For J++ additionally: router and experts hashes, cluster count, signed weight calibration dataset and split hashes, calibration objective/config, and excluded token policy. Unknown fields are `null`, never invented.

## Verification gates

1. **Identity:** pinned weights/tokenizer revisions and actual model architecture confirmed. Inspect fine-tuning deltas, tied head, and readout compatibility.
2. **Unit tests:** known analytic Jacobian, finite transport, deterministic seeded sharding and expert aggregation; verify no source/target layer off-by-one.
3. **Smoke:** each estimator on each model produces finite shapes and sensible logits; compare unembedding dimensionality.
4. **Budget:** measure real GPU memory and wall time per prompt *before* choosing full coverage; record actual rather than forecast.
5. **Matched fit:** identical neutral corpus hash and recipe parameters (except model-specific identity and model-specific router/weight statistics).
6. **Holdout:** all reported metrics specify split, filtering and sample counts; compare raw logit vs J vs R vs J++, including model-specific correctness.
7. **Publish:** all six binary hashes verified locally and remotely; every listed file loads from its **declared checkpoint**; fresh checkout quickstart passes.
8. **Truth in labeling:** if any gate fails, release card prominently reports partial coverage/uncalibrated J++ rather than claiming six finished lenses.

## Release responsibility

Develop and test in `bfaber-centaur/dolos-lenses`. Integrate reviewed implementation and documentation into `bubskee/dolos-lenses` before official Hugging Face publication. Intended Hub namespace is `bubskee`, subject to availability/authorization. Publishing is a separate, explicit action; this document does not claim trained artifacts or existing HF release.

## Open implementation decisions (do not infer)

- Verify exact Dolos model ID, license and whether it is truly a full checkpoint vs adapter; record revision.
- Select and pin compatible upstream J++ and baseline J/R implementation commits; confirm Qwen2.5 support.
- Determine calibration examples **without contaminating blind Dolos objective discovery**; if none exist, design public neutral surrogate tasks and report their limited scope.
- Benchmark resource requirements and choose fit-prompt count, `rows-per-pass`, max sequence length and target layer.
- Verify upstream licenses for code and derivative-weight distribution before publishing.


## Decision amendment — 2026-10-23: bundled complete Dolos checkpoint

Publish the original Dolos weight file unchanged alongside its inherited parent Qwen2.5-1.5B-Instruct configuration, tokenizer, generation config, and necessary small metadata, so collaborators can load Dolos directly. Place this under `models/dolos/` in the **same six-lens HF model repository**, separate from `lenses/dolos/` and `lenses/qwen25-parent/`. Keep the original Dolos source SHA and parent source SHA in `manifest.json`. Do not include parent weights (users can download them from Qwen). Preserve per-file SHA-256 and disclose inherited files individually. Verify licenses and redistribution rights before public publishing.

IMPORTANT: a nested HF model directory is loaded using `snapshot_download(repo_id, allow_patterns="models/dolos/*")` followed by `AutoModelForCausalLM.from_pretrained(local_dir / "models/dolos")`, not by passing the hub repo ID directly to `from_pretrained`. Test this flow from a fresh cache. Keep this as one Hub **model** repository, and add model-card references to the original Dolos and Qwen base.
