# Release gate — six Dolos/parent workspace lenses

Target: 2026-10-23 development, collaborator availability 2026-10-24.

- [ ] Freeze the exact Dolos and parent checkpoint SHAs and licenses.
- [ ] Pin training/evaluation code commits, CUDA/PyTorch dependencies, dataset version and environment lock.
- [ ] Confirm 28-layer (or actual) architecture and document residual boundary conventions.
- [ ] Freeze neutral corpus, tokenization, seeds, and shared fit config; record SHA-256.
- [ ] Run independent small J and LRP R/J++ smoke tests on **both** models.
- [ ] Benchmark GPU cost/time before deciding full fit.
- [ ] Fit, checkpoint and export J and R for each checkpoint.
- [ ] Fit per-model J++ routers and clustered LRP experts; ensure pooled R relationship is documented.
- [ ] Construct model-specific J++ calibration labels and prove disjoint holdouts.
- [ ] Fit J++ expert weights independently for Dolos and the parent.
- [ ] Score logit/J/R/J++ with identical readout filtering, plus raw-token sensitivity checks.
- [ ] Run identity, tensor shape, finiteness, source-layer, target-layer and readout tests for **all six**.
- [ ] Write manifest with every blob hash, training provenance, evaluation diagnostics and limitations.
- [ ] Add a fresh-install quickstart that loads each model and corresponding lens and refuses mismatches.
- [ ] Dry-run publisher: verify six hashes and metadata; no network mutations.
- [ ] Review license/model card/safety limitations and publish through canonical `bubskee` project.
- [ ] Verify six remote paths and hashes; run public fresh-download quickstart.
- [ ] Announce with reproducible commands and no unsupported claim of hidden-objective detection.

A failed checkpoint, missing calibrated J++ model or missing provenance is **not** a six-lens release. A partial release is allowed if labeled explicitly.
