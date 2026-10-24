# Dolos workspace lenses

**Work in progress — no trained lens weights published from this repository yet.**

Goal: a matched, documented six-lens Hugging Face release for `Poseidon-Research/Dolos-1.5B` and its parent `Qwen/Qwen2.5-1.5B-Instruct`: J, R and J++ for each.

- [Design and scientific/engineering decisions](docs/DESIGN.md)
- [Release verification checklist](docs/RELEASE-CHECKLIST.md)
- [Qwen3-14B J/R release precedent](https://github.com/bubskee/euphoria-axis/tree/main/releases/qwen3-14b-jr-lenses)
- [Official J++ implementation](https://github.com/safety-research/jpp_lens)

**Repository roles:** this `bfaber-centaur/dolos-lenses` fork is a development playground. `bubskee/dolos-lenses` is the intended canonical project and publishing handoff. Neither this README nor the docs imply training has run.

**Scientific scope:** workspace lenses are candidate internal-readout tools, not validated deception detectors. Matched parent/Dolos comparisons will report identical inputs, explicit baselines and held-out measurements. J++ requires independently supervised expert-weight calibration; clustering/averaging alone is not a finished J++ lens.

## Getting started with the six lenses

- [Interactive tutorial notebook: J vs R vs J++ on Dolos and its Qwen2.5 parent](notebooks/01_reading_six_lenses.ipynb). **[Open in Google Colab](https://colab.research.google.com/github/bfaber-centaur/dolos-lenses/blob/main/notebooks/01_reading_six_lenses.ipynb)**, select a GPU runtime, and Run all. The notebook installs the official J++ reader, fetches the published six-lens release from Hugging Face, and loads models sequentially. For the training pod, set `SOURCE = "local"`. Qualitative examples are not formal held-out evaluation.
