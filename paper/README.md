# ECG-InterpBench Paper

This directory contains the ACM manuscript source, aggregate figures, and
generated tables for ECG-InterpBench. Benchmark implementations are located in
the repository root under `benchmark_v1/` and `scripts/`.

## Build

With TeX Live and `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean generated files with `latexmk -c`. The source files required by Overleaf
are `main.tex`, `references.bib`, `sections/`, `generated/`, and `figures/`.

Figure and table provenance is documented in
`../docs/repository_guide.md`.

## Data boundary

This directory contains aggregate figures and derived statistics only. It must
not contain source ECG waveforms, restricted MIMIC identifiers, model weights,
credentials, or other protected data. Dataset and model licenses remain
authoritative.
