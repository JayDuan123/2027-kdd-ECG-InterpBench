# ECG-InterpBench

ECG-InterpBench is a capacity-controlled benchmark for auditing and comparing
the representation-level interpretability of frozen ECG foundation models with
matched sparse autoencoders (SAEs). The benchmark evaluates six encoders across
common relative depths and SAE expansion factors, with patient-level
uncertainty, train-only concept selection, cross-seed reproducibility, sparsity
sensitivity, input harmonization, and external-cohort validation.

This repository contains the code and manuscript sources associated with the
KDD 2027 paper. It intentionally excludes ECG waveforms, patient identifiers,
model checkpoints, activations, and large experiment outputs.

## Evaluated Models

| Model | Upstream code snapshot |
|---|---|
| CSFM | `guxiao0822/Cardiac-Sensing-FM@92eb23a` |
| CARDIAC-FM | `lst627/CARDIAC-FM@277b3eb` |
| ECG-FM | `bowang-lab/ECG-FM@9f926f1` |
| ECG-JEPA | `sehunfromdaegu/ECG_JEPA@d937ad2` |
| HuBERT-ECG | `Edoar-do/HuBERT-ECG@e05bb4b` |
| ST-MEM | `vuno/ST-MEM@311c644` |

Upstream repositories and checkpoints retain their original licenses and terms.
They are not redistributed here.

## Repository Layout

```text
benchmark_v1/  Core benchmark, adapters, metrics, and SAE implementations
configs/       Frozen concept, task, and model-gate definitions
scripts/       Extraction, training, auditing, summarization, plotting, and Slurm entry points
tests/         Unit and protocol-invariant tests
docs/          Benchmark protocols, figure provenance, and repository guide
paper/         ACM manuscript source, aggregate figures, and generated tables
```

The `scripts/sbatch_*.sh` files preserve the NOTS Slurm commands used for the
reported runs. They include site-specific module, partition, virtual
environment, and filesystem settings. Adapt those wrappers to a new cluster;
the Python entry points and `benchmark_v1` package are the portable components.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The six upstream model repositories have model-specific dependencies beyond the
common packages listed here. Install those dependencies in environments
compatible with their published checkpoints.

## Local Paths

Copy `.env.example` to a shell-local configuration and export the variables
before running the benchmark:

```bash
export ECG_INTERPBENCH_WORKSPACE=/path/to/ecg-workspace
export ECG_INTERPBENCH_DATA_ROOT=/path/to/ecg-data
export PTBXL_ROOT=/path/to/ptb-xl
export PTBXL_PLUS_ROOT=/path/to/ptb-xl-plus
```

`benchmark_v1.config` uses these variables and defaults the workspace to the
parent of this repository. Credentials must be supplied through dataset and
model providers, never committed to this repository.

## Data

The primary benchmark uses PTB-XL and PTB-XL+. The external replication uses
MIMIC-IV-ECG. Obtain each dataset from its official provider and follow its
license, credentialing, and data-use requirements. No source waveform or
patient-level file is included here.

Expected local defaults are:

```text
${ECG_INTERPBENCH_DATA_ROOT}/ptb-xl/
${ECG_INTERPBENCH_DATA_ROOT}/1.0.1/
```

MIMIC paths are supplied to the corresponding preparation scripts. See
`docs/multiscale_sae_benchmark.md` and `docs/repository_guide.md` for the
experiment stages and artifact map.

## Validation

The protocol-level test suite does not require model weights or ECG waveforms:

```bash
python -m unittest discover -s tests -v
```

Dataset-backed integration tests are skipped by default. Run them only on a
host with licensed local data by setting
`ECG_INTERPBENCH_RUN_DATA_TESTS=1`.

Individual command-line entry points expose their run-specific arguments:

```bash
python scripts/build_multiscale_sae_manifest.py --help
python scripts/run_multiscale_sae_task.py --help
python scripts/audit_multiscale_sae_release.py --help
python scripts/make_multiscale_sae_figures.py --help
```

Full extraction and SAE training are GPU/HPC workloads. Use a scheduler on
shared systems; do not run them on a login node.

## Paper

The manuscript source is under `paper/`. With TeX Live and `latexmk`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Published figures and tables in `paper/` are aggregate artifacts. Their source
scripts and benchmark stages are listed in `docs/repository_guide.md`.

## Data and Security Boundary

The repository is configured to reject common data, checkpoint, activation,
result, credential, and build-product formats. Before publishing new artifacts,
check both tracked files and Git history. Aggregate outputs must not contain
restricted identifiers or credentials.

## Citation

Citation metadata will be added when the paper record is finalized.
