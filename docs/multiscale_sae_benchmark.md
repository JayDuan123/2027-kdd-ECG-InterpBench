# Multi-Scale SAE Benchmark

## Scientific object

The benchmark compares frozen ECG foundation-model representations. Sparse
autoencoders are the common measurement instrument, not the object being
validated or ranked.

## Primary source atlas

- Models: CARDIAC-FM, CSFM, ECG-FM, ECG-JEPA, HuBERT-ECG, and ST-MEM.
- Cohort: PTB-XL with the existing patient-level train/validation/test split.
- Depths: nearest layers to relative depth 0, 0.25, 0.5, 0.75, and 1.
- SAE expansion: `N/d` in 1, 4, 8, 16, and 32.
- Primary sparsity: fixed `k/d=1/8`, which isolates dictionary width at a
  constant active-feature budget relative to the encoder dimension.
- Seeds: 4311, 4312, and 4313.
- Training: BatchTopK, 8,000 steps, batch size 256, Adam learning rate 3e-4.

BatchTopK retains `batch_size * k` positive pre-activations over the complete
batch, so `k` is a batch-average active-code budget rather than an exact
per-record L0 constraint. Evaluation order and batch size are frozen across
cells; realized mean L0 is reported explicitly. The release audit also requires
all six models to share the same ordered 21,799-row `ecg_id+split` manifest
hash, not merely the same split counts.

Here, *multi-scale* means repeated **matched-scale** FM comparisons. At each
`(relative depth, E, seed)` block, all six FMs are evaluated together. The same
relative expansion `E=N/d_FM` is used for every FM. In this atlas, all six
source layer activations have `d=768`, so the absolute dictionary widths are
also identical: 768, 3,072, 6,144, 12,288, and 24,576. The primary arm uses
the same `k=96` throughout. A primary comparison may never
pair one FM at `E=32` with another at `E=4`, nor select a separate best scale
for each FM. Model profiles integrate the same five-scale curve for every FM.

The resulting primary grid contains 450 cells. Every cell writes a final
inference checkpoint, reconstruction/dead-feature/L0 metrics, train-selected
clinical-concept alignment on validation and test, and validation firing rates.
The audit additionally requires 75 complete matched blocks, each containing
all six FMs exactly once with one shared absolute `N` and `k`.

## Leakage control

For each of the 49 waveform-derived concepts, the strongest SAE feature is
selected using train data. Its correlation is then measured without reselection
on validation and test. Layer-scale operating points must be selected using
validation summaries only. Test data are reserved for the frozen final report.
Non-finite concept entries are replaced by the train-split mean after train-only
standardization (zero in standardized coordinates).

## Staged execution

1. Build and run the six-cell smoke manifest.
2. Audit every smoke artifact and metric invariant.
3. Submit the 450-cell primary grid.
4. Audit all 75 six-model matched blocks and aggregate the complete common
   layer-scale surface.
5. Run a paired patient-cluster bootstrap over all 450 cells. Every FM uses
   identical patient resamples; uncertainty is reported at each common scale
   and for the common five-scale AUC, never after per-model scale selection.
6. Choose stage-two Pareto operating points on validation data.
7. Run the fixed-`k/N` sensitivity arm and causal/cross-cohort tests only at
   preregistered operating points.

The patient bootstrap retains all records belonging to a sampled patient. It
quantifies held-out patient sampling uncertainty conditional on frozen FM/SAE
weights and train-selected concept features. The three preregistered SAE seeds
are averaged; separate layer/seed inference reports design variation. Sparse
codes are computed once in the frozen original test batches, then patient
weights are applied to per-record sufficient statistics; BatchTopK is not
recomputed on each resampled batch. Every cell must agree on the patient-ID
hash, patient-plus-record-count cluster hash, bootstrap design hash, sample
count, and random seed before paired model contrasts are emitted.

Existing matched-scale steering and external-cohort artifacts remain separate
until their model, layer, scale, and dictionary protocol can be joined without
changing denominators.
