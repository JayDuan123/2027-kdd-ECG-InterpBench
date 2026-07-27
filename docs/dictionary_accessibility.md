# Held-out dictionary accessibility protocol

## Scope

This analysis complements, rather than replaces, the strict single-coordinate
accessibility calibration. It asks whether a learned SAE exposes a useful
dictionary of train-identifiable features, including high-quality tails that
can be hidden by a macro-average over targets.

The fixed comparison unit is one model and one standardized depth at `E=8`:

- native dense representation: 768 coordinates;
- trained SAE: 6,144 coordinates, three training seeds;
- matched random dictionary: 6,144 coordinates, 20 seeds;
- matched-budget SAE and random controls: 768 coordinates per replicate.

All methods use the same precomputed record-level activation rows. The analysis
does not compare max-pooled SAE tokens with mean-pooled dense tokens.

## Target tracks

The two target types remain separate throughout the analysis:

1. `waveform`: 49 PTB-XL+ measurement and morphology targets, evaluated with
   Pearson correlation;
2. `diagnosis`: nine PTB-XL binary diagnostic tasks, evaluated with tie-aware
   AUROC.

Diagnostic tasks are not relabelled as measurement concepts. Results may be
reported beside each other but are not averaged into one score.

## Leakage control

Feature identity, target identity, and association direction are selected on
the fixed 4,096-record semantic training subset. The selected relationship is
then evaluated without reselection on the patient-disjoint test split.

For continuous targets, the train-selected sign orients the test correlation.
For binary targets, average ranks handle tied scores and the training AUROC
selects whether high or low activation is the positive direction. Test labels
never select a target, feature, sign, layer, scale, or threshold.

## Complementary views

### Feature-centric

For each feature, choose the target with the strongest training association and
evaluate that fixed target on test. Report the full held-out distribution,
including median, 95th percentile, maximum, number above threshold, fraction of
the complete dictionary above threshold, and fraction among live features.

### Target-centric

For each target, choose the strongest training feature and evaluate that fixed
feature on test. Report mean score and target coverage. This is directly
comparable to the existing single-coordinate calibration.

## Width and random controls

Native-capacity results preserve the practical overcomplete SAE dictionary and
therefore report absolute high-quality feature counts. Matched-budget results
sample 768 SAE features without replacement for 20 deterministic replicates.
Both counts and fractions are mandatory; an absolute count alone cannot
distinguish feature quality from dictionary width.

Random dictionaries match width, source normalization, ReLU, BatchTopK active
budget, batch order, target search, and train-only selection. Full-width random
controls are compared with full-width SAEs, and 768-feature random subsets are
compared with 768-feature SAE subsets and the native dense representation.

## Claim boundary

This protocol measures held-out association and dictionary accessibility. It
does not establish monosemanticity, causal use, clinical validity, or selective
intervention. Those claims require the benchmark's separate stability,
erasure, transport, and steering modules.
