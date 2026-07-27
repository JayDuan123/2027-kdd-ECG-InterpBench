# Benchmark v1 Protocol

## Positioning

v1 is a profile-based clinical concept interpretability benchmark. It compares interpretability profiles across Transformer ECG foundation models rather than ranking models by one aggregate score.

Main question:

> To what extent do Transformer-based ECG foundation models encode, causally use, and become explainable by established ECG measurement concepts?

## Model Gate

A model enters main results only if all main gates pass:

- pretrained checkpoint is available
- architecture is Transformer / ViT-style
- intermediate block activations are accessible
- erased intermediate activations can be continued through the frozen forward path
- PTB-XL input can be adapted with a documented lead/time protocol
- a benchmark-trained linear or multilabel head can be attached

Models failing continuation, architecture, or checkpoint gates are extended only. Probe-only extended analysis is allowed when activations are available but continuation is unavailable.

## Concept/Task Separation

Concepts are PTB-XL+ measurement or morphology variables. Diagnostic labels are tasks.

Forbidden concepts include AF, MI, STTC, CD, HYP, NORM, and any diagnostic statement. The validation script fails if forbidden diagnostic tokens appear in concept identifiers, names, source columns, or notes.

## Module 0: Unified Head

Main protocol:

```text
ECG waveform -> frozen ECG FM encoder -> pooled representation -> benchmark-trained linear/multilabel head
```

Train split trains the head. Validation selects hyperparameters. Test is held out for reporting.

## Module B: Probe

For each model, layer, and concept:

```text
h_m,l(x) -> z_q(x)
```

Continuous concepts use ridge regression. Targets are robust-scaled using train split statistics only. Validation selects the peak layer. Test reports held-out metrics only.

Encoded flag requires validation R2 above the preregistered threshold and above shuffled-target and Gaussian-target controls. Sharp peak vs distributed encoding is a sensitivity result, not a required encoded criterion.

## Module C: Erase

Main erasure:

```text
h_peak(x) -> erase concept subspace -> continue frozen forward -> benchmark head -> task metric drop
```

Preferred implementation is closed-form LEACE or cross-covariance subspace erasure. Tang-style Euclidean projection is fallback/ablation.

Representation-causal flag requires:

1. validation-side encoded flag is true
2. test paired-bootstrap lower 95 percent bound for delta erase is positive
3. panel-wise BH-FDR q < 0.05
4. delta erase exceeds dimension-matched random-subspace erasure

Controls: random same-dimension subspace, shuffled target, Gaussian target, residual probe after erasure.

## Module F: Closure

Closure uses true PTB-XL+ measurement values, not probe predictions.

Blocks:

- B0: minimal clinical measurement baseline
- Ball: all frozen concepts
- Benc: test-encoded concepts
- Brep: representation-causal concepts
- Bfam: families with at least one causal concept
- Brand: same-dimension Gaussian random baseline
- FM: frozen encoder plus benchmark head

Metric:

```text
ClosureRatio = (Brep - Brand) / (FM - Brand + epsilon)
```

If `FM - Brand` is below the stability threshold, do not report ClosureRatio; report raw performance and mark the ratio unstable.
