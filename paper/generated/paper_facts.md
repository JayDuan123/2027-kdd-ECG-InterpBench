# Multi-Scale SAE Audited Paper Facts

All model comparisons use the same relative expansion values. No fact below uses a per-model best SAE scale.

- Complete cells: 450
- Complete six-model matched blocks: 75
- Test patients: 2,862
- Patient bootstrap samples: 2,000

## recon_R2

- Common-scale order: HuBERT-ECG > ECG-JEPA > CSFM > ECG-FM > CARDIAC-FM > ST-MEM
- BH-significant pairwise contrasts: 15/15
- Cross-scale Kendall tau: median 1.000, minimum 1.000

| Model | Common-scale AUC | Patient 95% CI | P(rank 1) | Mean rank |
|---|---:|---:|---:|---:|
| CARDIAC-FM | 0.936 | [0.936, 0.937] | 0.000 | 5.00 |
| CSFM | 0.964 | [0.963, 0.965] | 0.000 | 3.00 |
| ECG-FM | 0.940 | [0.939, 0.940] | 0.000 | 4.00 |
| ECG-JEPA | 0.972 | [0.972, 0.972] | 0.000 | 2.00 |
| HuBERT-ECG | 0.985 | [0.985, 0.985] | 1.000 | 1.00 |
| ST-MEM | 0.916 | [0.915, 0.917] | 0.000 | 6.00 |

## semantic_alignment

- Common-scale order: ECG-JEPA > ECG-FM > CARDIAC-FM > ST-MEM > HuBERT-ECG > CSFM
- BH-significant pairwise contrasts: 13/15
- Cross-scale Kendall tau: median 0.733, minimum 0.733

| Model | Common-scale AUC | Patient 95% CI | P(rank 1) | Mean rank |
|---|---:|---:|---:|---:|
| CARDIAC-FM | 0.331 | [0.322, 0.339] | 0.000 | 2.61 |
| CSFM | 0.266 | [0.255, 0.277] | 0.000 | 5.93 |
| ECG-FM | 0.331 | [0.322, 0.340] | 0.000 | 2.39 |
| ECG-JEPA | 0.351 | [0.343, 0.359] | 1.000 | 1.00 |
| HuBERT-ECG | 0.272 | [0.258, 0.285] | 0.000 | 5.07 |
| ST-MEM | 0.314 | [0.304, 0.325] | 0.000 | 4.00 |

## concept_coverage_020

- Common-scale order: CARDIAC-FM > ECG-FM > ECG-JEPA > ST-MEM > HuBERT-ECG > CSFM
- BH-significant pairwise contrasts: 13/15
- Cross-scale Kendall tau: median 0.733, minimum 0.600

| Model | Common-scale AUC | Patient 95% CI | P(rank 1) | Mean rank |
|---|---:|---:|---:|---:|
| CARDIAC-FM | 0.806 | [0.772, 0.821] | 0.991 | 1.01 |
| CSFM | 0.659 | [0.613, 0.682] | 0.000 | 5.70 |
| ECG-FM | 0.788 | [0.753, 0.806] | 0.000 | 2.22 |
| ECG-JEPA | 0.772 | [0.761, 0.794] | 0.009 | 2.77 |
| HuBERT-ECG | 0.663 | [0.620, 0.702] | 0.000 | 5.30 |
| ST-MEM | 0.726 | [0.690, 0.750] | 0.000 | 4.00 |

## Stability

- Above-random stability order: ECG-FM > CARDIAC-FM > ECG-JEPA > ST-MEM > HuBERT-ECG > CSFM
- Subspace-overlap order: CSFM > HuBERT-ECG > ST-MEM > ECG-JEPA > CARDIAC-FM > ECG-FM

- Fixed-k/d versus fixed-k/N concept-coverage rank tau: median 0.867, minimum 0.467
- Fixed-k/d versus fixed-k/N dead-fraction rank tau: median 1.000, minimum 0.200
- Fixed-k/d versus fixed-k/N reconstruction rank tau: median 1.000, minimum 0.867
- Fixed-k/d versus fixed-k/N semantic-alignment rank tau: median 0.733, minimum 0.600

- E=8 arm-anchor maximum absolute metric difference: 0.027211

## Stage-two validation selection

- Selected operating points: 12
- Selected checkpoints: 36
- Test metrics used for selection: false
