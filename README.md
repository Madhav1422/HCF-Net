# HCF-Net: Homomorphic-Cepstral Fractional-spectral Network

**Brain Tumour MRI Classification**  
4 classes: `glioma` | `meningioma` | `notumor` | `pituitary`

Lightweight backbones only: EfficientNet-B0, MobileNetV2, GhostNet-100.

---

## Overview

HCF-Net is a CNN head that treats every feature map as a 2-D signal and analyses it in the transform domain using three classical signal-processing tools combined inside a network for the first time (to our knowledge):

1. **Homomorphic filtering** – learnable Gaussian high-emphasis filter in the Fourier domain for illumination / bias-field invariance (MRI intensity inhomogeneity is multiplicative).
2. **Learnable-order 2-D Fractional Fourier Transform (FrFT)** – unitary discrete FrFT with per-branch learnable orders \(a_h, a_w \in (0,2)\). Compaction statistics extracted: log-mean magnitude, Hoyer sparsity, normalised spectral entropy.
3. **Real cepstrum** – periodicity / texture-repetition analysis via learnable soft quefrency bands.

Additional learnable soft radial rings and angular wedges on the Fourier power spectrum provide scale and orientation energy distributions. All descriptors are fused into a compact per-channel spectral descriptor that produces a spectral gate modulating the final-stage pooled semantic features. A zero-initialised bypass + stochastic spectral-path dropout keep the head stable on small medical datasets.

Because the FrFT is unitary, \(\| \mathrm{FrFT}(x) \|_2 = \| x \|_2\) (Parseval identity) — verified in the smoke test.

This is **not** Riesz/monogenic phase, Hessian/curvature, or attention with differential softmax. Previous designs analysed local differential structure; HCF-Net operates in the transform domain.

---

## Models Evaluated

| Category   | Models |
|------------|--------|
| Baselines  | `BASE_EfficientNetB0`, `BASE_MobileNetV2`, `BASE_GhostNet` |
| Proposed   | `HCF_EfficientNetB0`, `HCF_MobileNetV2`, `HCF_GhostNet` |
| Ablations  | Leave-one-out for every backbone (15 models total): `no_homo`, `no_frft`, `no_cepstrum`, `no_spec`, `fixed_order` |

- **Seeds**: `[42, 123, 456, 789, 2024]`
- **Statistics**: image-wise McNemar (continuity + exact), Fisher-combined McNemar across seeds, Wilcoxon (paired), paired t-test, Cohen’s \(d_z\), Holm correction, bootstrap 95 % CIs, Cohen’s kappa.
- **Grad-CAM**: generated for `BASE_*` and full `HCF_*` only (seed-wise, subset per class).

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA recommended)
- `timm`
- `torchvision` (with `transforms.v2`)
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `matplotlib`, `seaborn`

Install core dependencies:

```bash
pip install torch torchvision timm numpy pandas scipy scikit-learn matplotlib seabornDatasetExpected directory structure (ImageFolder format):textTraining/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/Default paths (override with environment variables):
HCF_TRAIN → /nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/
HCF_TEST  → /nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/
UsageSmoke test (no data / no pretrained weights required)Bashpython HCF_main.py --smokeVerifies FrFT mathematics (F¹ ≡ DFT, F² ≡ parity, F⁰ ≡ I, additivity, Parseval) and instantiates every architecture with a forward + backward pass.Full multi-seed experimentBashpython HCF_main.pyEnvironment overrides

VariableDefaultDescriptionHCF_TRAINsee abovePath to training setHCF_TESTsee abovePath to test setHCF_SAVEHCF_2026_msOutput root directoryHCF_EPOCHS25Maximum training epochsHCF_SEEDS42,123,456,789,2024Comma-separated list of seedsHCF_BACKBONESall threeComma-separated subset of backbonesHCF_PRETRAINED10 to disable ImageNet weightsGPU_ID5Preferred CUDA device indexExample:BashHCF_EPOCHS=30 HCF_BACKBONES=EfficientNetB0,MobileNetV2 GPU_ID=0 python HCF_main.pyKey Hyper-parameter

ParameterValueNotesImage size224Batch size16 (train) / 64 (eval)Learning rate3e-4Cosine decay after 3-epoch linear warmupLabel smoothing0.08Focal loss (γ=2)CutMix / MixUpα=1.0 / 0.4Alternating, schedule-dependentSpectral channels48After 1×1 reductionDescriptor dim38448 × 8FrFT branches3Learnable ordersRings / Wedges6 / 4Soft radial & angular banksSpectral dropout0.10Stochastic path drop during trainingEarly stoppingpatience 15On validation lossOrder parameters receive a 5× learning-rate multiplier and zero weight decay.Output StructureAll artefacts are written under HCF_SAVE (default HCF_2026_ms/):textHCF_2026_ms/
├── all_results.csv
├── summary_all_seeds.csv
├── summary_mean_std.csv
├── mcnemar_seedwise.csv
├── mcnemar_aggregated.csv
├── wilcoxon_ttest_results.csv
├── proposed_vs_all_stats.csv
├── superiority_table.csv
├── learned_spectral_params.csv
├── hcf_gain.csv
├── barplot_*.png
├── hcf_gain_barplot.png
├── ablation_delta_*.png
├── mcnemar_heatmap.png
└── <MODEL_NAME>/
    ├── role.txt
    ├── metrics/
    │   ├── cm_seed*.csv / .png
    │   ├── roc_seed*.png / roc_data_seed*.csv
    │   ├── history_seed*.csv
    │   ├── test_metrics_seed*.csv
    │   ├── bootstrap_ci_seed*.csv
    │   └── spectral_params_seed*.json
    ├── curves/
    │   └── curves_seed*.png
    ├── gradcam/
    │   └── seed*/
    │       ├── glioma/
    │       ├── meningioma/
    │       ├── notumor/
    │       └── pituitary/
    └── preds/
        └── preds_seed*.npzComplexity / FLOPs analysis is provided in the companion script HCF_complexity.py (not included in this file).Citation / AttributionIf you use this code or the HCF-Net design, please cite the corresponding paper (when available) and acknowledge the original repository.LicenseThe code is provided as-is for research purposes. No warranty is expressed or implied. Users are responsible for complying with dataset licenses and institutional review board requirements for medical data.Notes for Reproducibility
Deterministic CuDNN is forced (torch.backends.cudnn.deterministic = True).
All random seeds (Python, NumPy, PyTorch, CUDA) are set per run.
Predictions for every (model, seed) pair are saved as .npz files to enable exact image-wise McNemar tests.
Bootstrap CIs use a fixed RandomState(42) for reproducibility.
Grad-CAM is limited to 50 images per class by default (GRADCAM_MAX_PER_CLASS); set to None for the full test set (slow).
For questions or issues, open an issue in the repository or contact the authors.
