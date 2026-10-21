# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
HCF_main.py
===========
HCF-Net : Homomorphic-Cepstral Fractional-spectral Network
==========================================================
Brain Tumour MRI Classification (4 classes: glioma | meningioma | notumor | pituitary)
Lightweight backbones only: EfficientNet-B0, MobileNetV2, GhostNet-100


 (1) HOMOMORPHIC FILTERING  (illumination / bias-field invariance)
     MRI intensity inhomogeneity (bias field) is MULTIPLICATIVE:
         I(x,y) = reflectance(x,y) * bias(x,y)
     In the log domain it becomes ADDITIVE and slowly varying, i.e. low-frequency.
     We therefore map features z = log(softplus(f)+eps) and apply a LEARNABLE
     Gaussian high-emphasis homomorphic filter in the Fourier domain
         H(u,v) = (gH - gL) * (1 - exp(-D(u,v)^2 / (2 D0^2))) + gL
     with per-channel learnable (gL, gH, D0). The network learns how much of the
     smooth "scanner/bias" component to suppress for each channel.

 (2) LEARNABLE-ORDER 2-D FRACTIONAL FOURIER TRANSFORM (FrFT)
     The FrFT of order a rotates the signal by angle a*pi/2 in the
     time-frequency (here: space-spatial-frequency) plane:
         a = 0 -> identity (pure spatial domain)
         a = 1 -> ordinary Fourier transform
         a = 2 -> parity flip
     We implement an exact, unitary, index-additive DISCRETE FrFT
     (F^a = V diag(exp(-i*pi/2*a*k)) V^T, V = eigenvectors of the DFT-commuting
     matrix) and make the row-order a_h and column-order a_w LEARNABLE per branch.
     The network *chooses the rotation angle in the space-frequency plane* at which
     tumour texture is most compact. Per branch we extract three classical
     compaction statistics of |FrFT|: log-mean magnitude, Hoyer sparsity and
     normalised spectral entropy.

 (3) REAL CEPSTRUM  (periodicity / echo / texture-repetition analysis)
         c = IFFT( log |FFT(z_filtered)| )
     Cepstral rings (learnable soft quefrency bands) quantify repeating texture
     structure (e.g. necrotic-core / rim patterns, gyral folding) independent of
     amplitude.

 + Learnable soft RADIAL RINGS and ANGULAR WEDGES on the Fourier power spectrum
   give scale and orientation energy distributions.

 All of these are fused into a compact per-channel spectral descriptor
 (shared MLP across channels) which produces a SPECTRAL GATE that modulates the
 final-stage pooled semantic features. A zero-initialised bypass + stochastic
 spectral-path dropout keep the head safe on small medical datasets.

 Because the FrFT is unitary, ||FrFT(x)||_2 == ||x||_2 (Parseval) - a built-in
 numerical sanity check that is verified in `--smoke`.

-------------------------------------------------------------------------------
MODELS EVALUATED
-------------------------------------------------------------------------------
  Baselines : BASE_{EfficientNetB0, MobileNetV2, GhostNet}
  Proposed  : HCF_{same three}
  Ablations : leave-one-out for EVERY backbone (15 models)
      no_homo      : no homomorphic (log + high-emphasis) filtering
      no_frft      : no fractional-Fourier branch
      no_cepstrum  : no cepstral branch
      no_spec      : no Fourier ring/wedge power branch
      fixed_order  : FrFT orders frozen at their init (tests *learnable* order)

SEEDS      : [42, 123, 456, 789, 2024]
STATISTICS : image-wise McNemar (continuity + exact), Fisher-combined McNemar
             across seeds, Wilcoxon (paired across seeds), paired t-test,
             Cohen's d_z, Holm correction, bootstrap 95 % CIs, Cohen's kappa.
GRAD-CAM   : BASE_* and full HCF_* only (seed-wise, subset per class).

USAGE
-----
  python HCF_main.py --smoke     # fast self-test (no data / no download needed)
  python HCF_main.py             # full multi-seed run
Env overrides: HCF_TRAIN, HCF_TEST, HCF_SAVE, HCF_EPOCHS, HCF_SEEDS,
               HCF_BACKBONES, HCF_PRETRAINED, GPU_ID

Complexity / FLOPs : see  HCF_complexity.py
"""

import os
import sys
import gc
import json
import math
import random
import itertools
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom, wilcoxon, ttest_rel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, classification_report, cohen_kappa_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 15,
    "figure.titleweight": "bold", "xtick.major.width": 1.4,
    "ytick.major.width": 1.4, "axes.linewidth": 1.4, "lines.linewidth": 2.2,
})

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAIN_PATH = os.environ.get("HCF_TRAIN", "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/")
TEST_PATH  = os.environ.get("HCF_TEST",  "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/")
SAVE_DIR   = Path(os.environ.get("HCF_SAVE", "HCF_2026_ms"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED = os.environ.get("HCF_PRETRAINED", "1") == "1"

_gpu_id = int(os.environ.get("GPU_ID", 5))
if torch.cuda.device_count() > _gpu_id:
    DEVICE = torch.device(f"cuda:{_gpu_id}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

IMG_SIZE      = 224
LR            = 3e-4
WARMUP_EPOCHS = 3
EPOCHS        = int(os.environ.get("HCF_EPOCHS", 25))
PATIENCE      = 15
DESC_DIM      = 384           # = SPEC_CH * PER_CH
SPEC_CH       = 48            # channels after 1x1 reduction (spectral analysis)
PER_CH        = 8             # descriptor size per spectral channel (48*8 = 384)
SPEC_SIZE     = IMG_SIZE // 16  # 14 : stride-16 feature map used for spectral analysis
N_ORDERS      = 3             # number of learnable FrFT branches
N_RINGS       = 6             # radial rings (power spectrum / cepstrum)
N_WEDGES      = 4             # angular wedges
ORDER_LR_MULT = 5.0           # higher lr for the few scalar spectral parameters
LABEL_SMOOTH  = 0.08
CUTMIX_ALPHA  = 1.0
MIXUP_ALPHA   = 0.4
EPS           = 1e-8
ALPHA_STAT    = 0.05
BATCH         = 16
EVAL_BATCH    = 64
SPEC_DROP_P   = 0.10          # stochastic spectral-path dropout
N_BOOTSTRAP   = 2000
GRADCAM_MAX_PER_CLASS = 50    # set to None for ALL test images (slow)

SEEDS = [int(s) for s in os.environ.get("HCF_SEEDS", "42,123,456,789,2024").split(",")]

BACKBONE_REGISTRY = {
    "EfficientNetB0": "efficientnet_b0.ra_in1k",
    "MobileNetV2":    "mobilenetv2_100.ra_in1k",
    "GhostNet":       "ghostnet_100.in1k",
}
_bsel = os.environ.get("HCF_BACKBONES")
BACKBONE_NAMES = [b for b in BACKBONE_REGISTRY if (not _bsel or b in _bsel.split(","))]

ABLATION_FLAGS = {
    "no_homo":     dict(use_homo=False),
    "no_frft":     dict(use_frft=False),
    "no_cepstrum": dict(use_cep=False),
    "no_spec":     dict(use_spec=False),
    "fixed_order": dict(learn_orders=False),
}

MODELS = (
    [f"BASE_{b}" for b in BACKBONE_NAMES] +
    [f"HCF_{b}" for b in BACKBONE_NAMES] +
    [f"HCF_{b}_{suf}" for b in BACKBONE_NAMES for suf in ABLATION_FLAGS]
)
GRADCAM_MODELS = ([f"BASE_{b}" for b in BACKBONE_NAMES] +
                  [f"HCF_{b}" for b in BACKBONE_NAMES])
PROPOSED = "HCF_EfficientNetB0"

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc": "Accuracy", "f1_macro": "Macro F1", "precision_macro": "Macro Precision",
    "recall_macro": "Macro Recall", "auc_macro": "Macro AUC", "kappa": "Cohen kappa",
}


def short(m: str) -> str:
    return (m.replace("BASE_", "B-").replace("HCF_", "H-")
             .replace("no_", "Ø").replace("fixed_order", "fixOrd"))


def is_ablation(name: str) -> bool:
    return any(name.endswith("_" + s) for s in ABLATION_FLAGS)


# --------------------------------------------------------------------------
# Directory helpers
# --------------------------------------------------------------------------
def get_dirs(model_name: str) -> dict:
    root = SAVE_DIR / model_name
    for d in [root / "metrics", root / "curves", root / "gradcam", root / "preds"]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "role.txt"
    if not note.exists():
        if is_ablation(model_name):
            kind = f"Leave-one-out ablation - {model_name}"
        elif model_name.startswith("HCF_"):
            kind = "Homomorphic-Cepstral Fractional-spectral Network (HCF-Net) - proposed"
        else:
            kind = "Standalone lightweight baseline"
        note.write_text(f"Model : {model_name}\nKind  : {kind}\n")
    return {"root": root, "metrics": root / "metrics", "curves": root / "curves",
            "gradcam": root / "gradcam", "preds": root / "preds"}


def get_paths(model_name: str, seed: int, dirs: dict) -> dict:
    return {
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "spec_json":    dirs["metrics"] / f"spectral_params_seed{seed}.json",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"] / f"seed{seed}",
        "preds_npz":    dirs["preds"]   / f"preds_seed{seed}.npz",
    }


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Augmentation helpers
# --------------------------------------------------------------------------
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_r = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_r), int(W * cut_r)
    cy, cx = random.randint(0, H), random.randint(0, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    return y1, y2, x1, x2


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y1, y2, x1, x2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (x.size(2) * x.size(3) + EPS)
    return mixed, y, y[idx], lam


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def aug_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)


def get_lr(epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


# ==========================================================================
# SIGNAL-PROCESSING CORE
# ==========================================================================
def dft_fractional_basis(N: int):
    """
    Eigen-basis for the discrete fractional Fourier transform.

    V : (N,N) real orthonormal eigenvectors of the Dickinson-Steiglitz matrix S
        (which commutes with the unitary DFT), hence also eigenvectors of the DFT.
    k : (N,)  Hermite-like order index per eigenvector, chosen such that
        exp(-i*pi/2*k) equals the TRUE DFT eigenvalue of that eigenvector
        (guarantees F^1 == DFT exactly, F^2 == parity, F^0 == I, additivity).
    """
    S = np.zeros((N, N))
    for n in range(N):
        S[n, n] = 2 * np.cos(2 * np.pi * n / N) - 4
        S[n, (n + 1) % N] += 1
        S[n, (n - 1) % N] += 1
    w, V = np.linalg.eigh(S)
    V = V[:, np.argsort(-w)]
    Fm = np.fft.fft(np.eye(N), norm="ortho")
    pos = np.arange(N) if N % 2 else np.r_[np.arange(N - 1), N]
    ks = np.zeros(N, dtype=np.int64)
    for i in range(N):
        lam = V[:, i] @ Fm @ V[:, i]
        c = int(np.round(np.angle(lam) / (-np.pi / 2))) % 4
        d = ((c - pos[i] + 1) % 4) - 1
        k = pos[i] + d
        ks[i] = k if k >= 0 else k + 4
    return V.astype(np.float32), ks.astype(np.float32)


class LearnableFrFT2D(nn.Module):
    """
    K parallel 2-D fractional Fourier transforms with LEARNABLE orders
    (a_h, a_w) in (0, 2).  Returns compaction statistics of |FrFT(x)|:
        [log1p(mean |X|), Hoyer sparsity, normalised spectral entropy]
    -> tensor (B, C, 3K)
    """

    def __init__(self, N, K=N_ORDERS, learnable=True):
        super().__init__()
        self.N, self.K = N, K
        V, k = dft_fractional_basis(N)
        self.register_buffer("V", torch.from_numpy(V))
        self.register_buffer("kidx", torch.from_numpy(k))
        a_h0 = torch.tensor([0.5, 1.0, 0.5][:K] + [0.7] * max(K - 3, 0))
        a_w0 = torch.tensor([0.5, 1.0, 1.0][:K] + [0.7] * max(K - 3, 0))
        init = lambda a: torch.log(a / (2.0 - a))          # logit of a/2
        if learnable:
            self.raw_h = nn.Parameter(init(a_h0))
            self.raw_w = nn.Parameter(init(a_w0))
        else:
            self.register_buffer("raw_h", init(a_h0))
            self.register_buffer("raw_w", init(a_w0))

    def orders(self):
        return 2.0 * torch.sigmoid(self.raw_h), 2.0 * torch.sigmoid(self.raw_w)

    def matrices(self, a):
        """a: (K,) real orders -> (K,N,N) complex unitary F^a."""
        theta = -0.5 * math.pi * a[:, None] * self.kidx[None, :]     # (K,N)
        phase = torch.polar(torch.ones_like(theta), theta)           # (K,N) complex
        Vc = self.V.to(phase.dtype)
        return (Vc[None] * phase[:, None, :]) @ Vc.T[None]

    def forward(self, x):
        B, C, N, _ = x.shape
        a_h, a_w = self.orders()
        Mh, Mw = self.matrices(a_h), self.matrices(a_w)
        xc = x.to(Mh.dtype)
        X = torch.einsum("kij,bcjl,kml->bkcim", Mh, xc, Mw)          # Mh x Mw^T
        mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-8)           # (B,K,C,N,N)
        flat = mag.flatten(3)
        n = float(N * N)
        l1 = flat.sum(-1)
        l2 = torch.sqrt((flat ** 2).sum(-1) + 1e-8)
        hoyer = (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0)
        p = flat ** 2 / ((flat ** 2).sum(-1, keepdim=True) + 1e-8)
        ent = -(p * torch.log(p + 1e-12)).sum(-1) / math.log(n)
        mlog = torch.log1p(flat.mean(-1))
        st = torch.stack([mlog, hoyer, ent], dim=-1)                 # (B,K,C,3)
        return st.permute(0, 2, 1, 3).reshape(B, C, self.K * 3)


class SoftRingBank(nn.Module):
    """Learnable soft radial bands on a (u,v) grid -> (B,C,R) band averages."""

    def __init__(self, radius_norm, n_rings=N_RINGS):
        super().__init__()
        self.register_buffer("r", radius_norm)                        # (N,N) in [0,1]
        mu = torch.linspace(0.05, 0.95, n_rings)
        self.mu = nn.Parameter(mu)
        self.raw_sigma = nn.Parameter(torch.full((n_rings,), math.log(math.exp(0.10) - 1)))

    def forward(self, m):
        sigma = F.softplus(self.raw_sigma) + 0.02
        mask = torch.exp(-(self.r[None] - self.mu[:, None, None]) ** 2 /
                         (2 * sigma[:, None, None] ** 2))             # (R,N,N)
        mask = mask / (mask.sum((1, 2), keepdim=True) + EPS)
        return torch.einsum("bchw,rhw->bcr", m, mask)


class SpectralFrontEnd(nn.Module):
    """
    Spectral analysis of a stride-16 feature map:
        1x1 reduce -> homomorphic (log + learnable high-emphasis) filtering
        -> {Fourier rings, wedges, real-cepstrum rings, learnable-order FrFT stats}
        -> shared per-channel MLP -> (B, SPEC_CH*PER_CH)
    """

    def __init__(self, c_in, use_homo=True, use_frft=True, use_cep=True,
                 use_spec=True, learn_orders=True):
        super().__init__()
        self.use_homo, self.use_frft = use_homo, use_frft
        self.use_cep, self.use_spec = use_cep, use_spec
        N, cs = SPEC_SIZE, SPEC_CH
        self.N = N

        self.reduce = nn.Sequential(nn.Conv2d(c_in, cs, 1, bias=False),
                                    nn.BatchNorm2d(cs))

        fy = torch.fft.fftfreq(N)[:, None].expand(N, N)
        fx = torch.fft.fftfreq(N)[None, :].expand(N, N)
        radius = torch.sqrt(fx ** 2 + fy ** 2)
        self.register_buffer("R2", radius ** 2)
        rn = radius / radius.max()
        theta = torch.atan2(fy, fx)
        wedge = []
        for w in range(N_WEDGES):
            phi = w * math.pi / N_WEDGES
            m = torch.exp(2.0 * (torch.cos(2 * (theta - phi)) - 1.0)) * (radius > 0).float()
            wedge.append(m / (m.sum() + EPS))
        self.register_buffer("wedge", torch.stack(wedge))             # (W,N,N)

        # homomorphic filter parameters (per channel)
        self.raw_gL = nn.Parameter(torch.full((cs,), math.log(0.4 / 0.6)))
        self.raw_gH = nn.Parameter(torch.full((cs,), math.log(math.exp(0.5) - 1)))
        self.raw_D0 = nn.Parameter(torch.full((cs,), math.log(math.exp(0.12) - 1)))

        dim = 0
        if use_spec:
            self.ring_spec = SoftRingBank(rn); dim += N_RINGS + N_WEDGES
        if use_cep:
            self.ring_cep = SoftRingBank(rn); dim += N_RINGS
        if use_frft:
            self.frft = LearnableFrFT2D(N, N_ORDERS, learnable=learn_orders)
            dim += 3 * N_ORDERS
        if dim == 0:
            raise ValueError("At least one spectral branch must be active.")
        self.feat_dim = dim
        self.fnorm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, PER_CH))

    def homomorphic_response(self):
        gL = torch.sigmoid(self.raw_gL)
        gH = 1.0 + F.softplus(self.raw_gH)
        D0 = F.softplus(self.raw_D0) + 0.02
        R2 = self.R2[None]
        H = (gH - gL)[:, None, None] * (1 - torch.exp(-R2 / (2 * D0[:, None, None] ** 2))) \
            + gL[:, None, None]
        return H[None]                                                # (1,C,N,N)

    def forward(self, f16):
        if f16.shape[-1] != self.N or f16.shape[-2] != self.N:
            f16 = F.adaptive_avg_pool2d(f16, self.N)
        r = self.reduce(f16)
        if self.use_homo:
            z = torch.log(F.softplus(r) + 1e-3)                       # multiplicative -> additive
            Zf = torch.fft.fft2(z, norm="ortho") * self.homomorphic_response()
            zf = torch.fft.ifft2(Zf, norm="ortho").real
        else:
            Zf = torch.fft.fft2(r, norm="ortho")
            zf = r
        mag = torch.sqrt(Zf.real ** 2 + Zf.imag ** 2 + 1e-6)

        feats = []
        if self.use_spec:
            power = mag ** 2
            feats.append(torch.log1p(self.ring_spec(power)))
            feats.append(torch.log1p(torch.einsum("bchw,rhw->bcr", power, self.wedge)))
        if self.use_cep:
            cep = torch.fft.ifft2(torch.log(mag), norm="ortho").real.abs()
            feats.append(self.ring_cep(cep))
        if self.use_frft:
            feats.append(self.frft(zf))
        f = torch.cat([t.reshape(t.size(0), t.size(1), -1) for t in feats], dim=-1)
        return self.mlp(self.fnorm(f)).flatten(1)                     # (B, SPEC_CH*PER_CH)


# ==========================================================================
# Models
# ==========================================================================
def _build_backbone(timm_name, out_indices):
    bb = timm.create_model(timm_name, pretrained=PRETRAINED, features_only=True,
                           out_indices=out_indices)
    return bb, list(bb.feature_info.channels())


class BaselineModel(nn.Module):
    def __init__(self, timm_name, nc):
        super().__init__()
        self.bb, ch = _build_backbone(timm_name, (4,))
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(ch[-1], nc)
        self.last_feat = None

    def forward(self, x):
        feat = self.bb(x)[-1]
        self.last_feat = feat
        return self.head(self.drop(F.adaptive_avg_pool2d(feat, 1).flatten(1))),

    def spectral_params(self):
        return {}


class HCFModel(nn.Module):
    def __init__(self, timm_name, nc, use_homo=True, use_frft=True, use_cep=True,
                 use_spec=True, learn_orders=True):
        super().__init__()
        self.bb, ch = _build_backbone(timm_name, (3, 4))
        c16, c32 = ch
        assert SPEC_CH * PER_CH == DESC_DIM
        self.spec = SpectralFrontEnd(c16, use_homo, use_frft, use_cep, use_spec, learn_orders)
        self.proj_V = nn.Linear(c32, DESC_DIM, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(DESC_DIM), nn.Linear(DESC_DIM, DESC_DIM))
        self.proj_s = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.out = nn.Linear(DESC_DIM, DESC_DIM)
        self.bypass = nn.Linear(c32, DESC_DIM, bias=False)
        nn.init.zeros_(self.bypass.weight)
        self.norm = nn.LayerNorm(DESC_DIM)
        self.gap_proj = nn.Sequential(nn.Linear(c32, DESC_DIM, bias=False),
                                      nn.LayerNorm(DESC_DIM))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(DESC_DIM, nc)
        self.last_feat = None

    def forward(self, x):
        f16, f32 = self.bb(x)
        self.last_feat = f32
        pooled = F.adaptive_avg_pool2d(f32, 1).flatten(1)
        if self.training and random.random() < SPEC_DROP_P:
            desc = self.gap_proj(pooled)
        else:
            s = self.spec(f16)
            g = torch.sigmoid(self.gate(s))
            desc = self.norm(self.out(g * self.proj_V(pooled))
                             + self.proj_s(s) + self.bypass(pooled))
        return self.head(self.drop(desc)),

    @torch.no_grad()
    def spectral_params(self):
        d = {}
        sp = self.spec
        if sp.use_frft:
            a_h, a_w = sp.frft.orders()
            d["frft_order_h"] = [round(float(v), 4) for v in a_h]
            d["frft_order_w"] = [round(float(v), 4) for v in a_w]
        if sp.use_homo:
            d["homo_gL_mean"] = round(float(torch.sigmoid(sp.raw_gL).mean()), 4)
            d["homo_gH_mean"] = round(float((1 + F.softplus(sp.raw_gH)).mean()), 4)
            d["homo_D0_mean"] = round(float((F.softplus(sp.raw_D0) + 0.02).mean()), 4)
        return d


def create_model(name, nc):
    kind, rest = name.split("_", 1)
    if kind == "BASE":
        if rest not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {rest}")
        return BaselineModel(BACKBONE_REGISTRY[rest], nc)
    if kind == "HCF":
        for bname in BACKBONE_REGISTRY:
            if rest == bname:
                return HCFModel(BACKBONE_REGISTRY[bname], nc)
            if rest.startswith(bname + "_"):
                suf = rest[len(bname) + 1:]
                if suf not in ABLATION_FLAGS:
                    raise ValueError(f"Unknown ablation: {suf}")
                return HCFModel(BACKBONE_REGISTRY[bname], nc, **ABLATION_FLAGS[suf])
    raise ValueError(f"Bad model name: {name}")


# ==========================================================================
# Grad-CAM (hooks the final feature tensor: robust for features_only backbones)
# ==========================================================================
class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)[0]
        feat = self.model.last_feat
        if cls is None:
            cls = int(logits.argmax(1).item())
        grad = torch.autograd.grad(logits[:, cls].sum(), feat)[0]
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feat.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        mn, mx = cam.min(), cam.max()
        cam = ((cam - mn) / (mx - mn + EPS)).detach().cpu().numpy()[0, 0]
        return cam, logits.detach()


_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def denorm(t):
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ==========================================================================
# Test metrics + bootstrap CIs
# ==========================================================================
def _macro_auc(yt, ypr):
    vals = []
    for ci in range(ypr.shape[1]):
        b = (yt == ci).astype(int)
        if b.min() == b.max():
            continue
        fpr, tpr, _ = roc_curve(b, ypr[:, ci])
        vals.append(auc(fpr, tpr))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP, alpha=0.05):
    n = len(y_true)
    rng = np.random.RandomState(42)
    acc = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        acc["acc"].append(accuracy_score(yt, yp))
        acc["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        acc["precision_macro"].append(precision_score(yt, yp, average="macro", zero_division=0))
        acc["recall_macro"].append(recall_score(yt, yp, average="macro", zero_division=0))
        acc["auc_macro"].append(_macro_auc(yt, ypr))
    out = {}
    for m in METRICS:
        lo, hi = np.nanpercentile(acc[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = (float(lo), float(hi))
    return out


def save_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob, out_dir):
    rows = []
    report = classification_report(y_true, y_pred, target_names=classes,
                                   labels=list(range(len(classes))),
                                   output_dict=True, zero_division=0)
    per_auc = []
    for ci, cls in enumerate(classes):
        r = report[cls]
        fpr, tpr, _ = roc_curve((y_true == ci).astype(int), y_prob[:, ci])
        cls_auc = auc(fpr, tpr); per_auc.append(cls_auc)
        cm_b = confusion_matrix((y_true == ci).astype(int), (y_pred == ci).astype(int),
                                labels=[0, 1])
        tn, fp = cm_b[0, 0], cm_b[0, 1]
        rows.append({"model": model_name, "seed": seed, "class": cls,
                     "precision": round(r["precision"], 4), "recall": round(r["recall"], 4),
                     "f1_score": round(r["f1-score"], 4), "support": int(r["support"]),
                     "auc": round(cls_auc, 4), "specificity": round(tn / (tn + fp + EPS), 4)})
    acc = accuracy_score(y_true, y_pred)
    mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mrec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mauc = float(np.mean(per_auc))
    kappa = cohen_kappa_score(y_true, y_pred)
    cis = bootstrap_ci(y_true, y_pred, y_prob)

    for tag, vals in [("MACRO_AVG", {"precision": mprec, "recall": mrec, "f1_score": mf1, "auc": mauc}),
                      ("ACCURACY", {"f1_score": acc}), ("KAPPA", {"f1_score": kappa})]:
        row = {"model": model_name, "seed": seed, "class": tag, "precision": "", "recall": "",
               "f1_score": "", "support": int(y_true.shape[0]), "auc": "", "specificity": ""}
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)

    val_map = {"acc": acc, "f1_macro": mf1, "precision_macro": mprec,
               "recall_macro": mrec, "auc_macro": mauc}
    ci_rows = [{"model": model_name, "seed": seed, "metric": m,
                "value": round(val_map[m], 4),
                "ci_lo": round(cis[m][0], 4), "ci_hi": round(cis[m][1], 4)} for m in METRICS]
    pd.DataFrame(ci_rows).to_csv(Path(out_dir) / f"bootstrap_ci_seed{seed}.csv", index=False)
    out_path = Path(out_dir) / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    [saved] {out_path.name}  (+ bootstrap CIs)")
    return acc, mf1, mprec, mrec, mauc, kappa


# ==========================================================================
# Train + evaluate one (model, seed)
# ==========================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smooth=LABEL_SMOOTH):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_param_groups(model):
    bb_ids = {id(p) for p in model.bb.parameters()}
    order_names = ("raw_h", "raw_w")
    bb, head, order = [], [], []
    for n, p in model.named_parameters():
        if id(p) in bb_ids:
            bb.append(p)
        elif n.endswith(order_names):
            order.append(p)
        else:
            head.append(p)
    groups = [{"params": bb, "lr_mult": 0.1, "weight_decay": 1e-2},
              {"params": head, "lr_mult": 1.0, "weight_decay": 1e-2}]
    if order:
        groups.append({"params": order, "lr_mult": ORDER_LR_MULT, "weight_decay": 0.0})
    return groups


def run_one_seed(seed: int, model_name: str) -> dict:
    done_csv = SAVE_DIR / "all_results.csv"
    if done_csv.exists():
        ex = pd.read_csv(done_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty and (SAVE_DIR / model_name / "preds" / f"preds_seed{seed}.npz").exists():
            print(f"  =>  {model_name} seed={seed} already done - skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs = get_dirs(model_name)
    paths = get_paths(model_name, seed, dirs)
    print(f"\n{'=' * 72}\n  Seed {seed}  |  {model_name}\n{'=' * 72}")

    norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(30),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        v2.RandomGrayscale(p=0.05), v2.RandomAffine(degrees=0, shear=10),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True), norm,
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True), norm,
    ])

    full_train = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds = datasets.ImageFolder(TEST_PATH, transform=val_tf)
    classes = full_train.classes
    nc = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=0.15,
        stratify=full_train.targets, random_state=seed)
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(Subset(full_val_ref, val_idx), batch_size=EVAL_BATCH,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=2)

    model = create_model(model_name, nc).to(DEVICE)
    optimizer = optim.AdamW(build_param_groups(model), lr=LR)
    criterion = FocalLoss()

    best_loss, best_state, patience_ctr = float("inf"), None, 0
    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}

    for epoch in range(EPOCHS):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE); torch.cuda.empty_cache()
        gc.collect()

        cur_lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr * g["lr_mult"]

        use_cutmix = (epoch % 2 == 0)
        cm_alpha = CUTMIX_ALPHA * 0.5 * (1 - math.cos(math.pi * min(epoch / 20.0, 1.0)))
        apply_aug = cm_alpha > 0.05

        model.train()
        tl = tc = tt = 0
        for bidx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if apply_aug and x.size(0) > 1:
                    if use_cutmix:
                        x_a, ya, yb, lam = cutmix_data(x, y, cm_alpha)
                    else:
                        x_a, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
                    logits = model(x_a)[0]
                    loss = aug_criterion(criterion, logits, ya, yb, lam)
                else:
                    logits = model(x)[0]
                    loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                tl += loss.item() * x.size(0)
                tc += (logits.argmax(1) == y).sum().item()
                tt += y.size(0)
            except RuntimeError as e:
                print(f"    ! batch {bidx} skipped: {e}")
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        train_loss, train_acc = tl / max(tt, 1), tc / max(tt, 1)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)[0]
                vl += criterion(logits, y).item() * x.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += y.size(0)
        val_loss, val_acc = vl / max(vt, 1), vc / max(vt, 1)

        for k, v in zip(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
                        [epoch + 1, round(train_loss, 6), round(train_acc, 6),
                         round(val_loss, 6), round(val_acc, 6)]):
            history[k].append(v)
        extra = ""
        sp = model.spectral_params()
        if "frft_order_h" in sp:
            extra = f"  a_h={sp['frft_order_h']}"
        print(f"  [{epoch + 1:2d}/{EPOCHS}]  Tr {train_loss:.4f}/{train_acc:.4f}  "
              f"Va {val_loss:.4f}/{val_acc:.4f}  lr={cur_lr:.2e}{extra}")

        if val_loss < best_loss:
            best_loss, best_state, patience_ctr = val_loss, deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  => Early stopping."); break

    model.load_state_dict(best_state)
    paths["spec_json"].write_text(json.dumps(model.spectral_params(), indent=2))

    # ---- training curves
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history["epoch"], history["train_loss"], "b-", label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"], "c--", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], "r-", alpha=0.85, label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"], "m--", alpha=0.85, label="Val Acc")
    ax2.set_ylabel("Accuracy", color="r"); ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title(f"{model_name}  -  Seed {seed}"); ax1.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(paths["curves_png"], dpi=160, bbox_inches="tight"); plt.close()

    # ---- batched test inference (exact predictions saved for McNemar)
    model.eval()
    yt_l, yp_l, ypr_l = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))[0]
            prob = F.softmax(logits, dim=1).cpu().numpy()
            ypr_l.append(prob); yp_l.append(prob.argmax(1)); yt_l.append(y.numpy())
    y_true, y_pred, y_prob = np.concatenate(yt_l), np.concatenate(yp_l), np.concatenate(ypr_l)
    np.savez(paths["preds_npz"], y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    # ---- Grad-CAM (BASE_* and full HCF_* only), subset per class
    if model_name in GRADCAM_MODELS:
        cam_base = paths["gradcam_base"]
        for c in classes:
            (cam_base / c).mkdir(parents=True, exist_ok=True)
        targets = np.array(test_ds.targets)
        sel = []
        for ci in range(nc):
            ids = np.where(targets == ci)[0]
            sel.extend(ids if GRADCAM_MAX_PER_CLASS is None else ids[:GRADCAM_MAX_PER_CLASS])
        cam_engine = GradCAM(model)
        for i in sel:
            try:
                x, y = test_ds[int(i)]
                xb = x.unsqueeze(0).to(DEVICE)
                with torch.enable_grad():
                    cam, logits = cam_engine.generate(xb, cls=int(y))
                pred = int(logits.argmax(1).item())
                orig = denorm(x)
                tag = "correct" if pred == y else "wrong"
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
                axes[0].imshow(orig); axes[0].axis("off"); axes[0].set_title("Original")
                axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[1].axis("off")
                axes[1].set_title("GradCAM")
                axes[2].imshow(orig); axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
                axes[2].set_title(f"True: {classes[y]} | Pred: {classes[pred]} ({tag})")
                axes[2].axis("off")
                fig.suptitle(f"{model_name}  -  Seed {seed}", y=1.01)
                plt.tight_layout()
                plt.savefig(cam_base / classes[y] / f"img_{int(i):05d}.png",
                            dpi=110, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"    ! gradcam img {i} skipped: {e}")
        model.zero_grad(set_to_none=True)

    acc, mf1, mprec, mrec, mauc, kappa = save_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob, out_dir=dirs["metrics"])

    cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(nc))),
                         index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=0.6, linecolor="gray",
                ax=ax, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix - {model_name}  Seed {seed}")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout(); plt.savefig(paths["cm_png"], dpi=150, bbox_inches="tight"); plt.close()

    roc_rows = []
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, cls in enumerate(classes):
        fpr, tpr, thr = roc_curve((y_true == j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{cls} (AUC={roc_auc:.3f})")
        for f_, t_, th in zip(fpr, tpr, thr):
            roc_rows.append({"model": model_name, "seed": seed, "class": cls,
                             "fpr": round(float(f_), 6), "tpr": round(float(t_), 6),
                             "threshold": round(float(th), 6), "auc": round(roc_auc, 6)})
    ax.plot([0, 1], [0, 1], "k--", lw=1.2); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f"ROC - {model_name}  Seed {seed}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout(); plt.savefig(paths["roc_png"], dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    print(f"  [ok] Acc={acc:.4f}  F1={mf1:.4f}  AUC={mauc:.4f}  kappa={kappa:.4f}")
    return {"seed": seed, "model": model_name, "acc": round(acc, 4), "f1_macro": round(mf1, 4),
            "precision_macro": round(mprec, 4), "recall_macro": round(mrec, 4),
            "auc_macro": round(mauc, 4), "kappa": round(kappa, 4)}


# ==========================================================================
# Statistics
# ==========================================================================
def load_preds(model_name, seed):
    d = np.load(SAVE_DIR / model_name / "preds" / f"preds_seed{seed}.npz")
    return d["y_true"], d["y_pred"]


def mcnemar_test_pair(y_true, pa, pb):
    ca, cb = (pa == y_true), (pb == y_true)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return np.nan, np.nan, np.nan, b, c
    chi2_v = (abs(b - c) - 1.0) ** 2 / n
    p_chi2 = float(chi2.sf(chi2_v, df=1))
    p_exact = float(min(2 * binom.cdf(min(b, c), n, 0.5), 1.0))
    return chi2_v, p_chi2, p_exact, b, c


def fisher_combine(pvals):
    p = np.asarray([x for x in pvals if not np.isnan(x)], dtype=float)
    if len(p) == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(np.clip(p, 1e-300, 1.0)))
    return float(chi2.sf(stat, df=2 * len(p)))


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def run_mcnemar_all_seeds():
    print("\n=== McNemar (image-wise, all seeds) ===")
    rows = []
    for seed in SEEDS:
        preds, y_ref = {}, None
        for m in MODELS:
            try:
                yt, yp = load_preds(m, seed)
                preds[m] = yp
                y_ref = yt if y_ref is None else y_ref
            except Exception as e:
                print(f"  ! seed {seed} {m}: {e}")
        for m1, m2 in itertools.combinations(list(preds), 2):
            chi2_v, p_chi2, p_exact, b, c = mcnemar_test_pair(y_ref, preds[m1], preds[m2])
            rows.append({"seed": seed, "model_A": m1, "model_B": m2, "b": b, "c": c,
                         "chi2": chi2_v, "p_chi2": p_chi2, "p_exact": p_exact,
                         "sig_exact": (not np.isnan(p_exact)) and p_exact < ALPHA_STAT,
                         "winner": m1 if b > c else (m2 if c > b else "tie")})
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_DIR / "mcnemar_seedwise.csv", index=False)
    if not df.empty:
        agg = (df.groupby(["model_A", "model_B"])
                 .agg(mean_p_exact=("p_exact", "mean"), n_sig_exact=("sig_exact", "sum"),
                      n_seeds=("seed", "count"), b_sum=("b", "sum"), c_sum=("c", "sum"))
                 .reset_index())
        agg["fisher_p_exact"] = [
            fisher_combine(df[(df.model_A == a) & (df.model_B == b)]["p_exact"].values)
            for a, b in zip(agg.model_A, agg.model_B)]
        agg.to_csv(SAVE_DIR / "mcnemar_aggregated.csv", index=False)
    print(f"  [saved] mcnemar_seedwise.csv ({len(df)} rows), mcnemar_aggregated.csv")
    return df


def run_paired_stats(df):
    """
    NOTE: with n=5 seeds the smallest attainable two-sided exact Wilcoxon p is
    2/2^5 = 0.0625, so Wilcoxon can NEVER reach p<0.05. We still report it, but
    the paired t-test, Cohen's d_z, and the Fisher-combined McNemar (image-wise)
    are the decisive tests.
    """
    print("\n=== Wilcoxon + paired t-test + Cohen d_z (across seeds) ===")
    rows = []
    for metric in METRICS + ["kappa"]:
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df.model == m1].sort_values("seed")[metric].values
            s2 = df[df.model == m2].sort_values("seed")[metric].values
            if len(s1) < 3 or len(s1) != len(s2):
                continue
            d = s1 - s2
            try:
                w_stat, w_p = wilcoxon(s1, s2, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            try:
                t_stat, t_p = ttest_rel(s1, s2)
            except Exception:
                t_stat, t_p = np.nan, np.nan
            dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
            rows.append({"metric": metric, "model_A": m1, "model_B": m2,
                         "mean_A": float(s1.mean()), "mean_B": float(s2.mean()),
                         "delta_A_minus_B": float(d.mean()), "cohen_dz": dz,
                         "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                         "ttest_stat": t_stat, "ttest_p": t_p})
    res = pd.DataFrame(rows)
    res.to_csv(SAVE_DIR / "wilcoxon_ttest_results.csv", index=False)
    res[(res.model_A == PROPOSED) | (res.model_B == PROPOSED)].to_csv(
        SAVE_DIR / "proposed_vs_all_stats.csv", index=False)
    print(f"  [saved] wilcoxon_ttest_results.csv ({len(res)} rows)")
    return res


def build_superiority_table(df, mc_df, wt_df):
    mc_lookup = {}
    if not mc_df.empty:
        for (a, b), g in mc_df.groupby(["model_A", "model_B"]):
            fp = fisher_combine(g["p_exact"].values)
            mc_lookup[(a, b)] = mc_lookup[(b, a)] = fp
    wt_lookup = {}
    for r in wt_df.itertuples():
        wt_lookup[(r.metric, r.model_A, r.model_B)] = (r.wilcoxon_p, r.ttest_p, r.cohen_dz, r.delta_A_minus_B)
        wt_lookup[(r.metric, r.model_B, r.model_A)] = (r.wilcoxon_p, r.ttest_p,
                                                       -r.cohen_dz if not pd.isna(r.cohen_dz) else np.nan,
                                                       -r.delta_A_minus_B)
    rows = []
    for metric in METRICS + ["kappa"]:
        sub = []
        for m in MODELS:
            if m == PROPOSED:
                continue
            key = (metric, PROPOSED, m)
            if key not in wt_lookup:
                continue
            w_p, t_p, dz, delta = wt_lookup[key]
            sub.append({"Metric": METRIC_LABELS[metric], "Proposed": PROPOSED, "Compared_to": m,
                        "Proposed_mean": df[df.model == PROPOSED][metric].mean(),
                        "Other_mean": df[df.model == m][metric].mean(),
                        "Delta": delta, "Cohen_dz": dz, "Wilcoxon_p": w_p, "t_test_p": t_p,
                        "McNemar_fisher_p": mc_lookup.get((PROPOSED, m), np.nan)})
        if not sub:
            continue
        t_adj = holm_adjust([s["t_test_p"] for s in sub])
        mc_adj = holm_adjust([s["McNemar_fisher_p"] for s in sub])
        for s, ta, ma in zip(sub, t_adj, mc_adj):
            s["t_test_p_holm"] = ta
            s["McNemar_fisher_p_holm"] = ma
            s["sig_t_test_holm"] = bool(s["Delta"] > 0 and not np.isnan(ta) and ta < ALPHA_STAT)
            s["sig_McNemar_holm"] = bool(s["Delta"] > 0 and not np.isnan(ma) and ma < ALPHA_STAT)
            s["Verdict"] = ("SIGNIFICANT WIN" if (s["sig_t_test_holm"] or s["sig_McNemar_holm"])
                            else ("numerical win" if s["Delta"] > 0 else "no win"))
        rows.extend(sub)
    sup = pd.DataFrame(rows)
    sup.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print("\n=== Superiority Table (Proposed vs all) ===")
    print(sup.round(4).to_string(index=False))
    return sup


def collect_spectral_params():
    rows = []
    for m in MODELS:
        if not m.startswith("HCF_"):
            continue
        for seed in SEEDS:
            p = SAVE_DIR / m / "metrics" / f"spectral_params_seed{seed}.json"
            if p.exists():
                d = json.loads(p.read_text()); d.update({"model": m, "seed": seed})
                rows.append(d)
    if rows:
        pd.DataFrame(rows).to_csv(SAVE_DIR / "learned_spectral_params.csv", index=False)
        print("  [saved] learned_spectral_params.csv (learned FrFT orders / homomorphic filter)")


# ==========================================================================
# Plots
# ==========================================================================
def save_plots(df, all_seeds):
    palette = sns.color_palette("tab20", max(len(MODELS), 3))
    for metric in METRICS + ["kappa"]:
        means, stds, labs = [], [], []
        for m in MODELS:
            r = df[df.model == m]
            if not r.empty:
                means.append(r[metric].mean()); stds.append(r[metric].std() if len(r) > 1 else 0.0)
                labs.append(short(m))
        if not means:
            continue
        x = np.arange(len(means))
        fig, ax = plt.subplots(figsize=(20, 7))
        bars = ax.bar(x, means, yerr=stds, color=palette[:len(means)], edgecolor="k",
                      linewidth=0.8, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} - All Models (mean±std, n={len(all_seeds)} seeds)")
        ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
        for bar, mv in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{mv:.3f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        plt.tight_layout(); plt.savefig(SAVE_DIR / f"barplot_{metric}.png", dpi=160, bbox_inches="tight")
        plt.close()

    gain_rows = []
    for b in BACKBONE_NAMES:
        for metric in METRICS:
            bm = df[df.model == f"BASE_{b}"][metric].values
            tm = df[df.model == f"HCF_{b}"][metric].values
            if len(bm) and len(tm):
                gain_rows.append({"backbone": b, "metric": METRIC_LABELS[metric],
                                  "gain": round(float(tm.mean() - bm.mean()), 4)})
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(SAVE_DIR / "hcf_gain.csv", index=False)
    if not gain_df.empty:
        gp = gain_df.pivot(index="backbone", columns="metric", values="gain")
        gp = gp[[METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in gp.columns]]
        xp = np.arange(len(gp)); w = 0.14; pal = sns.color_palette("Set2", len(gp.columns))
        fig, ax = plt.subplots(figsize=(12, 6))
        for ci, col in enumerate(gp.columns):
            off = (ci - len(gp.columns) / 2 + 0.5) * w
            vals = gp[col].values
            bars = ax.bar(xp + off, vals, w, label=col, color=pal[ci], edgecolor="k")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.001 if v >= 0 else -0.004),
                        f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        ax.axhline(0, color="k", lw=1.2, ls="--")
        ax.set_xticks(xp); ax.set_xticklabels(gp.index, rotation=15, ha="right")
        ax.set_ylabel("HCF gain (HCF_X - BASE_X)")
        ax.set_title(f"HCF Gain per Backbone × Metric (mean over {len(all_seeds)} seeds)")
        ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(SAVE_DIR / "hcf_gain_barplot.png", dpi=180, bbox_inches="tight")
        plt.close()

    for b in BACKBONE_NAMES:
        full = f"HCF_{b}"; rows = []
        for metric in METRICS:
            fv = df[df.model == full][metric].values
            if len(fv) == 0:
                continue
            for suf in ABLATION_FLAGS:
                av = df[df.model == f"{full}_{suf}"][metric].values
                if len(av):
                    rows.append({"ablation": suf, "metric": METRIC_LABELS[metric],
                                 "delta": float(av.mean() - fv.mean())})
        if not rows:
            continue
        piv = pd.DataFrame(rows).pivot(index="ablation", columns="metric", values="delta")
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax, edgecolor="k", width=0.8)
        ax.axhline(0, color="k", lw=1.2, ls="--")
        ax.set_title(f"Ablation Δ (ablation - full) - {b}"); ax.set_ylabel("Δ metric")
        ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); plt.savefig(SAVE_DIR / f"ablation_delta_{b}.png", dpi=160, bbox_inches="tight")
        plt.close()

    try:
        mc = pd.read_csv(SAVE_DIR / "mcnemar_seedwise.csv")
        pm = pd.DataFrame(np.ones((len(MODELS), len(MODELS))), index=MODELS, columns=MODELS)
        for (m1, m2), g in mc.groupby(["model_A", "model_B"]):
            pv = g["p_exact"].mean()
            if not pd.isna(pv):
                pm.loc[m1, m2] = pv; pm.loc[m2, m1] = pv
        sm = {m: short(m) for m in MODELS}
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(pm.rename(index=sm, columns=sm).astype(float), annot=True, fmt=".3f",
                    cmap="RdYlGn_r", vmin=0, vmax=0.1, mask=np.eye(len(MODELS), dtype=bool),
                    ax=ax, linewidths=0.2, linecolor="white", annot_kws={"size": 5},
                    cbar_kws={"label": f"Mean McNemar exact p (n={len(all_seeds)} seeds)"})
        ax.set_title("Pairwise McNemar exact p-values (image-wise, mean over seeds)")
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
        plt.tight_layout(); plt.savefig(SAVE_DIR / "mcnemar_heatmap.png", dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  ! mcnemar heatmap skipped: {e}")
    print("  [ok] All comparison plots saved.")


# ==========================================================================
# Smoke test (no dataset / no pretrained weights required)
# ==========================================================================
def smoke_test():
    global PRETRAINED
    PRETRAINED = False
    print("== HCF smoke test ==")
    # 1. FrFT math
    for N in (7, 14):
        fr = LearnableFrFT2D(N, 3)
        Fm = torch.fft.fft(torch.eye(N), norm="ortho")
        one = fr.matrices(torch.ones(1))[0]
        two = fr.matrices(2 * torch.ones(1))[0]
        zero = fr.matrices(torch.zeros(1))[0]
        add = fr.matrices(torch.tensor([0.3]))[0] @ fr.matrices(torch.tensor([0.5]))[0] \
            - fr.matrices(torch.tensor([0.8]))[0]
        par = torch.eye(N)[(-torch.arange(N)) % N].to(two.dtype)
        print(f"  N={N}: |F^1-DFT|={float((one - Fm).abs().max()):.1e}  "
              f"|F^2-P|={float((two - par).abs().max()):.1e}  "
              f"|F^0-I|={float((zero - torch.eye(N)).abs().max()):.1e}  "
              f"additivity={float(add.abs().max()):.1e}")
        assert float((one - Fm).abs().max()) < 1e-4
        x = torch.randn(2, 5, N, N)
        a = fr.matrices(torch.tensor([0.37]))[0]
        X = a @ x.to(a.dtype) @ a.T
        print(f"      Parseval: ||x||={float(x.norm()):.4f}  ||FrFT x||={float(X.abs().pow(2).sum().sqrt()):.4f}")
    # 2. all architectures
    names = ["BASE_" + b for b in BACKBONE_REGISTRY] + ["HCF_" + b for b in BACKBONE_REGISTRY]
    names += [f"HCF_{BACKBONE_NAMES[0]}_{s}" for s in ABLATION_FLAGS]
    for n in names:
        m = create_model(n, 4)
        npar = sum(p.numel() for p in m.parameters()) / 1e6
        m.train(); x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
        out = m(x)[0]; loss = out.logsumexp(1).mean(); loss.backward()
        m.eval()
        with torch.no_grad():
            out2 = m(x)[0]
        gc_ = GradCAM(m).generate(x[:1], cls=0)[0]
        assert out.shape == (4, 4) and out2.shape == (4, 4) and gc_.shape == (IMG_SIZE, IMG_SIZE)
        assert torch.isfinite(out2).all()
        extra = ""
        if isinstance(m, HCFModel) and m.spec.use_frft and m.spec.frft.raw_h.requires_grad:
            g = m.spec.frft.raw_h.grad
            extra = f"  grad(order)={'ok' if g is not None and torch.isfinite(g).all() else 'NONE'}"
        print(f"  {n:38s} params={npar:6.2f}M  out={tuple(out2.shape)}{extra}")
    print("== smoke test PASSED ==")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke_test(); sys.exit(0)

    print("\n" + "=" * 72)
    print("  HCF-Net: Homomorphic-Cepstral Fractional-spectral Network")
    print(f"  Proposed : {PROPOSED}")
    print(f"  Seeds    : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device   : {DEVICE}   Epochs: {EPOCHS}   DESC_DIM: {DESC_DIM}")
    print(f"  Models   : {len(MODELS)}  (BASE + HCF + leave-one-out ablations)")
    print(f"  Ablations: {list(ABLATION_FLAGS)}")
    print("=" * 72)

    all_results = []
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            all_results.append(run_one_seed(seed, model_name))
            (pd.DataFrame(all_results).drop_duplicates(subset=["model", "seed"])
               .reset_index(drop=True).to_csv(SAVE_DIR / "all_results.csv", index=False))

    df = (pd.read_csv(SAVE_DIR / "all_results.csv")
            .drop_duplicates(subset=["model", "seed"]).reset_index(drop=True))
    print("\n=== Results (all seeds) ===")
    print(df.to_string(index=False))
    df.to_csv(SAVE_DIR / "summary_all_seeds.csv", index=False)
    summary = df.groupby("model")[METRICS + ["kappa"]].agg(["mean", "std"]).round(4)
    summary.to_csv(SAVE_DIR / "summary_mean_std.csv")
    print("\n=== Mean ± Std ===")
    print(summary)

    mc_df = run_mcnemar_all_seeds()
    wt_df = run_paired_stats(df)
    build_superiority_table(df, mc_df, wt_df)
    collect_spectral_params()
    save_plots(df, SEEDS)
    print(f"\n{'=' * 72}\n  All outputs saved to: {SAVE_DIR.resolve()}\n{'=' * 72}")
