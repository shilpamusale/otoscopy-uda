# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/DANN_FullFeatureAnalysis_FINAL.py
# Role: Feature extraction, t-SNE, linear probes, MMD, and statistical tests for the DANN pipeline.
#
# This is original manuscript code, kept VERBATIM. Do NOT refactor or bugfix it
# here — changing it would break reproducibility of the paper's results. The
# corrected, refactored, tested version of this logic lives in the product
# package under src/otoscopy_audit/. See experiments/README.md for rationale.
#
# Settings are in the USER SETTINGS block below (hardcoded paths are expected
# here; the product package reads them from configs/ instead).
# ============================================================================

import copy
import math
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS — change these paths and parameters to match your setup
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_ROOT  = "/Users/jordanvilla/Desktop/TM_Datasets/eardrumDs/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/Datos_Chile_Combined/"

# Where all feature files and plots will be saved
FEATURE_OUTPUT_DIR = "/Users/jordanvilla/Desktop/Feature_Output/Chile/"

# Name for this target dataset — used in filenames and plot titles
# Change to "Chile" when running on the Chile dataset
TARGET_NAME = "Chile"

NUM_EPOCHS              = 20
BATCH_SIZE              = 16
LEARNING_RATE           = 1e-4
VAL_SPLIT               = 0.10
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE              = 224
RANDOM_SEED             = 42

N_TARGET_ADAPT  = 50
N_REPEATS       = 30
N_TEST_NORMAL   = 50
N_TEST_ABNORMAL = 50
DOMAIN_LOSS_WEIGHT = 1.0

USE_PRETRAINED  = True
FREEZE_BACKBONE = False

# How many images to sample per class for t-SNE (keeps plots readable)
# Set to None to use all images
TSNE_SAMPLE_PER_CLASS = None

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
Path(FEATURE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(RANDOM_SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"Output directory: {FEATURE_OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class BinaryFolderDataset(Dataset):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root_dir, transform=None, verify_images=False):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        self.bad_files = []

        normal_dir   = self.root_dir / "normal"
        abnormal_dir = self.root_dir / "abnormal"

        if not normal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {normal_dir}")
        if not abnormal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {abnormal_dir}")

        self.samples.extend(self._gather_images(normal_dir,   label=0, verify_images=verify_images))
        self.samples.extend(self._gather_images(abnormal_dir, label=1, verify_images=verify_images))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")

        print(f"Loaded {len(self.samples)} valid images from {root_dir}")
        print(f"  Normal:   {sum(1 for _, y in self.samples if y == 0)}")
        print(f"  Abnormal: {sum(1 for _, y in self.samples if y == 1)}")

        if self.bad_files:
            print(f"Skipped {len(self.bad_files)} unreadable files.")

    def _is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img.convert("RGB").load()
            return True
        except Exception:
            return False

    def _gather_images(self, folder, label, verify_images=False):
        items = []
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.IMG_EXTENSIONS:
                if verify_images:
                    if self._is_valid_image(path):
                        items.append((str(path), label))
                    else:
                        self.bad_files.append(str(path))
                else:
                    items.append((str(path), label))
        return items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        _, label, img_path = self.subset[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT REVERSAL LAYER
# ─────────────────────────────────────────────────────────────────────────────
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_grl=1.0):
        return GradientReversalFunction.apply(x, lambda_grl)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ResNet50DANN(nn.Module):
    def __init__(self, use_pretrained=True, freeze_backbone=False):
        super().__init__()

        backbone = models.resnet50(
            weights=ResNet50_Weights.DEFAULT if use_pretrained else None
        )

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim       = backbone.fc.in_features

        self.class_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 2)
        )

        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        if freeze_backbone:
            for param in self.class_classifier.parameters():
                param.requires_grad = True
            for param in self.domain_classifier.parameters():
                param.requires_grad = True

        self.layer_block_names = ["layer1", "layer2", "layer3", "layer4"]

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, lambda_grl=1.0):
        features      = self.extract_features(x)
        class_logits  = self.class_classifier(features)
        rev_features  = self.grl(features, lambda_grl)
        domain_logits = self.domain_classifier(rev_features)
        return class_logits, domain_logits

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return dict(accuracy=accuracy, sensitivity=sensitivity,
                specificity=specificity, precision=precision,
                recall=recall, f1=f1, auroc=auroc,
                tn=tn, fp=fp, fn=fn, tp=tp)


def print_metrics(title, loss, metrics):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Loss:        {loss:.4f}")
    for k in ['accuracy','sensitivity','specificity','precision','recall','f1','auroc']:
        print(f"{k.capitalize():<12}: {metrics[k]:.4f}")
    print(f"TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TP: {metrics['tp']}")

# ─────────────────────────────────────────────────────────────────────────────
# TARGET DATASET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices   = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]
    rng = random.Random(seed)
    n_normal   = n_total // 2
    n_abnormal = n_total - n_normal
    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError(f"Not enough target images for adaptation set of {n_total}.")
    sampled = rng.sample(normal_indices, n_normal) + rng.sample(abnormal_indices, n_abnormal)
    rng.shuffle(sampled)
    return sampled


def get_remaining_target_pool(dataset, excluded_indices):
    excluded_set       = set(excluded_indices)
    remaining_normal   = []
    remaining_abnormal = []
    for idx, (_, label) in enumerate(dataset.samples):
        if idx in excluded_set:
            continue
        (remaining_normal if label == 0 else remaining_abnormal).append(idx)
    return remaining_normal, remaining_abnormal


def get_lambda_grl(epoch_idx, num_epochs):
    p = float(epoch_idx) / max(1, num_epochs - 1)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_from_loader(model, loader, desc=""):
    model.eval()
    all_feats  = []
    all_labels = []
    all_paths  = []
    print(f"  Extracting features: {desc} ...")
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            feats  = model.extract_features(images)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats, np.array(all_labels), all_paths


def make_full_dataset_loader(dataset, transform):
    class FullTransformedDataset(Dataset):
        def __init__(self, base, tf):
            self.base = base
            self.tf   = tf
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img_path, label = self.base.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.tf:
                image = self.tf(image)
            return image, label, img_path

    ds     = FullTransformedDataset(dataset, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader

# ─────────────────────────────────────────────────────────────────────────────
# SAVE FEATURES TO CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_features_to_excel(feats, labels, paths, filename):
    feat_cols = {f"feat_{i}": feats[:, i] for i in range(feats.shape[1])}
    df = pd.DataFrame({"image_path": paths, "label": labels, **feat_cols})
    csv_filename = filename.replace(".xlsx", ".csv")
    out_path = Path(FEATURE_OUTPUT_DIR) / csv_filename
    df.to_csv(out_path, index=False)
    print(f"  Saved features → {out_path}  ({df.shape[0]} images × {feats.shape[1]} features)")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# t-SNE PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def run_tsne(feats, perplexity=30, random_state=42):
    import sklearn
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=random_state)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 2):
        tsne_kwargs['max_iter'] = 1000
    else:
        tsne_kwargs['n_iter'] = 1000
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(feats)


def sample_for_tsne(feats, labels, domains, n_per_class=None):
    if n_per_class is None:
        return feats, labels, domains
    rng    = np.random.default_rng(42)
    keep   = []
    unique = np.unique(np.stack([labels, domains], axis=1), axis=0)
    for lbl, dom in unique:
        idx = np.where((labels == lbl) & (domains == dom))[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        keep.extend(chosen.tolist())
    keep = sorted(keep)
    return feats[keep], labels[keep], domains[keep]


def plot_tsne_panel(source_feats, source_labels,
                    target_feats, target_labels,
                    stage_label, target_name, save_name):
    all_feats   = np.concatenate([source_feats, target_feats])
    all_labels  = np.concatenate([source_labels, target_labels])
    all_domains = np.array([0] * len(source_feats) + [1] * len(target_feats))

    all_feats, all_labels, all_domains = sample_for_tsne(
        all_feats, all_labels, all_domains, TSNE_SAMPLE_PER_CLASS
    )

    print(f"  Running t-SNE for: {stage_label} ({len(all_feats)} points) ...")
    embedded = run_tsne(all_feats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Feature Space — {stage_label}\nSource vs {target_name}",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    src_mask = all_domains == 0
    tgt_mask = all_domains == 1
    ax.scatter(embedded[src_mask, 0], embedded[src_mask, 1],
               c='#2166ac', alpha=0.55, s=18, label='Source', edgecolors='none')
    ax.scatter(embedded[tgt_mask, 0], embedded[tgt_mask, 1],
               c='#d6604d', alpha=0.55, s=18, label=target_name, edgecolors='none')
    ax.set_title('By Domain', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)

    ax = axes[1]
    norm_mask = all_labels == 0
    abn_mask  = all_labels == 1
    ax.scatter(embedded[norm_mask, 0], embedded[norm_mask, 1],
               c='#4dac26', alpha=0.55, s=18, label='Normal', edgecolors='none')
    ax.scatter(embedded[abn_mask, 0], embedded[abn_mask, 1],
               c='#b2182b', alpha=0.55, s=18, label='Abnormal', edgecolors='none')
    ax.set_title('By Class', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path = Path(FEATURE_OUTPUT_DIR) / save_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MOVEMENT ANALYSIS
# before_gap  = distance between source baseline and target before DANN
# after_gap   = distance between source baseline and target after DANN
# movement    = before_gap - after_gap (positive = DANN closed the gap)
# ─────────────────────────────────────────────────────────────────────────────
def compute_feature_movement(source_feats, target_before_feats, target_after_feats):
    n_feats     = source_feats.shape[1]
    before_gaps = np.zeros(n_feats)
    after_gaps  = np.zeros(n_feats)

    for i in range(n_feats):
        src   = source_feats[:, i]
        t_bef = target_before_feats[:, i]
        t_aft = target_after_feats[:, i]
        before_gaps[i] = abs(src.mean() - t_bef.mean())
        after_gaps[i]  = abs(src.mean() - t_aft.mean())

    movement = before_gaps - after_gaps
    return before_gaps, after_gaps, movement


def save_movement_to_excel(before_gaps, after_gaps, movement, filename):
    df = pd.DataFrame({
        "feature_index": np.arange(len(movement)),
        "before_gap":    before_gaps,
        "after_gap":     after_gaps,
        "movement":      movement,
    })
    out_path = Path(FEATURE_OUTPUT_DIR) / filename
    df.to_excel(out_path, index=False)
    print(f"  Saved movement scores → {out_path}")
    return df


def plot_movement_comparison(movement_osu, movement_chile, save_name):
    fig, ax = plt.subplots(figsize=(8, 7))

    high_osu_low_chile = (movement_osu > 0) & (movement_chile <= 0)
    both_high          = (movement_osu > 0) & (movement_chile >  0)
    both_low           = (movement_osu <= 0) & (movement_chile <= 0)
    low_osu_high_chile = (movement_osu <= 0) & (movement_chile >  0)

    ax.scatter(movement_osu[high_osu_low_chile], movement_chile[high_osu_low_chile],
               c='#d6604d', alpha=0.5, s=12, label=f'OSU only ({high_osu_low_chile.sum()})')
    ax.scatter(movement_osu[both_high], movement_chile[both_high],
               c='#4dac26', alpha=0.5, s=12, label=f'Both ({both_high.sum()})')
    ax.scatter(movement_osu[both_low], movement_chile[both_low],
               c='#aaaaaa', alpha=0.4, s=12, label=f'Neither ({both_low.sum()})')
    ax.scatter(movement_osu[low_osu_high_chile], movement_chile[low_osu_high_chile],
               c='#2166ac', alpha=0.5, s=12, label=f'Chile only ({low_osu_high_chile.sum()})')

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Feature Movement — OSU\n(positive = DANN closed gap toward source)', fontsize=11)
    ax.set_ylabel('Feature Movement — Chile\n(positive = DANN closed gap toward source)', fontsize=11)
    ax.set_title('Per-Feature DANN Movement:\nOSU vs Chile', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, title='Quadrant')
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out_path = Path(FEATURE_OUTPUT_DIR) / save_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved movement comparison plot → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DISCRIMINABILITY PER FEATURE
# Mann-Whitney U test per feature: source baseline vs target after DANN
# Low p-value = feature still domain-specific = DANN failed to align it
# ─────────────────────────────────────────────────────────────────────────────
def domain_discriminability(source_feats, target_feats, label):
    n_feats      = source_feats.shape[1]
    p_values     = np.zeros(n_feats)
    effect_sizes = np.zeros(n_feats)

    print(f"  Running domain discriminability tests for {label} ...")
    for i in range(n_feats):
        src = source_feats[:, i]
        tgt = target_feats[:, i]
        stat, p = mannwhitneyu(src, tgt, alternative='two-sided')
        p_values[i] = p
        n1, n2 = len(src), len(tgt)
        effect_sizes[i] = abs(1 - (2 * stat) / (n1 * n2))

    _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    df = pd.DataFrame({
        "feature_index":   np.arange(n_feats),
        "p_raw":           p_values,
        "p_fdr_corrected": p_corrected,
        "effect_size":     effect_sizes,
        "domain_specific": (p_corrected < 0.05) & (effect_sizes > 0.3),
    })
    out_path = Path(FEATURE_OUTPUT_DIR) / f"domain_discriminability_{label}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"  Saved domain discriminability → {out_path}")
    n_specific  = df['domain_specific'].sum()
    n_invariant = (~df['domain_specific']).sum()
    print(f"  {label}: {n_specific} domain-specific features, {n_invariant} domain-invariant features")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# PROBING CLASSIFIERS
# Tests whether features can separate normal from abnormal
# Compares all 2048 features vs domain-invariant features only
# ─────────────────────────────────────────────────────────────────────────────
def probe_features(feats, labels, feature_mask, subset_label):
    X = feats[:, feature_mask]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = {}
    for name, clf in [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVM_RBF',            SVC(kernel='rbf', probability=True, random_state=42)),
        ('kNN_5',              KNeighborsClassifier(n_neighbors=5)),
    ]:
        scores = cross_val_score(clf, X, labels, cv=5, scoring='f1')
        results[name] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"    {subset_label} | {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    return results


def run_probe_experiment(source_feats, source_labels,
                          target_feats, target_labels,
                          invariant_mask, target_name):
    print(f"\n{'='*60}")
    print(f"PROBING CLASSIFIER EXPERIMENT — {target_name}")
    print(f"{'='*60}")

    all_mask = np.ones(source_feats.shape[1], dtype=bool)

    rows = []
    for feats, labels, ds_label in [
        (source_feats, source_labels, "Source"),
        (target_feats, target_labels, target_name),
    ]:
        for mask, mask_label in [
            (all_mask,       "All 2048 features"),
            (invariant_mask, "Domain-invariant features only"),
        ]:
            res = probe_features(feats, labels, mask, f"{ds_label} | {mask_label}")
            for clf_name, scores in res.items():
                rows.append({
                    "dataset":     ds_label,
                    "feature_set": mask_label,
                    "n_features":  mask.sum(),
                    "classifier":  clf_name,
                    "f1_mean":     scores['mean'],
                    "f1_std":      scores['std'],
                })

    df = pd.DataFrame(rows)
    out_path = Path(FEATURE_OUTPUT_DIR) / f"probe_results_{target_name}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\n  Saved probe results → {out_path}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM MEAN DISCREPANCY (MMD)
# Measures the squared distance between source and target distributions
# in a reproducing kernel Hilbert space using an RBF kernel.
# Returns a single scalar — lower = more similar distributions.
# Computed three ways:
#   1. Overall   — all source vs all target images
#   2. Normal    — source normals vs target normals
#   3. Abnormal  — source abnormals vs target abnormals
# ─────────────────────────────────────────────────────────────────────────────
def compute_mmd(X, Y):
    """
    Compute MMD between feature matrices X (source) and Y (target).
    Uses RBF kernel with gamma set by the median heuristic.
    X: n_samples x n_features
    Y: m_samples x n_features
    Returns: scalar MMD value
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.metrics import pairwise_distances

    # Median heuristic for gamma — standard practice
    all_data = np.concatenate([X, Y])
    dists    = pairwise_distances(all_data, metric='euclidean')
    median_d = np.median(dists[dists > 0])
    gamma    = 1.0 / (2.0 * median_d ** 2 + 1e-8)

    XX = rbf_kernel(X, X, gamma=gamma).mean()
    YY = rbf_kernel(Y, Y, gamma=gamma).mean()
    XY = rbf_kernel(X, Y, gamma=gamma).mean()

    return float(XX + YY - 2 * XY)


def run_mmd_analysis(source_feats, source_labels,
                     target_before_feats, target_after_feats, target_labels,
                     target_name):
    """
    Compute MMD before and after DANN for three conditions:
      overall, normal-only, abnormal-only.
    Saves results to Excel and prints a summary table.
    """
    print(f"  Computing MMD for {target_name} ...")

    rows = []
    for condition, src_mask, tgt_mask in [
        ("Overall",  np.ones(len(source_labels),  dtype=bool),
                     np.ones(len(target_labels),   dtype=bool)),
        ("Normal",   source_labels == 0, target_labels == 0),
        ("Abnormal", source_labels == 1, target_labels == 1),
    ]:
        src  = source_feats[src_mask]
        tbef = target_before_feats[tgt_mask]
        taft = target_after_feats[tgt_mask]

        mmd_before = compute_mmd(src, tbef)
        mmd_after  = compute_mmd(src, taft)
        reduction  = mmd_before - mmd_after
        pct        = (reduction / mmd_before * 100) if mmd_before > 0 else 0.0

        rows.append({
            "dataset":        target_name,
            "condition":      condition,
            "n_source":       src_mask.sum(),
            "n_target":       tgt_mask.sum(),
            "mmd_before":     round(mmd_before, 6),
            "mmd_after":      round(mmd_after,  6),
            "mmd_reduction":  round(reduction,  6),
            "reduction_pct":  round(pct, 2),
        })

        print(f"    {condition:<10} before={mmd_before:.6f}  after={mmd_after:.6f}  "
              f"reduction={reduction:.6f} ({pct:.1f}%)")

    df = pd.DataFrame(rows)
    out_path = Path(FEATURE_OUTPUT_DIR) / f"mmd_analysis_{target_name}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"  Saved MMD results → {out_path}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        class_logits, _ = self.model(input_tensor, lambda_grl=0.0)

        if class_idx is None:
            class_idx = class_logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = class_logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam_grid(model, image_paths, labels, title, save_name, n_images=8):
    target_layer = list(model.feature_extractor.children())[-2][-1]
    grad_cam     = GradCAM(model, target_layer)

    n_show = min(n_images, len(image_paths))
    fig, axes = plt.subplots(n_show, 2, figsize=(8, n_show * 3))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for row, (path, label) in enumerate(zip(image_paths[:n_show], labels[:n_show])):
        raw_img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        raw_np  = np.array(raw_img) / 255.0

        inp = eval_transform(Image.open(path).convert("RGB")).unsqueeze(0)
        cam = grad_cam.generate(inp)

        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * raw_np + 0.5 * heatmap

        class_str = "Normal" if label == 0 else "Abnormal"
        axes[row, 0].imshow(raw_np)
        axes[row, 0].set_title(f"Original ({class_str})", fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(np.clip(overlay, 0, 1))
        axes[row, 1].set_title("Grad-CAM", fontsize=8)
        axes[row, 1].axis('off')

    plt.tight_layout()
    out_path = Path(FEATURE_OUTPUT_DIR) / save_name
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Grad-CAM grid → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch_dann(model, source_loader, target_loader, optimizer,
                         class_criterion, domain_criterion,
                         lambda_grl, domain_loss_weight=1.0):
    model.train()
    running_total_loss  = 0.0
    running_class_loss  = 0.0
    running_domain_loss = 0.0
    source_labels_all   = []
    source_preds_all    = []
    target_iter         = cycle(target_loader)

    for source_images, source_labels, _ in source_loader:
        target_images, _, _ = next(target_iter)

        source_images  = source_images.to(device)
        source_labels  = source_labels.to(device)
        target_images  = target_images.to(device)

        optimizer.zero_grad()

        source_class_logits, source_domain_logits = model(source_images, lambda_grl=lambda_grl)
        _, target_domain_logits                   = model(target_images, lambda_grl=lambda_grl)

        class_loss           = class_criterion(source_class_logits, source_labels)
        source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=device)
        target_domain_labels = torch.ones(target_images.size(0),  dtype=torch.long, device=device)
        domain_loss          = 0.5 * (domain_criterion(source_domain_logits, source_domain_labels) +
                                      domain_criterion(target_domain_logits, target_domain_labels))
        total_loss           = class_loss + domain_loss_weight * domain_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_total_loss  += total_loss.item() * source_images.size(0)
        running_class_loss  += class_loss.item() * source_images.size(0)
        running_domain_loss += domain_loss.item() * source_images.size(0)

        source_preds = torch.argmax(source_class_logits, dim=1)
        source_labels_all.extend(source_labels.detach().cpu().numpy())
        source_preds_all.extend(source_preds.detach().cpu().numpy())

    n = len(source_loader.dataset)
    return (running_total_loss / n, running_class_loss / n,
            running_domain_loss / n, accuracy_score(source_labels_all, source_preds_all))


def evaluate_source_classification(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            class_logits, _ = model(images, lambda_grl=0.0)
            loss  = criterion(class_logits, labels)
            probs = torch.softmax(class_logits, dim=1)[:, 1]
            preds = torch.argmax(class_logits, dim=1)
            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths


def evaluate_target_test_subset(model, dataset, subset_indices, criterion):
    subset         = Subset(dataset, subset_indices)
    subset_dataset = TransformedSubset(subset, transform=eval_transform)
    loader         = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return evaluate_source_classification(model, loader, criterion)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

full_source_dataset = BinaryFolderDataset(TRAIN_ROOT,  transform=None, verify_images=True)
full_target_dataset = BinaryFolderDataset(TARGET_ROOT, transform=None, verify_images=True)

source_train_size = int((1 - VAL_SPLIT) * len(full_source_dataset))
source_val_size   = len(full_source_dataset) - source_train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
source_train_subset, source_val_subset = random_split(
    full_source_dataset, [source_train_size, source_val_size], generator=generator
)

source_train_dataset = TransformedSubset(source_train_subset, transform=train_transform)
source_val_dataset   = TransformedSubset(source_val_subset,   transform=eval_transform)

target_adapt_indices = make_balanced_fixed_target_adapt_set(
    full_target_dataset, n_total=N_TARGET_ADAPT, seed=RANDOM_SEED
)
target_adapt_subset  = Subset(full_target_dataset, target_adapt_indices)
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=train_transform)

remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset, excluded_indices=target_adapt_indices
)

print(f"\nFixed target adaptation set: {len(target_adapt_indices)} images")
print(f"Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"Remaining abnormal for testing: {len(remaining_abnormal_indices)}")

if len(remaining_normal_indices)   < N_TEST_NORMAL:
    raise ValueError("Not enough remaining normal images for testing.")
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError("Not enough remaining abnormal images for testing.")

source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
source_val_loader   = DataLoader(source_val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

source_full_loader = make_full_dataset_loader(full_source_dataset, eval_transform)
target_full_loader = make_full_dataset_loader(full_target_dataset, eval_transform)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = ResNet50DANN(use_pretrained=USE_PRETRAINED, freeze_backbone=FREEZE_BACKBONE).to(device)

class_criterion  = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()
optimizer        = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1A: BASELINE FEATURE EXTRACTION (BEFORE DANN)
# Pretrained ResNet50 before any DANN training — fixed reference point
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1A: BASELINE FEATURE EXTRACTION (BEFORE DANN)")
print("="*60)

torch.save(model.state_dict(), Path(FEATURE_OUTPUT_DIR) / "baseline_model_20260616.pth")
print("  Saved baseline model checkpoint → baseline_model.pth")

print("\nExtracting baseline features ...")
source_feats_before, source_labels_arr, source_paths = \
    extract_features_from_loader(model, source_full_loader, "Source (baseline)")

target_feats_before, target_labels_arr, target_paths = \
    extract_features_from_loader(model, target_full_loader, f"{TARGET_NAME} (baseline)")

save_features_to_excel(source_feats_before, source_labels_arr, source_paths,
                       "source_features_baseline.xlsx")
save_features_to_excel(target_feats_before, target_labels_arr, target_paths,
                       f"{TARGET_NAME}_features_baseline.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A: t-SNE BEFORE DANN
# Both source and target through baseline model — shows original domain gap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2A: t-SNE PLOTS — BEFORE DANN")
print("="*60)

plot_tsne_panel(
    source_feats_before, source_labels_arr,
    target_feats_before, target_labels_arr,
    stage_label = "Before DANN",
    target_name = TARGET_NAME,
    save_name   = f"tsne_before_DANN_{TARGET_NAME}.png"
)

# ─────────────────────────────────────────────────────────────────────────────
# DANN TRAINING (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DANN TRAINING")
print("="*60)

best_model_wts         = copy.deepcopy(model.state_dict())
best_val_loss          = float("inf")
epochs_without_improve = 0

for epoch in range(NUM_EPOCHS):
    lambda_grl = get_lambda_grl(epoch, NUM_EPOCHS)

    train_total, train_class, train_domain, train_acc = train_one_epoch_dann(
        model=model, source_loader=source_train_loader,
        target_loader=target_adapt_loader, optimizer=optimizer,
        class_criterion=class_criterion, domain_criterion=domain_criterion,
        lambda_grl=lambda_grl, domain_loss_weight=DOMAIN_LOSS_WEIGHT,
    )

    val_loss, val_acc, _, _, _, _ = evaluate_source_classification(
        model, source_val_loader, class_criterion
    )

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Total: {train_total:.4f} | Class: {train_class:.4f} | "
        f"Domain: {train_domain:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | lambda: {lambda_grl:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss          = val_loss
        best_model_wts         = copy.deepcopy(model.state_dict())
        epochs_without_improve = 0
        torch.save(model.state_dict(),
                   Path(FEATURE_OUTPUT_DIR) / f"best_ResNet50_20260616_dann_{TARGET_NAME}.pth")
    else:
        epochs_without_improve += 1

    if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch+1}.")
        break

model.load_state_dict(best_model_wts)

val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_source_classification(
    model, source_val_loader, class_criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
print_metrics("SOURCE VALIDATION RESULTS (DANN)", val_loss, val_metrics)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1B: FEATURE EXTRACTION AFTER DANN
# Target through DANN model → for movement, discriminability, probing
# Source through DANN model → for t-SNE only (same model = comparable spaces)
# source_feats_before remains the fixed reference for all other analyses
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1B: FEATURE EXTRACTION — AFTER DANN")
print("="*60)

target_feats_after, _, _ = \
    extract_features_from_loader(model, target_full_loader, f"{TARGET_NAME} (after DANN)")

save_features_to_excel(target_feats_after, target_labels_arr, target_paths,
                       f"{TARGET_NAME}_features_after_dann.xlsx")

# Source also extracted after DANN — used ONLY for t-SNE
# Both must go through the same model so the feature spaces are directly comparable
source_feats_dann, _, _ = \
    extract_features_from_loader(model, source_full_loader, "Source (after DANN, for t-SNE only)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B: t-SNE AFTER DANN
# Both source and target through the DANN model — same model, comparable spaces
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2B: t-SNE PLOTS — AFTER DANN")
print("="*60)

plot_tsne_panel(
    source_feats_dann, source_labels_arr,
    target_feats_after, target_labels_arr,
    stage_label = "After DANN",
    target_name = TARGET_NAME,
    save_name   = f"tsne_after_DANN_{TARGET_NAME}.png"
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURE MOVEMENT ANALYSIS
# source_feats_before is the fixed reference throughout
# before_gap = distance from source baseline to target before DANN
# after_gap  = distance from source baseline to target after DANN
# movement   = before_gap - after_gap (positive = DANN closed the gap)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: FEATURE MOVEMENT ANALYSIS")
print("="*60)

before_gaps, after_gaps, movement = compute_feature_movement(
    source_feats_before, target_feats_before, target_feats_after
)
movement_df = save_movement_to_excel(before_gaps, after_gaps, movement,
                                      f"feature_movement_{TARGET_NAME}.xlsx")

n_moved   = (movement > 0).sum()
n_unmoved = (movement <= 0).sum()
print(f"  Features where DANN closed the gap: {n_moved} / {len(movement)}")
print(f"  Features where DANN had no effect:  {n_unmoved} / {len(movement)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: DOMAIN DISCRIMINABILITY PER FEATURE
# Compares source baseline to target after DANN
# Low p-value = feature still domain-specific = DANN failed to align it
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: DOMAIN DISCRIMINABILITY (after DANN)")
print("="*60)

discrim_df = domain_discriminability(source_feats_before, target_feats_after, TARGET_NAME)

invariant_mask   = ~discrim_df['domain_specific'].values
domain_spec_mask = discrim_df['domain_specific'].values
print(f"  Domain-invariant features: {invariant_mask.sum()}")
print(f"  Domain-specific features:  {domain_spec_mask.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: PROBING CLASSIFIERS
# Source baseline features vs target after DANN features
# Tests whether features separate normal from abnormal
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: PROBING CLASSIFIERS")
print("="*60)

probe_df = run_probe_experiment(
    source_feats_before, source_labels_arr,
    target_feats_after, target_labels_arr,
    invariant_mask, TARGET_NAME
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5B: MMD ANALYSIS
# Overall, normal-only, and abnormal-only MMD before and after DANN
# source_feats_before is the fixed reference throughout
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5B: MMD ANALYSIS")
print("="*60)

mmd_df = run_mmd_analysis(
    source_feats  = source_feats_before,
    source_labels = source_labels_arr,
    target_before_feats = target_feats_before,
    target_after_feats  = target_feats_after,
    target_labels = target_labels_arr,
    target_name   = TARGET_NAME
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: GRAD-CAM VISUALIZATION")
print("="*60)

target_normal_paths    = [p for p, l in zip(target_paths, target_labels_arr) if l == 0][:8]
target_normal_labels   = [0] * len(target_normal_paths)
target_abnormal_paths  = [p for p, l in zip(target_paths, target_labels_arr) if l == 1][:8]
target_abnormal_labels = [1] * len(target_abnormal_paths)

if target_normal_paths:
    save_gradcam_grid(
        model, target_normal_paths, target_normal_labels,
        title     = f"Grad-CAM — {TARGET_NAME} Normal Images (After DANN)",
        save_name = f"gradcam_{TARGET_NAME}_normal.png",
        n_images  = len(target_normal_paths)
    )

if target_abnormal_paths:
    save_gradcam_grid(
        model, target_abnormal_paths, target_abnormal_labels,
        title     = f"Grad-CAM — {TARGET_NAME} Abnormal Images (After DANN)",
        save_name = f"gradcam_{TARGET_NAME}_abnormal.png",
        n_images  = len(target_abnormal_paths)
    )

source_normal_paths    = [p for p, l in zip(source_paths, source_labels_arr) if l == 0][:8]
source_normal_labels   = [0] * len(source_normal_paths)
source_abnormal_paths  = [p for p, l in zip(source_paths, source_labels_arr) if l == 1][:8]
source_abnormal_labels = [1] * len(source_abnormal_paths)

if source_normal_paths:
    save_gradcam_grid(
        model, source_normal_paths, source_normal_labels,
        title     = "Grad-CAM — Source Normal Images (After DANN)",
        save_name = "gradcam_source_normal.png",
        n_images  = len(source_normal_paths)
    )

if source_abnormal_paths:
    save_gradcam_grid(
        model, source_abnormal_paths, source_abnormal_labels,
        title     = "Grad-CAM — Source Abnormal Images (After DANN)",
        save_name = "gradcam_source_abnormal.png",
        n_images  = len(source_abnormal_paths)
    )

# ─────────────────────────────────────────────────────────────────────────────
# REPEATED TARGET TESTING (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REPEATED TARGET TESTING")
print("="*60)

repeat_results = []

for repeat_idx in range(N_REPEATS):
    rng              = random.Random(RANDOM_SEED + repeat_idx)
    sampled_normal   = rng.sample(remaining_normal_indices,   N_TEST_NORMAL)
    sampled_abnormal = rng.sample(remaining_abnormal_indices, N_TEST_ABNORMAL)
    test_indices     = sampled_normal + sampled_abnormal
    rng.shuffle(test_indices)

    test_loss, test_acc, y_test, y_test_pred, y_test_prob, _ = evaluate_target_test_subset(
        model=model, dataset=full_target_dataset,
        subset_indices=test_indices, criterion=class_criterion
    )
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
    repeat_results.append({"repeat": repeat_idx + 1, "loss": test_loss, **test_metrics})

    print(
        f"Repeat {repeat_idx+1:02d}/{N_REPEATS} | "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"Sens: {test_metrics['sensitivity']:.4f} | "
        f"Spec: {test_metrics['specificity']:.4f} | "
        f"AUROC: {test_metrics['auroc']:.4f}"
    )

metric_names = ["loss","accuracy","sensitivity","specificity",
                "precision","recall","f1","auroc"]

print("\n" + "=" * 70)
print(f"REPEATED TARGET TEST RESULTS — DANN ResNet50 → {TARGET_NAME}")
print("=" * 70)

for metric in metric_names:
    vals     = np.array([r[metric] for r in repeat_results], dtype=float)
    mean_val = np.nanmean(vals)
    sd_val   = np.nanstd(vals, ddof=1)
    print(f"{metric.capitalize():<12}: {mean_val:.4f} ± {sd_val:.4f}")

for cm_m in ["tn","fp","fn","tp"]:
    vals = np.array([r[cm_m] for r in repeat_results], dtype=float)
    print(f"{cm_m.upper():<12}: {vals.mean():.2f} ± {vals.std(ddof=1):.2f}")

results_csv = Path(FEATURE_OUTPUT_DIR) / f"dann_repeat_results_{TARGET_NAME}.csv"
pd.DataFrame(repeat_results).to_csv(results_csv, index=False)
print(f"\nRepeat results saved → {results_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"ALL OUTPUTS SAVED TO: {FEATURE_OUTPUT_DIR}")
print("="*60)
output_files = [
    "baseline_model.pth                          ← model before DANN",
    f"best_ResNet50_dann_{TARGET_NAME}.pth         ← best DANN model",
    "source_features_baseline.csv                ← source features before DANN (fixed reference)",
    f"{TARGET_NAME}_features_baseline.csv         ← target features before DANN",
    f"{TARGET_NAME}_features_after_dann.csv       ← target features after DANN",
    f"feature_movement_{TARGET_NAME}.xlsx         ← per-feature movement scores",
    f"domain_discriminability_{TARGET_NAME}.xlsx  ← per-feature domain specificity",
    f"probe_results_{TARGET_NAME}.xlsx            ← probing classifier F1 scores",
    f"tsne_before_DANN_{TARGET_NAME}.png          ← t-SNE before DANN (both through baseline model)",
    f"tsne_after_DANN_{TARGET_NAME}.png           ← t-SNE after DANN (both through DANN model)",
    f"gradcam_{TARGET_NAME}_normal.png            ← Grad-CAM normal images",
    f"gradcam_{TARGET_NAME}_abnormal.png          ← Grad-CAM abnormal images",
    "gradcam_source_normal.png                   ← Grad-CAM source normal",
    "gradcam_source_abnormal.png                 ← Grad-CAM source abnormal",
    f"mmd_analysis_{TARGET_NAME}.xlsx              ← MMD before/after DANN (overall + by class)",
    f"dann_repeat_results_{TARGET_NAME}.csv       ← 30-run performance metrics",
]
for f in output_files:
    print(f"  {f}")
