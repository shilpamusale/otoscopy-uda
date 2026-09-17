# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/BatchNorm_Resnet50.py
# Role: BatchNorm-adaptation baseline (adapt BN statistics to the target domain).
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
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Code fully commented by JV
# Set the items for training the source and UDA structure

TRAIN_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/eardrumDs/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/"

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# BatchNorm adaptation setup
N_TARGET_ADAPT = 75
N_REPEATS = 30
N_TEST_NORMAL = 50
N_TEST_ABNORMAL = 50

USE_PRETRAINED = True
FREEZE_BACKBONE = False

# seed placed for reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# DEVICE
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# images, datasets, extensions, etc. these are just all safety measures
class BinaryFolderDataset(Dataset):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root_dir, transform=None, verify_images=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.bad_files = []

        normal_dir = self.root_dir / "normal"
        abnormal_dir = self.root_dir / "abnormal"

        if not normal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {normal_dir}")
        if not abnormal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {abnormal_dir}")

        self.samples.extend(self._gather_images(normal_dir, label=0, verify_images=verify_images))
        self.samples.extend(self._gather_images(abnormal_dir, label=1, verify_images=verify_images))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")

        print(f"Loaded {len(self.samples)} valid images from {root_dir}")
        print(f"  Normal:   {sum(1 for _, y in self.samples if y == 0)}")
        print(f"  Abnormal: {sum(1 for _, y in self.samples if y == 1)}")

        if len(self.bad_files) > 0:
            print(f"Skipped {len(self.bad_files)} unreadable files:")
            for f in self.bad_files[:50]:
                print(f"  {f}")
            if len(self.bad_files) > 50:
                print("  ...")

    def _is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.load()
            return True
        except Exception:
            return False

    def _gather_images(self, folder, label, verify_images=True):
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
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        _, label, img_path = self.subset[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# augmentation and normalization of images to imagenet pixel intensities
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# standard resnet50 backdone used, pretrained weights placed
def build_resnet50_classifier(use_pretrained=True, freeze_backbone=False):
    if use_pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model

# define metrics we want to compute
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_metrics(title, loss, metrics):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Loss:        {loss:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"AUROC:       {metrics['auroc']:.4f}")
    print(f"TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TP: {metrics['tp']}")

# helpful functions so that everything runs smoothly
def make_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    rng = random.Random(seed)

    n_normal = n_total // 2
    n_abnormal = n_total - n_normal

    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError(f"Not enough target images to build fixed adaptation set of {n_total}.")

    sampled_normal = rng.sample(normal_indices, n_normal)
    sampled_abnormal = rng.sample(abnormal_indices, n_abnormal)

    adapt_indices = sampled_normal + sampled_abnormal
    rng.shuffle(adapt_indices)
    return adapt_indices


def get_remaining_target_pool(dataset, excluded_indices):
    excluded_set = set(excluded_indices)
    remaining_normal = []
    remaining_abnormal = []

    for idx, (_, label) in enumerate(dataset.samples):
        if idx in excluded_set:
            continue
        if label == 0:
            remaining_normal.append(idx)
        else:
            remaining_abnormal.append(idx)

    return remaining_normal, remaining_abnormal


def set_batchnorm_to_adapt(model): # this is essential, Only puts BN layers in train mode, other weights cannot update
    model.eval() #this line first puts all layers (convolutional layers, dropouts, BN, etc) in eval mode so they cannot be changes
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train() #if function says if it is a BN 2D layer, put it in training mode (turns BN adaptation on)
    return model
#so now only batch normalization layers can be altered

def adapt_batchnorm(model, target_loader):
    model = set_batchnorm_to_adapt(model) #calls on function we just made above (BN model)

    with torch.no_grad(): #this is needed to explicitly say do not update gradients or weights.
        for images, _, _ in target_loader: # loops through all target adaptation images, no path or labels but not really needed for BN adaptation
            images = images.to(device)
            _ = model(images) #this line might normally give logits, but we do not care about predictions, etc in BN adaptation, only updating those layers is important.

    model.eval() #model goes back into evaluation mode after the target images run through it, so no weights can be changed anymore
    return model #new model is with updated BN weights

# load in the data files
full_source_dataset = BinaryFolderDataset(TRAIN_ROOT, transform=None, verify_images=True)
full_target_dataset = BinaryFolderDataset(TARGET_ROOT, transform=None, verify_images=True)

source_train_size = int((1 - VAL_SPLIT) * len(full_source_dataset))
source_val_size = len(full_source_dataset) - source_train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
source_train_subset, source_val_subset = random_split(
    full_source_dataset,
    [source_train_size, source_val_size],
    generator=generator
)

source_train_dataset = TransformedSubset(source_train_subset, transform=train_transform)
source_val_dataset = TransformedSubset(source_val_subset, transform=eval_transform)

target_adapt_indices = make_balanced_fixed_target_adapt_set(
    full_target_dataset,
    n_total=N_TARGET_ADAPT,
    seed=RANDOM_SEED
)
target_adapt_subset = Subset(full_target_dataset, target_adapt_indices)
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=eval_transform)

remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset,
    excluded_indices=target_adapt_indices
)

print("\nFixed target BatchNorm adaptation set:")
print(f"  Total: {len(target_adapt_indices)}")
print(f"  Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"  Remaining abnormal for testing: {len(remaining_abnormal_indices)}")

if len(remaining_normal_indices) < N_TEST_NORMAL:
    raise ValueError("Not enough normal images left for repeated testing.")
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError("Not enough abnormal images left for repeated testing.")

source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# loss function, optimizer, etc. This is all standard with other codes used
model = build_resnet50_classifier(
    use_pretrained=USE_PRETRAINED,
    freeze_backbone=FREEZE_BACKBONE
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# setting up training and evaluation
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate_classifier(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths

# training loop
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting baseline training...\n")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, source_train_loader, criterion, optimizer)
    val_loss, val_acc, _, _, _, _ = evaluate_classifier(model, source_val_loader, criterion)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_resnet50_baseline_bnadapt.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
        break

model.load_state_dict(best_model_wts)

# source validation before BN adaptation
val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_classifier(
    model, source_val_loader, criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
print_metrics("SOURCE VALIDATION BEFORE BN ADAPTATION", val_loss, val_metrics)

# BN adaptation with set number of unlabelled target images
print("\nAdapting BatchNorm statistics on fixed target set...\n")
model = adapt_batchnorm(model, target_adapt_loader)

# source testing after CN updates, just to make sure you avoid forgetting the original source images.
val_loss_bn, val_acc_bn, y_val_bn, y_val_pred_bn, y_val_prob_bn, _ = evaluate_classifier(
    model, source_val_loader, criterion
)
val_metrics_bn = compute_metrics(y_val_bn, y_val_pred_bn, y_val_prob_bn)
print_metrics("SOURCE VALIDATION AFTER BN ADAPTATION", val_loss_bn, val_metrics_bn)

# repeat testing on target sets
def evaluate_target_test_subset(model, dataset, subset_indices, criterion):
    subset = Subset(dataset, subset_indices)
    subset_dataset = TransformedSubset(subset, transform=eval_transform)
    loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return evaluate_classifier(model, loader, criterion)

repeat_results = []

print("\nStarting repeated target testing...\n")

for repeat_idx in range(N_REPEATS):
    rng = random.Random(RANDOM_SEED + repeat_idx)

    sampled_normal = rng.sample(remaining_normal_indices, N_TEST_NORMAL)
    sampled_abnormal = rng.sample(remaining_abnormal_indices, N_TEST_ABNORMAL)

    test_indices = sampled_normal + sampled_abnormal
    rng.shuffle(test_indices)

    test_loss, test_acc, y_test, y_test_pred, y_test_prob, _ = evaluate_target_test_subset(
        model=model,
        dataset=full_target_dataset,
        subset_indices=test_indices,
        criterion=criterion
    )

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    repeat_results.append({
        "repeat": repeat_idx + 1,
        "loss": test_loss,
        **test_metrics
    })

    print(
        f"Repeat {repeat_idx + 1:02d}/{N_REPEATS} | "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"Sens: {test_metrics['sensitivity']:.4f} | "
        f"Spec: {test_metrics['specificity']:.4f} | "
        f"AUROC: {test_metrics['auroc']:.4f}"
    )

# performance summary overall
metric_names = [
    "loss",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "auroc",
]

print("\n" + "=" * 70)
print("REPEATED TARGET TEST RESULTS (RESNET50 + BATCHNORM ADAPTATION)")
print("=" * 70)

for metric in metric_names:
    values = np.array([r[metric] for r in repeat_results], dtype=float)
    mean_val = np.nanmean(values)
    sd_val = np.nanstd(values, ddof=1)
    print(f"{metric.capitalize():<12}: {mean_val:.4f} ± {sd_val:.4f}")

for cm_metric in ["tn", "fp", "fn", "tp"]:
    values = np.array([r[cm_metric] for r in repeat_results], dtype=float)
    mean_val = np.mean(values)
    sd_val = np.std(values, ddof=1)
    print(f"{cm_metric.upper():<12}: {mean_val:.2f} ± {sd_val:.2f}")

# just prints composition of each of the sets
adapt_labels = [full_target_dataset.samples[i][1] for i in target_adapt_indices]
print("\nFixed target BatchNorm adaptation set composition:")
print(f"  Normal used for BN adaptation:   {sum(1 for y in adapt_labels if y == 0)}")
print(f"  Abnormal used for BN adaptation: {sum(1 for y in adapt_labels if y == 1)}")