# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/CORAL_Resnet50.py
# Role: Train the CORAL (correlation alignment) adaptation model.
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

#paper for CORAL: Deep CORAL: Correlation Alignment for Deep Domain Adaptation
# settings, most predefined for our purpose
TRAIN_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/eardrumDs/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/"

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# CORAL setup
N_TARGET_ADAPT = 75
N_REPEATS = 30
N_TEST_NORMAL = 50
N_TEST_ABNORMAL = 50
CORAL_LOSS_WEIGHT = 0.1   # plays into how much you want CORAL loss to play into backpropagation

USE_PRETRAINED = True
FREEZE_BACKBONE = False


# standard seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# upload datasets
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


# audmentation and normalization of image intensities to imageNet
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


# Model backbone (resnet50, mobilnetV2, Yolo, whatever we decide)
class ResNet50FeatureClassifier(nn.Module): #
    def __init__(self, use_pretrained=True, freeze_backbone=False): #standard
        super().__init__()

        if use_pretrained:
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50(weights=None)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
#feature extractor and final layer classifier
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, 2)

        if freeze_backbone:
            for param in self.classifier.parameters():
                param.requires_grad = True
#runs feature through backbone, flatter to 1x2048, and finally a logit with softmax
    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features #returns logits AND features because features are needed for covariance matrices.


#  CORAL loss after alignment features
def coral_loss(source_features, target_features):
    d = source_features.size(1)
#subtract the mean of features so everything is relative to zero. i.e. normalized
    source_centered = source_features - source_features.mean(dim=0, keepdim=True) #last part calculates the mean across the batch for each feature
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
# calculates the covariancefor each domain
    source_cov = (source_centered.t() @ source_centered) / max(1, source_features.size(0) - 1) #divide by batchsize-1
    target_cov = (target_centered.t() @ target_centered) / max(1, target_features.size(0) - 1)# this is just a correction so that you get an unbiased estimate of the true population covariance

    loss = torch.mean((source_cov - target_cov) ** 2) #takes subtraction of the two matrices in their respective features, squares it so it is positive, and thus large differences are affected more than lower ones
    loss = loss / (4 * d * d) #directly from original CORAL paper, with dimension of 2048 this is going to be very small
    return loss
#torch mean above  gives the average squared difference between source and target covariance structures

# define metrics for evaluation
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


# helper functions to ensure we have the images needed for adapation and testing
def make_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    rng = random.Random(seed)

    n_normal = n_total // 2
    n_abnormal = n_total - n_normal

    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError(f"Not enough target images to build adaptation set of {n_total}.")

    sampled_normal = rng.sample(normal_indices, n_normal)
    sampled_abnormal = rng.sample(abnormal_indices, n_abnormal)

    adapt_indices = sampled_normal + sampled_abnormal
    rng.shuffle(adapt_indices)
    return adapt_indices

#rest of target images for testing
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


# datasets to GPU
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
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=train_transform)

remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset,
    excluded_indices=target_adapt_indices
)

print("\nFixed target adaptation set:")
print(f"  Total: {len(target_adapt_indices)}")
print(f"  Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"  Remaining abnormal for testing: {len(remaining_abnormal_indices)}")

if len(remaining_normal_indices) < N_TEST_NORMAL:
    raise ValueError("Not enough normal images left for repeated testing.")
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError("Not enough abnormal images left for repeated testing.")

source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# adams optimizer, cross entropy loss, etc
model = ResNet50FeatureClassifier(
    use_pretrained=USE_PRETRAINED,
    freeze_backbone=FREEZE_BACKBONE
).to(device)

class_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


# training and evaluation
def train_one_epoch_coral(model, source_loader, target_loader, optimizer,
                          class_criterion, coral_weight=0.1):
    model.train()

    running_total_loss = 0.0
    running_class_loss = 0.0
    running_coral_loss = 0.0

    source_labels_all = []
    source_preds_all = []

    target_iter = iter(target_loader)

    for source_images, source_labels, _ in source_loader:
        try:
            target_images, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_images, _, _ = next(target_iter)

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        source_logits, source_features = model(source_images)
        _, target_features = model(target_images)

        cls_loss = class_criterion(source_logits, source_labels)
        c_loss = coral_loss(source_features, target_features)

        total_loss = cls_loss + coral_weight * c_loss
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item() * source_images.size(0)
        running_class_loss += cls_loss.item() * source_images.size(0)
        running_coral_loss += c_loss.item() * source_images.size(0)

        source_preds = torch.argmax(source_logits, dim=1)
        source_labels_all.extend(source_labels.detach().cpu().numpy())
        source_preds_all.extend(source_preds.detach().cpu().numpy())

    epoch_total_loss = running_total_loss / len(source_loader.dataset)
    epoch_class_loss = running_class_loss / len(source_loader.dataset)
    epoch_coral_loss = running_coral_loss / len(source_loader.dataset)
    epoch_acc = accuracy_score(source_labels_all, source_preds_all)

    return epoch_total_loss, epoch_class_loss, epoch_coral_loss, epoch_acc


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

            logits, _ = model(images)
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


# training loop (based on best during initial training)
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting CORAL training...\n")

for epoch in range(NUM_EPOCHS):
    train_total_loss, train_class_loss, train_coral_loss, train_acc = train_one_epoch_coral(
        model=model,
        source_loader=source_train_loader,
        target_loader=target_adapt_loader,
        optimizer=optimizer,
        class_criterion=class_criterion,
        coral_weight=CORAL_LOSS_WEIGHT
    )

    val_loss, val_acc, _, _, _, _ = evaluate_classifier(
        model, source_val_loader, class_criterion
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Total: {train_total_loss:.4f} | "
        f"Train Class: {train_class_loss:.4f} | "
        f"Train CORAL: {train_coral_loss:.6f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_resnet50_coral.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
        break

model.load_state_dict(best_model_wts)


# validation results for source images after initial training
val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_classifier(
    model, source_val_loader, class_criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
print_metrics("SOURCE VALIDATION RESULTS", val_loss, val_metrics)


# testing on 30 sets of 100 images
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
        criterion=class_criterion
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


# overall summary
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
print("REPEATED TARGET TEST RESULTS (DEEP CORAL WITH RESNET50 BACKBONE)")
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


# prints composition of adapt set
adapt_labels = [full_target_dataset.samples[i][1] for i in target_adapt_indices]
print("\nFixed target adaptation set composition:")
print(f"  Normal used for CORAL adaptation:   {sum(1 for y in adapt_labels if y == 0)}")
print(f"  Abnormal used for CORAL adaptation: {sum(1 for y in adapt_labels if y == 1)}")