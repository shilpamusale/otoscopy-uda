# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/ADDA.py
# Role: ADDA (adversarial discriminative domain adaptation) baseline.
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


# Set these to preferences
TRAIN_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/eardrumDs/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/"

NUM_EPOCHS_SOURCE = 20
NUM_EPOCHS_ADDA = 20 # number of epochs the target encoder goes through
BATCH_SIZE = 16
LEARNING_RATE_SOURCE = 1e-4
LEARNING_RATE_ADDA = 1e-4
VAL_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# Number of unlabeled images for UDA
N_TARGET_ADAPT = 50

# repeated target testing
N_REPEATS = 30
N_TEST_NORMAL = 50
N_TEST_ABNORMAL = 50

USE_PRETRAINED = True
FREEZE_EARLY_LAYERS = False

# ADDA-specific
FEATURE_DIM = 2048 #size of features coming out of Resnet50
DISCRIMINATOR_HIDDEN = 512 #size of features after initial compression from discriminator/target encoder


# Seed = 42 for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# I use mac so device = mps
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# check images and verify nothing is broken, correct extensions,etc
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

        self.samples.extend(self._gather_images(normal_dir, 0, verify_images))
        self.samples.extend(self._gather_images(abnormal_dir, 1, verify_images))

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")

        print(f"Loaded {len(self.samples)} valid images from {root_dir}")
        print(f"  Normal:   {sum(1 for _, y in self.samples if y == 0)}")
        print(f"  Abnormal: {sum(1 for _, y in self.samples if y == 1)}")

        if self.bad_files:
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


# Augmentation and image pixel normalization to images of ImageNet which is what was used to train Resnet
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
#no augmentation for evaluating images
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# define performance metrics
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

#picks fix set of images for UDA (unlabelled target images)
def get_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    rng = random.Random(seed)
    n_normal = n_total // 2
    n_abnormal = n_total - n_normal
#make sure you have enough images in each database
    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError("Not enough target images to build fixed ADDA adaptation set.")

    picked_normal = rng.sample(normal_indices, n_normal)
    picked_abnormal = rng.sample(abnormal_indices, n_abnormal)
    picked = picked_normal + picked_abnormal
    rng.shuffle(picked)
    return picked


def get_remaining_target_pool(dataset, excluded_indices):
    excluded = set(excluded_indices)
    remaining_normal, remaining_abnormal = [], []

    for idx, (_, label) in enumerate(dataset.samples):
        if idx in excluded:
            continue
        if label == 0:
            remaining_normal.append(idx)
        else:
            remaining_abnormal.append(idx)

    return remaining_normal, remaining_abnormal


# build RN50 backbone
def build_resnet50_backbone(use_pretrained=True):
    if use_pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    if FREEZE_EARLY_LAYERS:
        for name, param in model.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False
#builds independent feature extractor and final classifier layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1]) #takes all Resnet50 features except the final 2 number logit
    classifier = nn.Linear(model.fc.in_features, 2) #this is what will get two numbers that are the logits for normal abnormal
    return feature_extractor, classifier #by returning separeate here we can use them in the initial source encoder and in making the target encoder


class SourceModel(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        self.encoder, self.classifier = build_resnet50_backbone(use_pretrained) #combined feature extraction and fc layer for phase 1

    def forward(self, x):
        feats = self.encoder(x) #run images through resnet50 and get features
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats) #get two class scores (logits)
        return logits

#again just combining the encoder and classifier for after the target encoder is built
class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
#same as sourcemodel above, its pushes images through the model and gets output two logits
    def forward(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits

#building discriminator, job is to look at the images and say target or source?
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512): #tells the discriminator the size of features coming in
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), #compresses the 1048 features to 512 of the most domain  relevant features
            nn.ReLU(), #allows discriminator to learn non linear domain features
            nn.Dropout(0.3), #prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2), #compresses features to 256 domain relevant features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2) # output logits for domain decision
        )

    def forward(self, x):
        return self.net(x) #gives the domain logits

# evaluation function
def evaluate_classifier(model, loader, criterion):
    model.eval() #enter evaluation mode
    running_loss = 0.0
    all_labels, all_preds, all_probs, all_paths = [], [], [], [] #gets labels, prediction, probabilities of abnormal, and paths for each image and creates lists

    with torch.no_grad(): #disables pytorch graph building so there is nothing updated in terms of weights, gradients, etc during evaluation
        for images, labels, paths in loader: #for loop that gives batch of images, labels, and paths
            images = images.to(device) #move images to the CPU
            labels = labels.to(device)

            logits = model(images) #images go through model and produce class logits
            loss = criterion(logits, labels) #cross entropy loss for class prediction and labels

            probs = torch.softmax(logits, dim=1)[:, 1] #convert logits to probability
            preds = torch.argmax(logits, dim=1) #make a decision based on softmax probability

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths


# import datasets
full_source_dataset = BinaryFolderDataset(TRAIN_ROOT, transform=None, verify_images=True)
full_target_dataset = BinaryFolderDataset(TARGET_ROOT, transform=None, verify_images=True)
#sets number of images for validation and training
source_train_size = int((1 - VAL_SPLIT) * len(full_source_dataset))
source_val_size = len(full_source_dataset) - source_train_size
#random seed for the images
generator = torch.Generator().manual_seed(RANDOM_SEED)
source_train_subset, source_val_subset = random_split(
    full_source_dataset,
    [source_train_size, source_val_size],
    generator=generator
)

source_train_dataset = TransformedSubset(source_train_subset, transform=train_transform)
source_val_dataset = TransformedSubset(source_val_subset, transform=eval_transform)

target_adapt_indices = get_balanced_fixed_target_adapt_set(
    full_target_dataset,
    n_total=N_TARGET_ADAPT,
    seed=RANDOM_SEED
)

target_adapt_subset = Subset(full_target_dataset, target_adapt_indices)
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=train_transform)
#sets it so that everything that is not within the 75 or whatever # for adapation are put into the pool for repeated testing later
remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset,
    excluded_indices=target_adapt_indices
)
#prints all safety checks for images we have for training, testing, adaptation etc
print("\nFixed ADDA target adaptation set:")
print(f"  Total: {len(target_adapt_indices)}")
print(f"  Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"  Remaining abnormal for testing: {len(remaining_abnormal_indices)}")
#if not enough images given the adaptation number it will tell you
if len(remaining_normal_indices) < N_TEST_NORMAL:
    raise ValueError("Not enough normal target images left for testing.")
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError("Not enough abnormal target images left for testing.")
#these are just dataloaders
source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# First step of ADDA here, train the source model like you would any other CNN
source_model = SourceModel(use_pretrained=USE_PRETRAINED).to(device)

class_criterion = nn.CrossEntropyLoss() #evaluate with cross entropy loss
source_optimizer = torch.optim.Adam( #adaps optimizer, standard
    filter(lambda p: p.requires_grad, source_model.parameters()), #gets the learned conditions from the model
    lr=LEARNING_RATE_SOURCE #overall it returns the weights that are to be altered by the adams optimizer
)

best_source_wts = copy.deepcopy(source_model.state_dict()) #keep best model weights during training for validation
best_val_loss = float("inf") #makes sure the first epoch is counted as an improvement
epochs_without_improvement = 0

print("\nStarting source training...\n")

for epoch in range(NUM_EPOCHS_SOURCE):
    source_model.train() #training mode
    running_loss = 0.0
    train_labels, train_preds = [], [] #

    for images, labels, _ in source_train_loader: #for source images (no path)
        images = images.to(device) #move to GPU
        labels = labels.to(device) #labels to GPU

        source_optimizer.zero_grad() #clear gradients from previous batch of images
        logits = source_model(images) #get logits for images
        loss = class_criterion(logits, labels) #cross entropy between predictions and true labels
        loss.backward() #backpropagatoin to update weights for model, this is only for source so no discriminator yet or GRL like in DANN
        source_optimizer.step() #updates weights

        preds = torch.argmax(logits, dim=1) #makes hard prediction for each image again
        running_loss += loss.item() * images.size(0)
        train_labels.extend(labels.cpu().numpy()) #add labels andpredictions to running list (next line is predictions)
        train_preds.extend(preds.cpu().numpy())

    train_loss = running_loss / len(source_train_loader.dataset)
    train_acc = accuracy_score(train_labels, train_preds)

    val_loss, val_acc, _, _, _, _ = evaluate_classifier(source_model, source_val_loader, class_criterion)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS_SOURCE}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
#early stopping if statement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_source_wts = copy.deepcopy(source_model.state_dict())
        epochs_without_improvement = 0
        torch.save(source_model.state_dict(), "best_source_resnet50_adda.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
        break

source_model.load_state_dict(best_source_wts) #makes sure we use the best weights during the training loop

val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_classifier(
    source_model, source_val_loader, class_criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
print_metrics("SOURCE VALIDATION RESULTS", val_loss, val_metrics)


# Second step: actually implemeneting ADDA
# freeze source encoder
# copy source encoder to target encoder
# train discriminator and target encoder adversarially
source_encoder = copy.deepcopy(source_model.encoder).to(device)
target_encoder = copy.deepcopy(source_model.encoder).to(device) #need to make a copy of the original trained Resnet50 on source images to be updated based on fool loss to make features match between source and target
classifier_head = copy.deepcopy(source_model.classifier).to(device) #classifier that gets features from target encoder to make predictions
discriminator = DomainDiscriminator(feature_dim=FEATURE_DIM, hidden_dim=DISCRIMINATOR_HIDDEN).to(device) #building discriminator that expects 2048 input features, which come from target encoder

for param in source_encoder.parameters():  #never updates weights of original source encoder/model
    param.requires_grad = False
for param in classifier_head.parameters(): # source classifier is not changed ever
    param.requires_grad = False

disc_criterion = nn.CrossEntropyLoss() #cross entropy loss is used for discriminator and target encoder fooling
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_ADDA) #updates discriminator weights
target_optimizer = torch.optim.Adam(target_encoder.parameters(), lr=LEARNING_RATE_ADDA) #updates target encoder weights
#these optimisers being separate is really important because they should not ever update each other
print("\nStarting ADDA adaptation...\n")

for epoch in range(NUM_EPOCHS_ADDA): #10 fo rus
    source_encoder.eval() #start with the source encoder
    classifier_head.eval() #use classifer built in phase 1
    target_encoder.train() #outputing features for discriminator
    discriminator.train() #learning to detect domains
#also no early stopping in ADDA because its all about features becoming similar, there is no actual testing/validation
    running_disc_loss = 0.0
    running_tgt_loss = 0.0

    source_iter = iter(source_train_loader) #iteraters allow for manual calling of the next batch
    target_iter = iter(target_adapt_loader)
    num_steps = min(len(source_train_loader), len(target_adapt_loader))#there are only enough loops run per epoch so that each target image is seen once
#basically 5-6 runs through with only 75 images, even less with minimal images, WE MAY WANT TO GET RID OF THIS, DISCUSS WITH DEPAUL GROUP
    for _ in range(num_steps): #number of steps equals unlabelled target images/batch size
        try:
            src_images, _, _ = next(source_iter) #get next batch
        except StopIteration: #unless batches have run out
            source_iter = iter(source_train_loader)
            src_images, _, _ = next(source_iter)

        try:
            tgt_images, _, _ = next(target_iter)
        except StopIteration: #unless batches run out, which it will before source images of course
            target_iter = iter(target_adapt_loader)
            tgt_images, _, _ = next(target_iter)
#source and target images into GPU ready to go through encoder
        src_images = src_images.to(device)
        tgt_images = tgt_images.to(device)


        # train discriminator
        with torch.no_grad():
            src_feat = torch.flatten(source_encoder(src_images), 1) #source images go through source encoder
        tgt_feat = torch.flatten(target_encoder(tgt_images), 1) #target images go through code encoder

        src_domain_labels = torch.zeros(src_feat.size(0), dtype=torch.long, device=device) # discriminator getting true labels
        tgt_domain_labels = torch.ones(tgt_feat.size(0), dtype=torch.long, device=device) #

        disc_optimizer.zero_grad() #clear gradients from last batch

        src_domain_logits = discriminator(src_feat) #source images go through discriminator and give logit for source vs domain
        tgt_domain_logits = discriminator(tgt_feat.detach()) #same for target images

        disc_loss_src = disc_criterion(src_domain_logits, src_domain_labels) #how wrong is it at detectin source images
        disc_loss_tgt = disc_criterion(tgt_domain_logits, tgt_domain_labels) #how wrong is it as detecting target images
        disc_loss = 0.5 * (disc_loss_src + disc_loss_tgt) #discriminator loss is equal weight of both domains, bc you dont want it to be better at detecting one over the other

        disc_loss.backward() #backpropagation through discriminator only (trying to get better at telling source and target differences)
        disc_optimizer.step() #update weights with adam optimizer (only for discriminator)


        # train target encoder to fool discriminator
        target_optimizer.zero_grad() #clears target encoder previous gradients

        tgt_feat = torch.flatten(target_encoder(tgt_images), 1) #runs target images through encoder and flattens to 2048 features
        tgt_domain_logits = discriminator(tgt_feat) #give logits for target images from discriminator

        # fool discriminator: want target to look like source (label 0)
        fool_labels = torch.zeros(tgt_feat.size(0), dtype=torch.long, device=device) #purposely labeling target images as source images
        tgt_loss = disc_criterion(tgt_domain_logits, fool_labels) #new target loss is the loss for mislabeled images

        tgt_loss.backward()
        target_optimizer.step() #all target encoder weights shifted to make more source like images

        running_disc_loss += disc_loss.item()
        running_tgt_loss += tgt_loss.item() #want target encoder loss to decrease

    print(
        f"ADDA Epoch [{epoch + 1}/{NUM_EPOCHS_ADDA}] "
        f"Discriminator Loss: {running_disc_loss / num_steps:.4f} | "
        f"Target Encoder Loss: {running_tgt_loss / num_steps:.4f}"
    )

# final adapted model = target encoder + source classifier
adapted_model = EncoderWithClassifier(target_encoder, classifier_head).to(device)

# Source validation on adapted model, not needed but good
src_val_loss, src_val_acc, src_y, src_pred, src_prob, _ = evaluate_classifier(
    adapted_model, source_val_loader, class_criterion
)
src_val_metrics = compute_metrics(src_y, src_pred, src_prob)
print_metrics("ADAPTED MODEL ON SOURCE VALIDATION", src_val_loss, src_val_metrics)

# Testing on 30 repeated sets of 100 target images
def evaluate_target_subset(model, dataset, subset_indices, criterion):
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

    test_loss, test_acc, y_test, y_test_pred, y_test_prob, _ = evaluate_target_subset(
        adapted_model,
        full_target_dataset,
        test_indices,
        class_criterion
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

# overall performance summary
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
print("REPEATED TARGET TEST RESULTS (ADDA WITH RESNET50 BACKBONE)")
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

adapt_labels = [full_target_dataset.samples[i][1] for i in target_adapt_indices]
print("\nFixed ADDA target adaptation set composition:")
print(f"  Normal used for ADDA adaptation:   {sum(1 for y in adapt_labels if y == 0)}")
print(f"  Abnormal used for ADDA adaptation: {sum(1 for y in adapt_labels if y == 1)}")