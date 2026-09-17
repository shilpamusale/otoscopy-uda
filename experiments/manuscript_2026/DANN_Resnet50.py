# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/DANN_Resnet50.py
# Role: Train the DANN (domain-adversarial neural network) adaptation model.
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
import math #this library is needed for scheduled increases in lambda for the GRL
import random
from itertools import cycle #this is to continue loop after images have from out in target images, bc each epoch needs target images
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Function #needed for GRL
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

# USER SETTINGS
TRAIN_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/eardrumDs/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/"

NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# DANN setup
N_TARGET_ADAPT = 50
N_REPEATS = 30
N_TEST_NORMAL = 50
N_TEST_ABNORMAL = 50
DOMAIN_LOSS_WEIGHT = 0.5

USE_PRETRAINED = True
FREEZE_BACKBONE = False

# Seed 42 for reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

# Like in other codes this is for mac, as pytorch will recognize the mps on mac.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load images from datasets into list
class BinaryFolderDataset(Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root_dir, transform=None, verify_images=True):
        self.root_dir = Path(root_dir)
        self.transform = transform #this is set to none above
        self.samples = [] #creating list of samples
        self.bad_files = [] #creates list of corrupt files if verify_images function says they are broken. All iamges are good on my computer at least

        normal_dir = self.root_dir / "normal"
        abnormal_dir = self.root_dir / "abnormal"
# error if the normal and abnormal folders are not seen
        if not normal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {normal_dir}")
        if not abnormal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {abnormal_dir}")
#add all samples to self.samples
        self.samples.extend(self._gather_images(normal_dir, label=0, verify_images=verify_images))
        self.samples.extend(self._gather_images(abnormal_dir, label=1, verify_images=verify_images))
#error in case there are no samples added to list
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
#should print how many images are found in each folder/dataset. Make sure these match what you are uploading.
        print(f"Loaded {len(self.samples)} valid images from {root_dir}")
        print(f"  Normal:   {sum(1 for _, y in self.samples if y == 0)}")
        print(f"  Abnormal: {sum(1 for _, y in self.samples if y == 1)}")
#tells you how many images that are corrupted. If any, make sure that your files match the types listed earlier.
        if len(self.bad_files) > 0:
            print(f"Skipped {len(self.bad_files)} unreadable files:")
            for f in self.bad_files[:50]:
                print(f"  {f}")
            if len(self.bad_files) > 50:
                print("  ...")

    def _is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB") #convert to RGB image
                img.load() #uses PIL library in opening to actually load the image into CPU
            return True
        except Exception:
            return False #instead of just crashing just returns false. make sure no images return as false.

    def _gather_images(self, folder, label, verify_images=True):
        items = [] #creates list of all the valid images from _gather_images
        for path in folder.rglob("*"): #search all subfolders in the abnormal folder. rglob function does this.
            if path.is_file() and path.suffix.lower() in self.IMG_EXTENSIONS: #filters out non valid image extensions, and converts JPG to jpg
                if verify_images:
                    if self._is_valid_image(path):
                        items.append((str(path), label)) #add images to the folder if the image is valid with correct extension
                    else:
                        self.bad_files.append(str(path))
                else:
                    items.append((str(path), label)) #not done because verify_images = true for us
        return items

    def __len__(self):
        return len(self.samples) #counts how many images are within the dataset that did not get removed, needed to know how many batches in one epoch

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path #unpacks images and returns the image, label, and path.


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

# Standard augmentation for training images
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #this normalizes to pixel intensities of imageNet, which the weights of RN50 are trained on.
                         std=[0.229, 0.224, 0.225]),
])
#testing images are not augmented
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# GradientReversalFunction is a pytorch auto function for reversing sign of backprop gradient
class GradientReversalFunction(Function):
    @staticmethod #means this happens regardless of the class, pytorch requires this
    def forward(ctx, x, lambda_grl): #ctx is a context function to store information from forward and backward pass through feature extractor. x is just the feature at hand
        ctx.lambda_grl = lambda_grl #this line ensures that the backward pass knows what lambda is for that particular epoch.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output): #gradient output here is the gradient for a particular feature coming from the domain discriminator
        return -ctx.lambda_grl * grad_output, None #CRITICAL LINE - takes negative lambda for that context and multiplies by gradient from domain discriminator that wants to optimize source vs target loss.
#the None here means do not update Lambda on own for GRL, it updates after each epoch.yes continue

#this just makes GradientReversalFunction, our written GRL, into a pytorch function so the GRL can be a layer before the feature extractor later
class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_grl=1.0): #
        return GradientReversalFunction.apply(x, lambda_grl) #  runs forward pass, saves lambda, and saves for backpass
#GradientReversalLayer is what is actually put into Resnet50. GradientReversalFunction (line 172) is what does the math for the gradients
# Resnet50 with DANN
class ResNet50DANN(nn.Module):
    def __init__(self, use_pretrained=True, freeze_backbone=False):
        super().__init__()

        if use_pretrained: #which is true because we are fine tuning weights already learned from imageNet
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            backbone = models.ResNet50(weights=None)

        if freeze_backbone: #ignore
            for param in backbone.parameters():
                param.requires_grad = False
#nn.sequential is a function in pytorch that lets you put output of one layer into the next.
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  #back.children tells it to take all features except the last, which is a classification of many imagenet scores
        self.feature_dim = backbone.fc.in_features #get features from resnet50 backbone, but not the final classification layer (FC)

        self.class_classifier = nn.Linear(self.feature_dim, 2) #makes final classifier as two classes Normal 0 and Abnormal 1, Class Classifier!

        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential( #DOMAIN CLASSIFIER
            nn.Linear(self.feature_dim, 256), #compresses domain discriminator features to 256
            nn.ReLU(), #adds non-linear learning to discriminator so that it can learn more patterns of domains
            nn.Dropout(0.3), #randomly silences 30# of neurons to prevent overfitting
            nn.Linear(256, 2) #use the features to output source or target
        )
#ignore because this is false
        if freeze_backbone:
            for param in self.class_classifier.parameters():
                param.requires_grad = True
            for param in self.domain_classifier.parameters():
                param.requires_grad = True

    def extract_features(self, x):
        x = self.feature_extractor(x) #run images through feature extractor of RN50
        x = torch.flatten(x, 1) #get a 1D feature backbone
        return x #return the backbone

    def forward(self, x, lambda_grl=1.0):
        features = self.extract_features(x)
        class_logits = self.class_classifier(features) #give logits for source predictions
        reversed_features = self.grl(features, lambda_grl) #send features through GRL
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits

# METRICS
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

# Picks a fixed unlabeled target adaptation set.
def make_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    rng = random.Random(seed)

    n_normal = n_total // 2
    n_abnormal = n_total - n_normal

    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError(
            f"Not enough target images to build the fixed adaptation set of {n_total} "
            f"(need about {n_normal} normal and {n_abnormal} abnormal)."
        )

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

# scheduled multiplication of domain loss for gradual adaptation (allows us to confirm source images are also being learned)
def get_lambda_grl(epoch_idx, num_epochs):
    p = float(epoch_idx) / max(1, num_epochs - 1) #float gives decimal not integer
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0 #allows sigmoidal increase in lambda, or increase in domain discriminators say in gradients.

# load in source and target sets
full_source_dataset = BinaryFolderDataset(TRAIN_ROOT, transform=None, verify_images=True)
full_target_dataset = BinaryFolderDataset(TARGET_ROOT, transform=None, verify_images=True)

# Source split: 90% train / 10% val
source_train_size = int((1 - VAL_SPLIT) * len(full_source_dataset))
source_val_size = len(full_source_dataset) - source_train_size

generator = torch.Generator().manual_seed(RANDOM_SEED) #seed here ensures same images each run of script
source_train_subset, source_val_subset = random_split(
    full_source_dataset,
    [source_train_size, source_val_size],
    generator=generator
)

source_train_dataset = TransformedSubset(source_train_subset, transform=train_transform) #training gets augmentation
source_val_dataset = TransformedSubset(source_val_subset, transform=eval_transform) #validation set does not

# Grab the same images from target set as numbered by N_Target_adapt
target_adapt_indices = make_balanced_fixed_target_adapt_set(
    full_target_dataset,
    n_total=N_TARGET_ADAPT,
    seed=RANDOM_SEED
)
target_adapt_subset = Subset(full_target_dataset, target_adapt_indices)
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=train_transform) #adaptation set gets augmentation

# Remaining target pool for repeated testing
remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset,
    excluded_indices=target_adapt_indices #gets rid of those use for UDA or DANN in this case and says the rest are for repeated testing
)

print("\nFixed target adaptation set:")
print(f"  Total: {len(target_adapt_indices)}")
print(f"  Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"  Remaining abnormal for testing: {len(remaining_abnormal_indices)}")
#safety checks to make sure we have enough images for repeated testing after removing the adaptation images
if len(remaining_normal_indices) < N_TEST_NORMAL:
    raise ValueError(
        f"After removing the 50 adaptation images, only {len(remaining_normal_indices)} normal images remain, "
        f"but {N_TEST_NORMAL} are needed for each test round."
    )
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError(
        f"After removing the 50 adaptation images, only {len(remaining_abnormal_indices)} abnormal images remain, "
        f"but {N_TEST_ABNORMAL} are needed for each test round."
    )

source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #random image order each epoch
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) #testing validation images is not random, but it doesnt matter
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #target adaptation images random, ensures change in analysis



# Model
model = ResNet50DANN(
    use_pretrained=USE_PRETRAINED,
    freeze_backbone=FREEZE_BACKBONE
).to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss() #except loss here is evaluating if it can measure the source vs target

optimizer = torch.optim.Adam( #standard adams optimizer
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE #set at beginning
)

# DANN TRAINING - functin for a full epoch
def train_one_epoch_dann(model, source_loader, target_loader, optimizer,
                         class_criterion, domain_criterion, lambda_grl, domain_loss_weight=1.0):
    model.train() #enter training mode, allows batch normal layers to update.
#loss is running for total, class, and domain discriminator
    running_total_loss = 0.0
    running_class_loss = 0.0
    running_domain_loss = 0.0

    source_labels_all = []
    source_preds_all = []

    target_iter = cycle(target_loader) #for continuous cycle when the batch reaches final image for target images.

    for source_images, source_labels, _ in source_loader: #for loop, gives 16 (batch size) source images with labels
        target_images, _, _ = next(target_iter) #get 16 unlabeled target images
#gets all information needed to the GPU
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad() #clear gradients from last batch
#images in forward pass need to go through GRL so that it can be called back with Pytorch, but nothing is changes.
        # source images go through feature extract then give source logits, source images go through domain discrim and give dataset logits
        source_class_logits, source_domain_logits = model(source_images, lambda_grl=lambda_grl)

        # target forward
        _, target_domain_logits = model(target_images, lambda_grl=lambda_grl) #dont use target class logits

        # class loss only on source
        class_loss = class_criterion(source_class_logits, source_labels) #class_criterion is cross entropy function in pytorch

        # labels for domains, source = 0, target = 1
        source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=device) #dtype=torch.long ensures that cross entropy gets an integer
        target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long, device=device)

        domain_loss_source = domain_criterion(source_domain_logits, source_domain_labels) #loss for determining source images
        domain_loss_target = domain_criterion(target_domain_logits, target_domain_labels) #oss for determining target images
        domain_loss = 0.5 * (domain_loss_source + domain_loss_target) #total loss for domain discrim * 0.5 (I set) to avoid catastophic forgetting

        total_loss = class_loss + domain_loss_weight * domain_loss
        total_loss.backward() #send backward through GRL
        optimizer.step() #updates weights based on gradients (which were flipped in previous lined for DD)

        running_total_loss += total_loss.item() * source_images.size(0)
        running_class_loss += class_loss.item() * source_images.size(0)
        running_domain_loss += domain_loss.item() * source_images.size(0)

        #source_probs = torch.softmax(source_class_logits, dim=1)[:, 1]
        source_preds = torch.argmax(source_class_logits, dim=1) #logits to prediction based on softmas for class
#collect labels and predictions from batch and put into the list
        source_labels_all.extend(source_labels.detach().cpu().numpy())
        source_preds_all.extend(source_preds.detach().cpu().numpy())

    epoch_total_loss = running_total_loss / len(source_loader.dataset)
    epoch_class_loss = running_class_loss / len(source_loader.dataset)
    epoch_domain_loss = running_domain_loss / len(source_loader.dataset)
    epoch_acc = accuracy_score(source_labels_all, source_preds_all)

    return epoch_total_loss, epoch_class_loss, epoch_domain_loss, epoch_acc


def evaluate_source_classification(model, loader, criterion):
    model.eval() #switch to eval mode, cannot adjust batch norm layers as images pass through

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    with torch.no_grad(): #do not have back propagation during testing and validation
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            class_logits, _ = model(images, lambda_grl=0.0) #gets rid of domain logits, lambda is zero so during forward pass test images are not altered at all.
            loss = criterion(class_logits, labels) #cross entropy between class prediction and label that was obtained

            probs = torch.softmax(class_logits, dim=1)[:, 1]
            preds = torch.argmax(class_logits, dim=1)

            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths

# Now we train and call everything in
best_model_wts = copy.deepcopy(model.state_dict()) #gets copy of initial weights
best_val_loss = float("inf") #hold onto weights for best validation loss in the case of early stopping
epochs_without_improvement = 0

print("\nStarting DANN training...\n")

for epoch in range(NUM_EPOCHS):
    lambda_grl = get_lambda_grl(epoch, NUM_EPOCHS) #lambda is previously calculated for each epoch
#here is all the forward passe of batches, loss calulcations, backward passes, and weight updates
    train_total_loss, train_class_loss, train_domain_loss, train_acc = train_one_epoch_dann(
        model=model,
        source_loader=source_train_loader,
        target_loader=target_adapt_loader,
        optimizer=optimizer,
        class_criterion=class_criterion,
        domain_criterion=domain_criterion,
        lambda_grl=lambda_grl,
        domain_loss_weight=DOMAIN_LOSS_WEIGHT,
    )
#evaluation on 10% holdout set
    val_loss, val_acc, _, _, _, _ = evaluate_source_classification(
        model, source_val_loader, class_criterion
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Total: {train_total_loss:.4f} | "
        f"Train Class: {train_class_loss:.4f} | "
        f"Train Domain: {train_domain_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"GRL λ: {lambda_grl:.4f}"
    )
#early stopping function, patience of five. My argument for this is that
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_ResNet50_dann.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
        break

model.load_state_dict(best_model_wts) #use the best weights from training during validation

# get metrics for validation
val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_source_classification(
    model, source_val_loader, class_criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob) #compute metrics is our call for performance evaluation
print_metrics("SOURCE VALIDATION RESULTS (CLASSIFIER)", val_loss, val_metrics)

# Test on held out target images
def evaluate_target_test_subset(model, dataset, subset_indices, criterion):
    subset = Subset(dataset, subset_indices)
    subset_dataset = TransformedSubset(subset, transform=eval_transform)
    loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return evaluate_source_classification(model, loader, criterion)


repeat_results = []

print("\nStarting repeated target testing...\n")

for repeat_idx in range(N_REPEATS): #30 repeat testing with randomly selected sets
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

# mean and SD for target image performance
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
print("REPEATED TARGET TEST RESULTS (DANN WITH DenseNet201 BACKBONE)")
print("=" * 70)
#print mean and SD for the various images.
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

