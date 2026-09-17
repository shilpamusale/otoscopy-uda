# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/MCC_Resnet50.py
# Role: Train the MCC (minimum class confusion) adaptation model.
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
from PIL import Image # needed this to import images

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models #transforms for the standard augmentation, models for pretrained CNN Resnetfor us
from torchvision.models import ResNet50_Weights #gets pretrained RN50

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
#adopted from https://github.com/thuml/Versatile-Domain-Adaptation - This is the skin paper i have shown JC regarding UDA
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

# MCC setup
N_TARGET_ADAPT = 75          # fixed unlabeled target images
N_REPEATS = 30
N_TEST_NORMAL = 50
N_TEST_ABNORMAL = 50
MCC_LOSS_WEIGHT = 0.5

USE_PRETRAINED = True
FREEZE_BACKBONE = False


# Seed set here for reproducible results.
def set_seed(seed=42):
    random.seed(seed) #random = pythons random number generator
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# This section is regarding computer/device running this script, this should really only affect the speed on the run time
if torch.backends.mps.is_available(): #This is true for me because I have a Mac
    device = torch.device("mps") #mps is metal performance shaders, which is on apple products
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# DATASET
class BinaryFolderDataset(Dataset):
    """
    Expected structure:
    root/
        normal/
        abnormal/
            optional nested pathology subfolders

    normal   -> 0
    abnormal -> 1
    """
    # I changed all images to jpeg on my computer, make sure your file type is here if running this
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#verify put in to make sure that images are valid, make sure numbers printed from images match all those in your folder, otherwise some images were corrupted
    def __init__(self, root_dir, transform=None, verify_images=True):
        self.root_dir = Path(root_dir)
        self.transform = transform #set this to none above
        self.samples = []
        self.bad_files = [] #in case any files cannot open later, to see which ones are broken

        normal_dir = self.root_dir / "normal"
        abnormal_dir = self.root_dir / "abnormal"
#These only verify images upload, if not will return missing folder. I had initial problems getting the folders from my desktop to mps so these are in
        if not normal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {normal_dir}")
        if not abnormal_dir.exists():
            raise FileNotFoundError(f"Missing folder: {abnormal_dir}")
#Labels training images as normal0 or abnormal1
        self.samples.extend(self._gather_images(normal_dir, label=0, verify_images=verify_images))
        self.samples.extend(self._gather_images(abnormal_dir, label=1, verify_images=verify_images))
#If the folder is there but empty, this will give the error
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {root_dir}")

        print(f"Loaded {len(self.samples)} valid images from {root_dir}")
        print(f"  Normal:   {sum(1 for _, y in self.samples if y == 0)}")
        print(f"  Abnormal: {sum(1 for _, y in self.samples if y == 1)}")



    def _is_valid_image(self, path):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB") #needed for ResNet and MobileNet in our project
                img.load() #loads the converted RGB image from above into the project pipeline to be used
            return True
        except Exception:
            return False

    def _gather_images(self, folder, label, verify_images=True):
        items = []
        for path in folder.rglob("*"): #The rglob function is needed here because abnormal folders have several pathologies so subfolders needed to be evaluated
            if path.is_file() and path.suffix.lower() in self.IMG_EXTENSIONS:
                if verify_images: #which is set to true in line 118 above, we can set verify_images to false in order to speed things up now too
                    if self._is_valid_image(path):
                        items.append((str(path), label)) #the file must be there and have .jpeg etc
                    else:
                        self.bad_files.append(str(path))
                else:
                    items.append((str(path), label))
        return items #items added to self.samples above.
    #self.samples now has every image that is verified, to be used for training, testing etc.
    def __len__(self):
        return len(self.samples) #simply gives length/number of images in verified images

    def __getitem__(self, idx):
        img_path, label = self.samples[idx] #gives path and number (0-880) etc for the images in the datasets
        image = Image.open(img_path).convert("RGB") #again convertes to RGB
        if self.transform: #set to none in beginning
            image = self.transform(image)
        return image, label, img_path #sends image, label and path to __getitem__

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



# Standard data augmentation in training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #These are the mean and SD for pixel values for RGB intensities in ImageNet, which was used to trained Resnet50
                         std=[0.229, 0.224, 0.225]), #i.e. it normalizes our images to what the model was trained on.
])
#validation images also get resized and normalized to RGB intensities of ImageNet
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# MCC LOSS
def minimum_class_confusion_loss(logits, temperature=2.0):
    """
    MCC-style loss on unlabeled target predictions.

    Steps:
    1. soften logits with temperature
    2. compute class probabilities
    3. entropy-weight samples (high-confidence samples count more)
    4. build class confusion matrix
    5. minimize off-diagonal confusion
    """
    probs = F.softmax(logits / temperature, dim=1)  # so from here on out probs = logits converted to probabilites with temp scaling

    # entropy-based weights
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # need 1e-8 so you never take log zero
    weights = 1.0 + torch.exp(-entropy)
    weights = weights / weights.sum().detach()

    weighted_probs = probs * weights.unsqueeze(1)  # probs adjusted for their respective weight based on entropy

    # class correlation/confusion matrix
    confusion = torch.matmul(weighted_probs.t(), probs)  # [C, C]

    # normalize rows
    confusion = confusion / (confusion.sum(dim=1, keepdim=True) + 1e-8)

    # minimize off-diagonal entries
    off_diag_sum = confusion.sum() - torch.trace(confusion)
    num_classes = logits.size(1)
    loss = off_diag_sum / num_classes
    return loss


# MODEL
def build_resnet50_classifier(use_pretrained=True, freeze_backbone=False):
    if use_pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT) #loads in weights of best ResNet50 model
    else:
        model = models.resnet50(weights=None) #this would be full on fine tuning, we do not want to start from scratch

    if freeze_backbone:#this is set to false, but if true the weights cannot be updated with our images/datasets
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2) #this replaces the final classification layer in RN50 with 0 and 1 for normal vs abnormal
#i.e. we are using transfer learning
    if freeze_backbone: #would unfreeze final classification layer but freeze_backbone is false for our experiments
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


# METRICS
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() #y_true = ground truth, y_pred = models prediction, y_prob = models predicted probability

    accuracy = accuracy_score(y_true, y_pred) #number true / total number of predictions = accuracy
    precision = precision_score(y_true, y_pred, zero_division=0)#calling sklearn for precision,recall and F1
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 #of those abnormal, how many are caught?
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0# of the normal images, how many were labelled normal?

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

# HELPERS - n_total is defined later as N_target_adapt, which is input in first section I believe
#this simply grabs position of target images in the dataset
def make_balanced_fixed_target_adapt_set(dataset, n_total, seed=42):
    normal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    rng = random.Random(seed)
#define how many target images per class - I initially set this to be equal, but we could look at using only normal, etc
    #However, only having normal would not allow for abnormal to be within the confusion matrix which is needed to drive the MCC model
    n_normal = n_total // 2
    n_abnormal = n_total - n_normal
#make sure we have the numbers needed in the databases for UDA
    if len(normal_indices) < n_normal or len(abnormal_indices) < n_abnormal:
        raise ValueError(
            f"Not enough target images to build fixed adaptation set of {n_total}."
        )

    sampled_normal = rng.sample(normal_indices, n_normal) #grabs n_total defined number of normal images from dataset
    sampled_abnormal = rng.sample(abnormal_indices, n_abnormal) #same but for abnormal

    adapt_indices = sampled_normal + sampled_abnormal #adapt_indices is now list with all unlabelled data
    rng.shuffle(adapt_indices)
    return adapt_indices

#Now this helper fctn will get the remaining pool, which can be used for external testing.
def get_remaining_target_pool(dataset, excluded_indices):
    excluded_set = set(excluded_indices)
#these also ensure no leakage, as there is no cross talk between adaptation set and target set
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


# To start training we need to first actually load in the datasets
full_source_dataset = BinaryFolderDataset(TRAIN_ROOT, transform=None, verify_images=True)
full_target_dataset = BinaryFolderDataset(TARGET_ROOT, transform=None, verify_images=True)

# Source split: we used 90/10 but this can change based on what we put in beginning
#int function ensures whole number of images in both sets
source_train_size = int((1 - VAL_SPLIT) * len(full_source_dataset))
source_val_size = len(full_source_dataset) - source_train_size
#gets positions of train and validation sets
generator = torch.Generator().manual_seed(RANDOM_SEED)
source_train_subset, source_val_subset = random_split(
    full_source_dataset,
    [source_train_size, source_val_size],
    generator=generator
)

source_train_dataset = TransformedSubset(source_train_subset, transform=train_transform)
source_val_dataset = TransformedSubset(source_val_subset, transform=eval_transform)

# Calls the unlabbeled images
target_adapt_indices = make_balanced_fixed_target_adapt_set(
    full_target_dataset,
    n_total=N_TARGET_ADAPT,
    seed=RANDOM_SEED
)
target_adapt_subset = Subset(full_target_dataset, target_adapt_indices)
target_adapt_dataset = TransformedSubset(target_adapt_subset, transform=train_transform) #here the target images are transformed like the training set

# Remaining target pool called for repeated testing
remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool(
    full_target_dataset,
    excluded_indices=target_adapt_indices
)
#prints numbers in each set, important, if images are corrupted and removed you should notice numebrs are off here
print("\nFixed target adaptation set:")
print(f"  Total: {len(target_adapt_indices)}")
print(f"  Remaining normal for testing:   {len(remaining_normal_indices)}")
print(f"  Remaining abnormal for testing: {len(remaining_abnormal_indices)}")
#this if statement is safety to notify you if you do not have enough testing images after removing the adaptation images
if len(remaining_normal_indices) < N_TEST_NORMAL:
    raise ValueError(
        f"After removing target adaptation images, only {len(remaining_normal_indices)} normal images remain."
    )
if len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
    raise ValueError(
        f"After removing target adaptation images, only {len(remaining_abnormal_indices)} abnormal images remain."
    )
#three data loaders needed during training because we have training, adaptation set, and validation set
source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #num workers here is images uploaded at a time, so zero means one image at a time, which is what Mac can handle in pytorch
source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) #shuffle false here because random order in testing does not really matter, they are all seen
target_adapt_loader = DataLoader(target_adapt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# MODEL / LOSS / OPTIMIZER
model = build_resnet50_classifier(
    use_pretrained=USE_PRETRAINED,
    freeze_backbone=FREEZE_BACKBONE
).to(device)

class_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE #defined earlier as 10e-4
)
#REALLY REALLY IMPORTANT, WE NEED TO INCLUDE VALIDATION RESULTS BEFORE MCC BECUASE WITHOUT IT WE CAN NOT SEE HOW THE UNLABELLED TARGET
#IMAGES ARE AFFECTING THE MODEL TO EVALUATE THE SOURCE. IN OTHER WORDS, WE DO NOT WANT IT TO PERFORM WELL ON JUST EXTERNAL, NEEDS TO MAINTAIN ITS INITIAL ABILITIES
# TRAINing
def train_one_epoch_mcc(model, source_loader, target_loader, optimizer,
                        class_criterion, mcc_loss_weight=0.5): #cross entropy with adams optimizer
    model.train() #enter training - needed because if in testing/eval the batch norm layers are not updated

    running_total_loss = 0.0 #combined loss
    running_class_loss = 0.0 #loss of cross entropy
    running_mcc_loss = 0.0 #loss for MCC - calculated by JV in paper if want to see
#tracking all three lets us see problem. If total loss if high but class loss low, MCC adaptation is the issue
    source_labels_all = []
    source_preds_all = []

    target_iter = iter(target_loader) #switch to iterator instead of dataloader

    for source_images, source_labels, _ in source_loader: #the _ here is the file path that was discarded, not needed
        try:
            target_images, _, _ = next(target_iter) #_, _ here is the label (not wanted int UDA) and the path.
        except StopIteration: #this allows everything to go on even after the number of unlabelled images runs out, resets it to the initial batch
            target_iter = iter(target_loader) #the try function here ensures you get training images when you run the source images through training too
            target_images, _, _ = next(target_iter) #gets new target images when they run out, as sometimes we are only using a few
#since there are so many more images in training, the target images may be seen more than once per epoch
        source_images = source_images.to(device) #brings images to the same place as model RN50 - mps for mac
        source_labels = source_labels.to(device)
        target_images = target_images.to(device) #notice no target labels sent

        optimizer.zero_grad() #clear gradient from previous batch
#forward pass through models with both source and target images - output is image logits for each image in 16 image batch
        #these are separate as they have different loss functions in the total loss equation.
        source_logits = model(source_images)
        target_logits = model(target_images)

        class_loss = class_criterion(source_logits, source_labels) #cross entropy loss calculated for source images
        mcc_loss = minimum_class_confusion_loss(target_logits) #MCC loss with

        total_loss = class_loss + mcc_loss_weight * mcc_loss #total loss used in backprop
        total_loss.backward()  #uses total loss to adjust gradients in backprop
        optimizer.step() #uses gradient from previous line to reduce loss in following batches by adjusting batches - basis for gradient descent
#add loss values to running totals
        running_total_loss += total_loss.item() * source_images.size(0)
        running_class_loss += class_loss.item() * source_images.size(0)
        running_mcc_loss += mcc_loss.item() * source_images.size(0)

        source_preds = torch.argmax(source_logits, dim=1) #this just assigns a prediction, 0,1 based on the logits that were output by the source.
        source_labels_all.extend(source_labels.detach().cpu().numpy()) #makes/adds to list the labels of source images
        source_preds_all.extend(source_preds.detach().cpu().numpy()) #makes/adds to a list of predictions for images in source folder (on training before any validation occurs)

    epoch_total_loss = running_total_loss / len(source_loader.dataset)
    epoch_class_loss = running_class_loss / len(source_loader.dataset)
    epoch_mcc_loss = running_mcc_loss / len(source_loader.dataset)
    epoch_acc = accuracy_score(source_labels_all, source_preds_all)

    return epoch_total_loss, epoch_class_loss, epoch_mcc_loss, epoch_acc


def evaluate_classifier(model, loader, criterion):
    model.eval() #eval mode so now weights and batch norm layers are frozen

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images) #get a logit output per validation image
            loss = criterion(logits, labels) #cross entropy loss

            probs = torch.softmax(logits, dim=1)[:, 1] #calculate softmax based on logits
            preds = torch.argmax(logits, dim=1) #assign 0 or 1 based on softmax

            running_loss += loss.item() * images.size(0) #
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths

# EARLY STOPPING
best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting MCC training...\n")

for epoch in range(NUM_EPOCHS):
    train_total_loss, train_class_loss, train_mcc_loss, train_acc = train_one_epoch_mcc(
        model=model,
        source_loader=source_train_loader,
        target_loader=target_adapt_loader,
        optimizer=optimizer,
        class_criterion=class_criterion,
        mcc_loss_weight=MCC_LOSS_WEIGHT
    )

    val_loss, val_acc, _, _, _, _ = evaluate_classifier(
        model, source_val_loader, class_criterion
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Total: {train_total_loss:.4f} | "
        f"Train Class: {train_class_loss:.4f} | "
        f"Train MCC: {train_mcc_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )
#here is the early stopping function. Although maybe unconventional in UDA, my argument is that we still want performance on source dataset to be as high as target
    #This is often forgot about, but known as catastrophic forgetting, in which the model completely shifts to the new dataset
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_resnet50_mcc.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
        break

model.load_state_dict(best_model_wts) #takes best validation weights in case that early stopping is triggered


# SOURCE VALIDATION RESULTS
val_loss, val_acc, y_val, y_val_pred, y_val_prob, _ = evaluate_classifier(
    model, source_val_loader, class_criterion
)
val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
print_metrics("SOURCE VALIDATION RESULTS", val_loss, val_metrics)


# REPEATED TARGET TESTING
def evaluate_target_test_subset(model, dataset, subset_indices, criterion):
    subset = Subset(dataset, subset_indices)
    subset_dataset = TransformedSubset(subset, transform=eval_transform) #no augmentation
    loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return evaluate_classifier(model, loader, criterion)

repeat_results = []

print("\nStarting repeated target testing...\n")

for repeat_idx in range(N_REPEATS):
    rng = random.Random(RANDOM_SEED + repeat_idx) #different test sets every time

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


# SUMMARY MEAN and SD

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
print("REPEATED TARGET TEST RESULTS (MCC WITH RESNET50 BACKBONE)")
print("=" * 70)

for metric in metric_names:
    values = np.array([r[metric] for r in repeat_results], dtype=float) #integers
    mean_val = np.nanmean(values) #get average
    sd_val = np.nanstd(values, ddof=1) #SD
    print(f"{metric.capitalize():<12}: {mean_val:.4f} ± {sd_val:.4f}") #capitalize and orient the print, not needed.
#here down we dont really need,
#for cm_metric in ["tn", "fp", "fn", "tp"]:
    #values = np.array([r[cm_metric] for r in repeat_results], dtype=float)
    #mean_val = np.mean(values)
    #sd_val = np.std(values, ddof=1)
    #print(f"{cm_metric.upper():<12}: {mean_val:.2f} ± {sd_val:.2f}")

# OPTIONAL: SHOW FIXED TARGET ADAPT SET COMPOSITION
#adapt_labels = [full_target_dataset.samples[i][1] for i in target_adapt_indices]
#print("\nFixed target adaptation set composition:")
#print(f"  Normal used for MCC adaptation:   {sum(1 for y in adapt_labels if y == 0)}")
#print(f"  Abnormal used for MCC adaptation: {sum(1 for y in adapt_labels if y == 1)}")