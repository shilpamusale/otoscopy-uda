# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/PeripheralMasking_FINAL.py
# Role: Peripheral-masking evaluation: annotate images, then 30-repeat masked-vs-original testing.
#
# This is original manuscript code, kept VERBATIM. Do NOT refactor or bugfix it
# here — changing it would break reproducibility of the paper's results. The
# corrected, refactored, tested version of this logic lives in the product
# package under src/otoscopy_audit/audit/masking.py. See experiments/README.md.
#
# Settings are in the USER SETTINGS block below (hardcoded paths are expected
# here; the product package reads them from configs/ instead).
# ============================================================================

"""
PERIPHERAL MASKING EVALUATION
==============================
Tests whether UDA (DANN) reduces peripheral feature dependence compared
to a baseline model, following the masking approach of McIntosh et al. 2024
(npj Digital Medicine) extended to the domain adaptation setting.

Four conditions evaluated with 30-repeat testing:
    Baseline model  + original images
    Baseline model  + TM-masked images  (periphery replaced with grey noise)
    DANN model      + original images
    DANN model      + TM-masked images

Metrics: Accuracy, Sensitivity, Specificity, F1, AUROC (mean +/- SD)

If DANN reduces peripheral dependence, the performance drop from masking
should be smaller for the DANN model than the baseline model.

STEPS:
    Step 1 - Annotate all images (or resume from previous progress)
    Step 2 - Run 30-repeat masked vs original evaluation

USAGE:
    python peripheral_masking_evaluation.py --step 1   # annotate
    python peripheral_masking_evaluation.py --step 2   # evaluate
    python peripheral_masking_evaluation.py --step all # both
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_NAME = "OSU"   # change to "Chile" when running for Chile

#TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/"
TARGET_ROOT = "/Users/jordanvilla/Desktop/TM_Datasets/Datos_Chile_Combined/"

BASELINE_CHECKPOINT = "/Users/jordanvilla/PyCharmMiscProject/20260616_Chile_best_resnet50_baseline.pth"
DANN_CHECKPOINT     = "/Users/jordanvilla/PyCharmMiscProject/best_ResNet50_dann_Chile2.pth"

OUTPUT_DIR      = f"/Users/jordanvilla/Desktop/NewMask/Chile/"
ANNOTATION_FILE = f"/Users/jordanvilla/Desktop/Papers_in_Progress/Aim_3_UDAPaper/feature_analysis/MaskingEval/SavedMasks/tm_annotations_all_Chile.json"

N_TARGET_ADAPT  = 50
N_TEST_NORMAL   = 50
N_TEST_ABNORMAL = 50
N_REPEATS       = 30
RANDOM_SEED     = 42
IMAGE_SIZE      = 224
BATCH_SIZE      = 16

NOISE_MEAN = 0.5
NOISE_STD  = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def get_all_images(root_dir):
    root  = Path(root_dir)
    items = []
    for label, subdir in [(0, 'normal'), (1, 'abnormal')]:
        folder = root / subdir
        if not folder.exists():
            continue
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                items.append({
                    'path':      str(p),
                    'label':     label,
                    'label_str': 'Normal' if label == 0 else 'Abnormal'
                })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 - ANNOTATION TOOL
# ─────────────────────────────────────────────────────────────────────────────
def run_annotation():
    print("\n" + "="*60)
    print(f"STEP 1: ANNOTATE ALL {TARGET_NAME} IMAGES")
    print("="*60)

    all_images = get_all_images(TARGET_ROOT)
    print(f"  Total images found: {len(all_images)}")

    if Path(ANNOTATION_FILE).exists():
        with open(ANNOTATION_FILE) as f:
            annotations = json.load(f)
        print(f"  Already annotated: {len(annotations)}")
    else:
        annotations = {}

    todo = [img for img in all_images if img['path'] not in annotations]
    print(f"  Remaining:         {len(todo)}")

    if not todo:
        print("  All images annotated!")
        return

    print("\n  INSTRUCTIONS:")
    print("  Click around the tympanic membrane boundary to draw a polygon.")
    print("  ENTER=confirm  BACKSPACE=undo vertex  R=reset  S=skip  Q=quit+save")
    print()

    matplotlib.use('TkAgg')

    for img_idx, img_info in enumerate(todo):
        img_path  = img_info['path']
        label_str = img_info['label_str']
        print(f"  [{img_idx+1}/{len(todo)}] {label_str} | {Path(img_path).name}")

        try:
            pil_img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            img_np  = np.array(pil_img) / 255.0
        except Exception as e:
            print(f"    Cannot load: {e}")
            annotations[img_path] = {'label': img_info['label'],
                                      'label_str': label_str,
                                      'polygon': None, 'skipped': True}
            continue

        state = {'vertices': [], 'done': False, 'skip': False, 'quit': False}

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.canvas.manager.set_window_title(
            f'[{img_idx+1}/{len(todo)}] {TARGET_NAME} {label_str} {Path(img_path).name}')
        ax.imshow(img_np)
        ax.set_title(
            f'{TARGET_NAME} | {label_str} | {Path(img_path).name}\n'
            'Click TM boundary. ENTER=confirm  BACKSPACE=undo  R=reset  S=skip  Q=quit',
            fontsize=8)
        ax.axis('off')

        scatter    = ax.scatter([], [], c='#FFD700', s=35, zorder=5)
        line,      = ax.plot([], [], 'y-', lw=1.5, zorder=4)
        fill_patch = [None]

        def update_display():
            verts = state['vertices']
            if verts:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                scatter.set_offsets(np.c_[xs, ys])
                line.set_data(xs + [xs[0]] if len(verts) > 1 else (xs, ys),
                              ys + [ys[0]] if len(verts) > 1 else (xs, ys))
                if fill_patch[0] is not None:
                    fill_patch[0].remove()
                    fill_patch[0] = None
                if len(verts) >= 3:
                    fill_patch[0] = ax.add_patch(plt.Polygon(
                        verts, closed=True, facecolor='#FFD700',
                        alpha=0.2, edgecolor='#FFD700', linewidth=1.5))
            else:
                scatter.set_offsets(np.empty((0, 2)))
                line.set_data([], [])
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
            state['vertices'].append((event.xdata, event.ydata))
            update_display()

        def on_key(event):
            if event.key == 'enter':
                if len(state['vertices']) >= 3:
                    state['done'] = True
                    plt.close(fig)
                else:
                    print("    Need at least 3 vertices")
            elif event.key == 'backspace':
                if state['vertices']:
                    state['vertices'].pop()
                    if fill_patch[0] is not None:
                        fill_patch[0].remove()
                        fill_patch[0] = None
                    update_display()
            elif event.key == 'r':
                state['vertices'].clear()
                if fill_patch[0] is not None:
                    fill_patch[0].remove()
                    fill_patch[0] = None
                update_display()
            elif event.key == 's':
                state['skip'] = True
                plt.close(fig)
            elif event.key == 'q':
                state['quit'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()

        if state['quit']:
            with open(ANNOTATION_FILE, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"\n  Saved — {len(annotations)} annotated. Run again to continue.")
            return

        if state['skip']:
            annotations[img_path] = {'label': img_info['label'],
                                      'label_str': label_str,
                                      'polygon': None, 'skipped': True}
        elif state['done']:
            annotations[img_path] = {'label': img_info['label'],
                                      'label_str': label_str,
                                      'polygon': state['vertices'],
                                      'skipped': False}
            print(f"    Saved {len(state['vertices'])} vertices")

        with open(ANNOTATION_FILE, 'w') as f:
            json.dump(annotations, f, indent=2)

    print(f"\n  Done. {len(annotations)} images annotated.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1.5 - PRECOMPUTE AND SAVE ALL MASKED IMAGES
# Run once after annotation. Saves masked images to disk so step 2
# loads from disk instead of remasking on the fly every repeat.
# ─────────────────────────────────────────────────────────────────────────────
MASKED_DIR = Path(OUTPUT_DIR) / "masked_images"

def run_precompute_masks():
    print("\n" + "="*60)
    print(f"STEP 1.5: PRECOMPUTING MASKED IMAGES — {TARGET_NAME}")
    print("="*60)

    if not Path(ANNOTATION_FILE).exists():
        print("  ERROR: Run step 1 first.")
        return

    with open(ANNOTATION_FILE) as f:
        annotations = json.load(f)

    valid = {k: v for k, v in annotations.items()
             if not v.get('skipped', False) and v.get('polygon') is not None}
    print(f"  Valid annotations: {len(valid)}")

    MASKED_DIR.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    already_done = set(p.stem for p in MASKED_DIR.glob("*.jpg"))
    todo = {k: v for k, v in valid.items()
            if Path(k).stem not in already_done}
    print(f"  Already masked: {len(already_done)}")
    print(f"  Remaining:      {len(todo)}")

    if not todo:
        print("  All images already masked. Nothing to do.")
        return

    noise_rng = np.random.default_rng(RANDOM_SEED)

    for idx, (img_path, ann) in enumerate(todo.items()):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(todo)} ...")
        try:
            orig = Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE))
            tm_mask    = make_polygon_mask(ann["polygon"], IMAGE_SIZE)
            masked_img = apply_peripheral_mask(orig, tm_mask, noise_rng)
            save_path  = MASKED_DIR / f"{Path(img_path).stem}.jpg"
            masked_img.save(save_path, quality=95)
        except Exception as e:
            print(f"    WARNING: could not mask {Path(img_path).name}: {e}")

    total_saved = len(list(MASKED_DIR.glob("*.jpg")))
    size_mb     = sum(p.stat().st_size for p in MASKED_DIR.glob("*.jpg")) / 1e6
    print(f"\n  Done. {total_saved} masked images saved to {MASKED_DIR}")
    print(f"  Total size: {size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 - EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def make_polygon_mask(polygon_vertices, image_size):
    y_coords, x_coords = np.mgrid[:image_size, :image_size]
    points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
    path   = MplPath(polygon_vertices)
    return path.contains_points(points).reshape(image_size, image_size).astype(float)


def apply_peripheral_mask(pil_img, tm_mask, rng):
    img_np    = np.array(pil_img).astype(float) / 255.0
    noise     = rng.normal(NOISE_MEAN, NOISE_STD, img_np.shape).clip(0, 1)
    tm_mask_3 = np.stack([tm_mask] * 3, axis=-1)
    masked    = img_np * tm_mask_3 + noise * (1 - tm_mask_3)
    return Image.fromarray((masked.clip(0, 1) * 255).astype(np.uint8))


def run_evaluation():
    matplotlib.use('Agg')

    print("\n" + "="*60)
    print(f"STEP 2: PERIPHERAL MASKING EVALUATION - {TARGET_NAME}")
    print("="*60)

    if not Path(ANNOTATION_FILE).exists():
        print("  ERROR: Run step 1 first.")
        return

    with open(ANNOTATION_FILE) as f:
        annotations = json.load(f)

    # Build path -> polygon lookup
    polygon_lookup = {k: v['polygon'] for k, v in annotations.items()
                      if not v.get('skipped', False) and v.get('polygon') is not None}
    print(f"  Valid annotations: {len(polygon_lookup)}")

    # ── LOAD DATASET EXACTLY AS ORIGINAL DANN SCRIPT ──────────────────────
    # Uses BinaryFolderDataset + make_balanced_fixed_target_adapt_set
    # so the adaptation set exclusion is identical to training
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    from torchvision import transforms, models
    from torchvision.models import ResNet50_Weights
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import (accuracy_score, f1_score,
                                  roc_auc_score, confusion_matrix)

    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"  Device: {device}")

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # BinaryFolderDataset — copied exactly from original DANN script
    class BinaryFolderDataset_:
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

            print(f"  Loaded {len(self.samples)} valid images from {root_dir}")
            print(f"    Normal:   {sum(1 for _, y in self.samples if y == 0)}")
            print(f"    Abnormal: {sum(1 for _, y in self.samples if y == 1)}")

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

    # TransformedSubset that applies optional TM masking
    class MaskableSubset:
        def __init__(self, base_dataset, indices, masked, polygon_lookup, noise_rng):
            self.samples       = [base_dataset.samples[i] for i in indices]
            self.masked        = masked
            self.polygon_lookup = polygon_lookup
            self.noise_rng     = noise_rng

        def __len__(self):
            return len(self.samples)

        def get_item(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            if self.masked:
                if img_path in self.polygon_lookup:
                    # Load pre-saved masked image from disk
                    masked_path = MASKED_DIR / f"{Path(img_path).stem}.jpg"
                    if masked_path.exists():
                        img = Image.open(masked_path).convert("RGB").resize(
                            (IMAGE_SIZE, IMAGE_SIZE))
                    else:
                        # Fallback: mask on the fly
                        tm_mask = make_polygon_mask(
                            self.polygon_lookup[img_path], IMAGE_SIZE)
                        img = apply_peripheral_mask(img, tm_mask, self.noise_rng)
            return eval_tf(img), label, img_path

    # make_balanced_fixed_target_adapt_set — identical to original DANN script
    def make_balanced_fixed_target_adapt_set_(dataset, n_total, seed=42):
        normal_indices   = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
        abnormal_indices = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]
        rng = random.Random(seed)
        n_normal   = n_total // 2
        n_abnormal = n_total - n_normal
        sampled    = rng.sample(normal_indices, n_normal) +                      rng.sample(abnormal_indices, n_abnormal)
        rng.shuffle(sampled)
        return sampled

    # get_remaining_target_pool — identical to original DANN script
    def get_remaining_target_pool_(dataset, excluded_indices):
        excluded_set       = set(excluded_indices)
        remaining_normal   = []
        remaining_abnormal = []
        for idx, (_, label) in enumerate(dataset.samples):
            if idx in excluded_set:
                continue
            (remaining_normal if label == 0 else remaining_abnormal).append(idx)
        return remaining_normal, remaining_abnormal

    full_target_dataset = BinaryFolderDataset_(TARGET_ROOT)

    target_adapt_indices = make_balanced_fixed_target_adapt_set_(
        full_target_dataset, n_total=N_TARGET_ADAPT, seed=RANDOM_SEED)

    remaining_normal_indices, remaining_abnormal_indices = get_remaining_target_pool_(
        full_target_dataset, excluded_indices=target_adapt_indices)

    print(f"  Adaptation set: {len(target_adapt_indices)} images excluded")
    print(f"  Test pool - Normal: {len(remaining_normal_indices)} | "
          f"Abnormal: {len(remaining_abnormal_indices)}")

    if len(remaining_normal_indices) < N_TEST_NORMAL or        len(remaining_abnormal_indices) < N_TEST_ABNORMAL:
        raise ValueError("Not enough images for testing.")

    # ── SAVE EXAMPLE MASKED IMAGES ─────────────────────────────────────────
    print("\n  Saving masking examples ...")
    noise_rng = np.random.default_rng(RANDOM_SEED)

    for indices, label_str in [
        (remaining_normal_indices,   "Normal"),
        (remaining_abnormal_indices, "Abnormal"),
    ]:
        img_path, _ = full_target_dataset.samples[indices[0]]
        if img_path not in polygon_lookup:
            continue
        try:
            orig_img = Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE))
            orig_np  = np.array(orig_img) / 255.0
            polygon  = polygon_lookup[img_path]
            tm_mask  = make_polygon_mask(polygon, IMAGE_SIZE)
            mask_img = apply_peripheral_mask(orig_img, tm_mask, noise_rng)
            mask_np  = np.array(mask_img) / 255.0
            poly_arr   = np.array(polygon)
            poly_close = np.vstack([poly_arr, poly_arr[0]])

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.suptitle(
                f"Peripheral Masking Example - {TARGET_NAME} {label_str}\n"
                f"{Path(img_path).name}\n"
                "Only annotated TM region shown to model during masked evaluation",
                fontsize=11, fontweight="bold")

            axes[0].imshow(orig_np)
            axes[0].set_title("Original Image", fontweight="bold", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(orig_np)
            axes[1].plot(poly_close[:,0], poly_close[:,1], "y-", lw=2, zorder=3)
            axes[1].fill(poly_arr[:,0], poly_arr[:,1],
                         color="#FFD700", alpha=0.2, zorder=2)
            overlay = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4))
            overlay[tm_mask == 0] = [1, 0, 0, 0.28]
            axes[1].imshow(overlay)
            axes[1].set_title(
                "TM Annotation\nYellow=preserved  Red=masked",
                fontweight="bold", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(mask_np)
            axes[2].set_title("Masked Image\n(fed to model)",
                              fontweight="bold", fontsize=11)
            axes[2].axis("off")

            plt.tight_layout()
            sp = Path(OUTPUT_DIR) / f"masking_example_{TARGET_NAME}_{label_str}.png"
            plt.savefig(sp, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"    Saved {label_str} example -> {sp}")
        except Exception as e:
            print(f"    Could not save {label_str} example: {e}")

    # ── LOAD MODELS ─────────────────────────────────────────────────────────
    # TWO DIFFERENT ARCHITECTURES — do not swap these checkpoints

    # Baseline: plain ResNet50 with replaced FC layer
    # Trained on source only, no UDA
    # ← PUT PATH TO best_resnet50_baseline.pth IN BASELINE_CHECKPOINT AT TOP OF SCRIPT
    class ResNet50Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            backbone    = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, 2)  # two-class output: normal vs abnormal
            self.model  = backbone

        def forward(self, x):
            return self.model(x)  # returns single tensor of class logits

    # DANN: ResNet50 with gradient reversal layer and domain classifier
    # Trained with DANN adaptation on target images
    # ← PUT PATH TO best_ResNet50_dann_OSU.pth OR Chile IN DANN_CHECKPOINT AT TOP OF SCRIPT
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

    class ResNet50DANN(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim       = backbone.fc.in_features
            self.class_classifier  = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(self.feature_dim, 2))
            self.grl = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, 2))

        def forward(self, x, lambda_grl=1.0):
            features      = torch.flatten(self.feature_extractor(x), 1)
            class_logits  = self.class_classifier(features)
            domain_logits = self.domain_classifier(self.grl(features, lambda_grl))
            return class_logits, domain_logits  # returns tuple — only class_logits used at eval

    def load_baseline_model(path):
        # Loads plain ResNet50 — must match best_resnet50_baseline.pth architecture
        m = ResNet50Baseline().to(device)
        m.model.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        print(f"  Loaded baseline: {path}")
        return m

    def load_dann_model(path):
        # Loads ResNet50DANN — must match best_ResNet50_dann checkpoint architecture
        m = ResNet50DANN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        print(f"  Loaded DANN: {path}")
        return m

    print("\n  Loading models ...")
    baseline_model = load_baseline_model(BASELINE_CHECKPOINT)  # ← plain ResNet50 source-only
    dann_model     = load_dann_model(DANN_CHECKPOINT)          # ← ResNet50DANN adapted

    # ── METRICS ─────────────────────────────────────────────────────────────
    def compute_metrics(yt, yp, ypr):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        acc  = accuracy_score(yt, yp)
        sens = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0.0
        f1   = f1_score(yt, yp, zero_division=0)
        try:    auroc = roc_auc_score(yt, ypr)
        except: auroc = float('nan')
        return {'accuracy': acc, 'sensitivity': sens,
                'specificity': spec, 'f1': f1, 'auroc': auroc}

    # ── 30-REPEAT EVALUATION — identical loop to original DANN script ────────
    conditions = [
        ('Baseline', 'Original', baseline_model, False),
        ('Baseline', 'Masked',   baseline_model, True),
        ('DANN',     'Original', dann_model,     False),
        ('DANN',     'Masked',   dann_model,     True),
    ]

    all_results = []
    # noise_rng only used as fallback if precomputed masks missing
    noise_rng = np.random.default_rng(RANDOM_SEED)

    for model_name, img_type, model, masked in conditions:
        print(f"\n  {model_name} + {img_type} ...")
        repeat_results = []

        for rep in range(N_REPEATS):
            # Identical sampling to original DANN script
            rng = random.Random(RANDOM_SEED + rep)
            sampled_normal   = rng.sample(remaining_normal_indices,   N_TEST_NORMAL)
            sampled_abnormal = rng.sample(remaining_abnormal_indices, N_TEST_ABNORMAL)
            test_indices     = sampled_normal + sampled_abnormal
            rng.shuffle(test_indices)

            y_true, y_pred, y_prob = [], [], []
            batch_imgs, batch_labels = [], []

            for idx in test_indices:
                img_path, label = full_target_dataset.samples[idx]
                try:
                    img = Image.open(img_path).convert('RGB').resize(
                        (IMAGE_SIZE, IMAGE_SIZE))
                    if masked:
                        masked_path = MASKED_DIR / f"{Path(img_path).stem}.jpg"
                        if masked_path.exists():
                            img = Image.open(masked_path).convert('RGB').resize(
                                (IMAGE_SIZE, IMAGE_SIZE))
                        elif img_path in polygon_lookup:
                            tm_mask = make_polygon_mask(
                                polygon_lookup[img_path], IMAGE_SIZE)
                            img = apply_peripheral_mask(img, tm_mask, noise_rng)
                        else:
                            continue  # skip if no annotation
                except Exception:
                    continue

                batch_imgs.append(eval_tf(img))
                batch_labels.append(label)

                if len(batch_imgs) == BATCH_SIZE:
                    tensors = torch.stack(batch_imgs).to(device)
                    with torch.no_grad():
                        # Baseline returns single tensor, DANN returns (class_logits, domain_logits)
                        out    = model(tensors) if model_name == 'Baseline'                                  else model(tensors, lambda_grl=0.0)
                        logits = out[0] if isinstance(out, tuple) else out
                        probs  = torch.softmax(logits, dim=1)[:, 1]
                        preds  = torch.argmax(logits, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())
                    y_true.extend(batch_labels)
                    batch_imgs, batch_labels = [], []

            if batch_imgs:
                tensors = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    out    = model(tensors) if model_name == 'Baseline'                              else model(tensors, lambda_grl=0.0)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs  = torch.softmax(logits, dim=1)[:, 1]
                    preds  = torch.argmax(logits, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                y_true.extend(batch_labels)

            repeat_results.append(compute_metrics(y_true, y_pred, y_prob))

            if (rep+1) % 10 == 0:
                m = repeat_results[-1]
                print(f"    Rep {rep+1}/{N_REPEATS} "
                      f"F1={m['f1']:.4f} AUROC={m['auroc']:.4f}")

        summary = {'model': model_name, 'image_type': img_type,
                   'dataset': TARGET_NAME}
        for m in ['accuracy','sensitivity','specificity','f1','auroc']:
            vals = np.array([r[m] for r in repeat_results], dtype=float)
            summary[f'{m}_mean'] = round(np.nanmean(vals), 4)
            summary[f'{m}_sd']   = round(np.nanstd(vals, ddof=1), 4)
        all_results.append(summary)
        print(f"    F1: {summary['f1_mean']:.4f} +/- {summary['f1_sd']:.4f} | "
              f"AUROC: {summary['auroc_mean']:.4f} +/- {summary['auroc_sd']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_excel(
        Path(OUTPUT_DIR) / f'masking_eval_{TARGET_NAME}.xlsx', index=False)

    # Performance drops
    print("\n" + "="*60)
    print("PERFORMANCE DROP (Original -> Masked)")
    print("="*60)
    drop_rows = []
    for model_name in ['Baseline', 'DANN']:
        orig = results_df[(results_df['model'] == model_name) &
                          (results_df['image_type'] == 'Original')].iloc[0]
        mask = results_df[(results_df['model'] == model_name) &
                          (results_df['image_type'] == 'Masked')].iloc[0]
        row = {'model': model_name, 'dataset': TARGET_NAME}
        for m in ['accuracy','sensitivity','specificity','f1','auroc']:
            drop = orig[f'{m}_mean'] - mask[f'{m}_mean']
            row[f'{m}_original'] = orig[f'{m}_mean']
            row[f'{m}_masked']   = mask[f'{m}_mean']
            row[f'{m}_drop']     = round(drop, 4)
            print(f"  {model_name} {m}: {orig[f'{m}_mean']:.4f} -> "
                  f"{mask[f'{m}_mean']:.4f}  drop={drop:+.4f}")
        drop_rows.append(row)
        print()

    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_excel(
        Path(OUTPUT_DIR) / f'masking_drops_{TARGET_NAME}.xlsx', index=False)

    # Figures
    plt.rcParams.update({'font.family': 'DejaVu Sans',
                         'axes.spines.top': False,
                         'axes.spines.right': False})

    metrics_to_plot = ['accuracy','sensitivity','specificity','f1','auroc']
    metric_labels   = ['Accuracy','Sensitivity','Specificity','F1','AUROC']

    # Figure 1: all metrics bar chart
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle(
        f'Peripheral Masking Evaluation - {TARGET_NAME}\n'
        'Performance on original vs TM-masked images (mean +/- SD, 30 repeats)\n'
        'Following McIntosh et al. 2024, extended to UDA setting',
        fontsize=12, fontweight='bold')

    for col, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[col]
        for i, (model_name, color) in enumerate(
                [('Baseline','#888888'), ('DANN','#2166ac')]):
            for j, (img_type, alpha, hatch) in enumerate(
                    [('Original', 0.85, ''), ('Masked', 0.45, '///')]):
                row  = results_df[(results_df['model'] == model_name) &
                                   (results_df['image_type'] == img_type)].iloc[0]
                mean = row[f'{metric}_mean']
                sd   = row[f'{metric}_sd']
                pos  = i * 2.2 + j * 0.9
                ax.bar(pos, mean, width=0.8, color=color,
                       alpha=alpha, hatch=hatch, zorder=2)
                ax.errorbar(pos, mean, yerr=sd, fmt='none', color='black',
                            capsize=4, linewidth=1.5, zorder=3)
                ax.text(pos, mean + sd + 0.01, f'{mean:.3f}',
                        ha='center', fontsize=7, fontweight='bold')

        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_xticks([0.45, 2.65])
        ax.set_xticklabels(['Baseline', 'DANN'], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.2, axis='y')
        if col == 0:
            ax.set_ylabel('Score', fontsize=10)

    handles = [
        plt.Rectangle((0,0),1,1, color='#888888', alpha=0.85,
                       label='Baseline Original'),
        plt.Rectangle((0,0),1,1, color='#888888', alpha=0.45,
                       hatch='///', label='Baseline Masked'),
        plt.Rectangle((0,0),1,1, color='#2166ac', alpha=0.85,
                       label='DANN Original'),
        plt.Rectangle((0,0),1,1, color='#2166ac', alpha=0.45,
                       hatch='///', label='DANN Masked'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(Path(OUTPUT_DIR) / f'masking_eval_barplot_{TARGET_NAME}.png',
                dpi=180, bbox_inches='tight')
    plt.close()

    # Figure 2: drop comparison
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.suptitle(
        f'Performance Drop from Peripheral Masking - {TARGET_NAME}\n'
        'Did DANN reduce peripheral feature dependence?\n'
        '(smaller bar = less peripheral reliance)',
        fontsize=12, fontweight='bold')

    for col, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax     = axes[col]
        b_drop = drop_df[drop_df['model']=='Baseline'][f'{metric}_drop'].values[0]
        d_drop = drop_df[drop_df['model']=='DANN'][f'{metric}_drop'].values[0]
        bars   = ax.bar([0,1], [b_drop, d_drop],
                        color=['#888888','#2166ac'], alpha=0.85,
                        width=0.6, zorder=2)
        ax.axhline(0, color='black', lw=0.8)
        for bar, val in zip(bars, [b_drop, d_drop]):
            ypos = val + 0.005 if val >= 0 else val - 0.015
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:+.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Baseline','DANN'], fontsize=9)
        if col == 0:
            ax.set_ylabel('Performance drop\n(Original - Masked)', fontsize=9)
        ax.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / f'masking_drop_comparison_{TARGET_NAME}.png',
                dpi=180, bbox_inches='tight')
    plt.close()

    print(f"\n  All outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['1','1.5','2','all'], default='all')
    args = parser.parse_args()
    if args.step in ('1', 'all'):
        run_annotation()
    if args.step in ('1.5', 'all'):
        run_precompute_masks()
    if args.step in ('2', 'all'):
        run_evaluation()
