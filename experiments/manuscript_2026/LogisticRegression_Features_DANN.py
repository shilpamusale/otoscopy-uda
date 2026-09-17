# ============================================================================
# FROZEN RESEARCH SCRIPT — part of the manuscript reproducibility record.
#
# File: experiments/manuscript_2026/LogisticRegression_Features_DANN.py
# Role: Logistic-regression probes on features extracted from the adapted models.
#
# This is original manuscript code, kept VERBATIM. Do NOT refactor or bugfix it
# here — changing it would break reproducibility of the paper's results. The
# corrected, refactored, tested version of this logic lives in the product
# package under src/otoscopy_audit/. See experiments/README.md for rationale.
#
# Settings are in the USER SETTINGS block below (hardcoded paths are expected
# here; the product package reads them from configs/ instead).
# ============================================================================

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
OSU_FEATURES_CSV   = "/Users/jordanvilla/Desktop/Feature_Output/featureanalysis_mcc75_weightpoint5/OSU/OSU_features_after_MCC.csv"
CHILE_FEATURES_CSV = "/Users/jordanvilla/Desktop/Feature_Output/featureanalysis_mcc75_weightpoint5/Chile/Chile_features_after_MCC.csv"

OSU_CHECKPOINT   = "/Users/jordanvilla/Desktop/Feature_Output/featureanalysis_mcc75_weightpoint5/OSU/best_ResNet50_mcc_OSU.pth"
CHILE_CHECKPOINT = "/Users/jordanvilla/Desktop/Feature_Output/featureanalysis_mcc75_weightpoint5/Chile/best_ResNet50_mcc_Chile.pth"

OSU_NORMAL_DIR   = "/Users/jordanvilla/Desktop/TM_Datasets/OSUdataset/Normal/"
CHILE_NORMAL_DIR = "/Users/jordanvilla/Desktop/TM_Datasets/Datos_Chile_Combined/Normal/"

OUTPUT_DIR = "/Users/jordanvilla/Desktop/Feature_Output/featureanalysis_mcc75_weightpoint5/Regression/"

# Specific images to use for Grad-CAM (filename without extension)
OSU_IMAGE_NAMES = [
    "AM10L", "AM7L", "AM26R", "AM40R", "AM73R",
    "AM83R", "AM97R", "AM117R", "AM117L", "AM213R"
]
CHILE_IMAGE_NAMES = [
    "n1", "n17", "n27", "n38", "n45",
    "n58", "n61", "n67", "n75", "n90"
]

N_TOP_FEATURES = 10
IMAGE_SIZE     = 224

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
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

class ResNet50DANN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim       = backbone.fc.in_features
        self.class_classifier  = nn.Sequential(
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

def load_model(checkpoint_path):
    model = ResNet50DANN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FEATURE CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading feature CSVs ...")
osu_df   = pd.read_csv(OSU_FEATURES_CSV)
chile_df = pd.read_csv(CHILE_FEATURES_CSV)

feat_cols = [c for c in osu_df.columns if c.startswith('feat_')]

osu_feats   = osu_df[feat_cols].values
osu_labels  = osu_df['label'].values

chile_feats  = chile_df[feat_cols].values
chile_labels = chile_df['label'].values

print(f"  OSU features:   {osu_feats.shape}")
print(f"  Chile features: {chile_feats.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: TRAIN PROBE CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining probe classifiers ...")

scaler_osu   = StandardScaler()
scaler_chile = StandardScaler()

X_osu   = scaler_osu.fit_transform(osu_feats)
X_chile = scaler_chile.fit_transform(chile_feats)

probe_osu = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
probe_osu.fit(X_osu, osu_labels)

probe_chile = LogisticRegression(max_iter=2000, random_state=42, C=0.1)
probe_chile.fit(X_chile, chile_labels)

osu_weights   = np.abs(probe_osu.coef_[0])
chile_weights = np.abs(probe_chile.coef_[0])

print(f"  OSU probe   — top feature weight: {osu_weights.max():.4f}")
print(f"  Chile probe — top feature weight: {chile_weights.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FIND FEATURES IMPORTANT FOR OSU BUT NOT CHILE
# ─────────────────────────────────────────────────────────────────────────────
print("\nIdentifying discriminative features ...")

osu_weights_norm   = osu_weights   / (osu_weights.max()   + 1e-8)
chile_weights_norm = chile_weights / (chile_weights.max() + 1e-8)
weight_diff        = osu_weights_norm - chile_weights_norm
feature_indices    = np.arange(len(weight_diff))

osu_important_idx   = feature_indices[np.argsort(weight_diff)[::-1]][:N_TOP_FEATURES]
chile_important_idx = feature_indices[np.argsort(weight_diff)][:N_TOP_FEATURES]

weight_df = pd.DataFrame({
    'feature_index':      feature_indices,
    'osu_probe_weight':   osu_weights,
    'chile_probe_weight': chile_weights,
    'osu_weight_norm':    osu_weights_norm,
    'chile_weight_norm':  chile_weights_norm,
    'weight_diff':        weight_diff,
}).sort_values('weight_diff', ascending=False)

out_path = Path(OUTPUT_DIR) / 'probe_weight_comparison.xlsx'
weight_df.to_excel(out_path, index=False)
print(f"  Saved weight comparison → {out_path}")
print(f"\nTop {N_TOP_FEATURES} OSU-important features: {list(osu_important_idx)}")
print(f"Top {N_TOP_FEATURES} Chile-important features: {list(chile_important_idx)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PROBE WEIGHT BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Probe Classifier Feature Weights\nDiagnostically Important Features per Dataset',
             fontsize=13, fontweight='bold')

for ax, df_sub, title in [
    (axes[0], weight_df.head(N_TOP_FEATURES),
     f'Top {N_TOP_FEATURES} Features: Important for OSU, Not Chile'),
    (axes[1], weight_df.tail(N_TOP_FEATURES).sort_values('weight_diff'),
     f'Top {N_TOP_FEATURES} Features: Important for Chile, Not OSU'),
]:
    x = np.arange(len(df_sub))
    ax.bar(x - 0.2, df_sub['osu_weight_norm'],   width=0.4,
           color='#2166ac', alpha=0.85, label='OSU probe weight')
    ax.bar(x + 0.2, df_sub['chile_weight_norm'], width=0.4,
           color='#d6604d', alpha=0.85, label='Chile probe weight')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sub['feature_index'].astype(int),
                       rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel('Normalized Probe Weight')
    ax.set_xlabel('Feature Index')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / 'probe_weight_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved probe weight plot")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING — BY SPECIFIC FILENAME
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def get_named_image_paths(folder, names):
    """
    Find specific images by filename stem (no extension).
    Searches recursively. Returns paths in the order given,
    printing a warning for any name not found.
    """
    folder = Path(folder)
    stem_to_path = {}
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            stem_to_path[p.stem] = p
            stem_to_path[p.stem.lower()] = p  # also index lowercase

    paths = []
    for name in names:
        if name in stem_to_path:
            paths.append(stem_to_path[name])
        elif name.lower() in stem_to_path:
            paths.append(stem_to_path[name.lower()])
        else:
            print(f"  WARNING: could not find '{name}' in {folder}")
    return paths

def load_image_tensor(path):
    return eval_transform(Image.open(path).convert('RGB')).unsqueeze(0)

def load_image_array(path):
    return np.array(Image.open(path).convert('RGB').resize(
        (IMAGE_SIZE, IMAGE_SIZE))) / 255.0

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
class FeatureGradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = list(model.feature_extractor.children())[-2][-1]
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, input, output):
        self.activations = output.detach()

    def _bwd(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, feature_idx):
        self.model.eval()
        inp = image_tensor.to(device)
        inp.requires_grad_(True)
        features = self.model.extract_features(inp)
        self.model.zero_grad()
        features[0, feature_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = torch.nn.functional.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def generate_gradcam_figure(osu_model, chile_model,
                             osu_image_paths, chile_image_paths,
                             feature_idx, feature_rank,
                             osu_weight, chile_weight,
                             list_label, save_name):
    osu_cam_gen   = FeatureGradCAM(osu_model)
    chile_cam_gen = FeatureGradCAM(chile_model)

    n = min(len(osu_image_paths), len(chile_image_paths))

    fig, axes = plt.subplots(n, 4, figsize=(14, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Feature {feature_idx} — {list_label}\n'
        f'OSU probe weight: {osu_weight:.4f}  |  Chile probe weight: {chile_weight:.4f}\n'
        f'Grad-CAM on Normal Images: OSU (left) vs Chile (right)',
        fontsize=12, fontweight='bold'
    )

    col_titles = ['OSU — Original', 'OSU — Grad-CAM',
                  'Chile — Original', 'Chile — Grad-CAM']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(
            title, fontsize=10, fontweight='bold',
            color='#2166ac' if 'OSU' in title else '#d6604d')

    for row in range(n):
        osu_raw   = load_image_array(osu_image_paths[row])
        chile_raw = load_image_array(chile_image_paths[row])

        osu_cam   = osu_cam_gen.generate(
            load_image_tensor(osu_image_paths[row]),   feature_idx)
        chile_cam = chile_cam_gen.generate(
            load_image_tensor(chile_image_paths[row]), feature_idx)

        osu_overlay   = np.clip(
            0.5 * osu_raw   + 0.5 * plt.cm.jet(osu_cam)[:, :, :3],   0, 1)
        chile_overlay = np.clip(
            0.5 * chile_raw + 0.5 * plt.cm.jet(chile_cam)[:, :, :3], 0, 1)

        # Image name labels on left
        osu_name   = Path(osu_image_paths[row]).stem
        chile_name = Path(chile_image_paths[row]).stem
        axes[row, 0].set_ylabel(f'{osu_name}', fontsize=8,
                                rotation=0, labelpad=45, va='center')

        axes[row, 0].imshow(osu_raw);       axes[row, 0].axis('off')
        axes[row, 1].imshow(osu_overlay);   axes[row, 1].axis('off')
        axes[row, 2].imshow(chile_raw);     axes[row, 2].axis('off')
        axes[row, 3].imshow(chile_overlay); axes[row, 3].axis('off')

        # Chile image name on right
        axes[row, 3].set_xlabel(chile_name, fontsize=8)

    line = plt.Line2D([0.505, 0.505], [0.02, 0.92],
                      transform=fig.transFigure, color='#cccccc', lw=1.5)
    fig.add_artist(line)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(OUTPUT_DIR) / save_name
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading models ...")
osu_model   = load_model(OSU_CHECKPOINT)
chile_model = load_model(CHILE_CHECKPOINT)

print("\nLoading specific normal images ...")
osu_normal_paths   = get_named_image_paths(OSU_NORMAL_DIR,   OSU_IMAGE_NAMES)
chile_normal_paths = get_named_image_paths(CHILE_NORMAL_DIR, CHILE_IMAGE_NAMES)
print(f"  OSU:   {len(osu_normal_paths)} images found")
print(f"  Chile: {len(chile_normal_paths)} images found")
for p in osu_normal_paths:
    print(f"    OSU:   {p.name}")
for p in chile_normal_paths:
    print(f"    Chile: {p.name}")

# OSU-important features
print(f"\nGrad-CAM — OSU-important features ...")
for rank, feat_idx in enumerate(osu_important_idx, 1):
    osu_w   = osu_weights[feat_idx]
    chile_w = chile_weights[feat_idx]
    print(f"  Rank {rank}: Feature {feat_idx} "
          f"(OSU={osu_w:.4f}, Chile={chile_w:.4f})")
    generate_gradcam_figure(
        osu_model, chile_model,
        osu_normal_paths, chile_normal_paths,
        feature_idx=feat_idx, feature_rank=rank,
        osu_weight=osu_w, chile_weight=chile_w,
        list_label=f'OSU-important rank {rank}',
        save_name=f'gradcam_probe_OSUimportant_feat{feat_idx}_rank{rank}.png'
    )

# Chile-important features
print(f"\nGrad-CAM — Chile-important features ...")
for rank, feat_idx in enumerate(chile_important_idx, 1):
    osu_w   = osu_weights[feat_idx]
    chile_w = chile_weights[feat_idx]
    print(f"  Rank {rank}: Feature {feat_idx} "
          f"(OSU={osu_w:.4f}, Chile={chile_w:.4f})")
    generate_gradcam_figure(
        osu_model, chile_model,
        osu_normal_paths, chile_normal_paths,
        feature_idx=feat_idx, feature_rank=rank,
        osu_weight=osu_w, chile_weight=chile_w,
        list_label=f'Chile-important rank {rank}',
        save_name=f'gradcam_probe_Chileimportant_feat{feat_idx}_rank{rank}.png'
    )

print(f"\nDone. All outputs saved to {OUTPUT_DIR}")
