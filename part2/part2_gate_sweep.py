import subprocess, sys
packages = [
    "segmentation-models-pytorch==0.3.3",
    "albumentations==1.3.1",
    "torchmetrics==1.2.0",
    "torchcam",
    "scikit-image",
    "tqdm",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("Packages installed")

import os, sys, json, random, warnings, time, pickle, copy
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
try:
    import seaborn as sns
except Exception:
    sns = None
    print("WARNING: seaborn unavailable (numpy/scipy mismatch) -- using matplotlib defaults.")
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

warnings.filterwarnings("ignore")
if sns is not None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "figure.figsize": (14, 6)})

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device     : {DEVICE}')
print(f'PyTorch    : {torch.__version__}')
print(f'CUDA       : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU        : {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB')
assert torch.cuda.is_available(), "GPU required for Part 2 -- enable GPU on Kaggle"
print("Imports complete.")

IN_KAGGLE = False  # RUNPOD RUN (patched)
if IN_KAGGLE:
    WORKING_DIR = "/kaggle/working"
else:
    WORKING_DIR = '/workspace/working'  # RUNPOD RUN (patched) -- same dir Part 1 wrote to
    os.makedirs(WORKING_DIR, exist_ok=True)
OUTPUT_DIR  = os.path.join(WORKING_DIR, 'fed_salgate_outputs')
CKPT_DIR    = os.path.join(WORKING_DIR, 'checkpoints')
PLOT_DIR    = os.path.join(WORKING_DIR, 'plots')
MASK_OUT_DIR = os.path.join(WORKING_DIR, 'masks')
for d in [OUTPUT_DIR, CKPT_DIR, PLOT_DIR, MASK_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Locate Part 1 outputs (read-only) ─────────────────────────────────────────
# Preferred: same-session /kaggle/working/fed_salgate_outputs (Part 1 just ran).
# Fallback: a Kaggle attachment in /kaggle/input that contains Part 1's outputs.
# The attachment can be nested at any depth, e.g.:
#   /kaggle/input/notebooks/<owner>/<slug>/fed_salgate_outputs/
#   /kaggle/input/<dataset>/fed_salgate_outputs/
#   /kaggle/input/<dataset>/config.json
# So we walk recursively and return the FIRST directory that contains ALL three
# of: config.json, client_indices.pkl, gt_val_indices.pkl.
def _find_part1_dir():
    if os.path.exists(os.path.join(OUTPUT_DIR, 'config.json')):
        return OUTPUT_DIR
    if IN_KAGGLE:
        input_dir = "/kaggle/input"
        if os.path.isdir(input_dir):
            hits = []
            for root, dirs, files in os.walk(input_dir):
                if ('config.json' in files and 'client_indices.pkl' in files
                        and 'gt_val_indices.pkl' in files):
                    hits.append(root)
                    if len(hits) >= 5:  # avoid scanning the whole tree twice
                        break
            if hits:
                for h in hits:  # prefer an exact fed_salgate_outputs folder
                    if os.path.basename(h) == 'fed_salgate_outputs':
                        return h
                return hits[0]
            # fallback: any dir holding a config.json that looks like Part 1's
            for root, dirs, files in os.walk(input_dir):
                if 'config.json' in files:
                    try:
                        with open(os.path.join(root, 'config.json')) as f:
                            cfg = json.load(f)
                        if 'FLOODNET_BASE' in cfg or 'IMG_SIZE' in cfg:
                            return root
                    except Exception:
                        pass
    return None

PART1_DIR = _find_part1_dir()
if PART1_DIR is not None:
    print(f'Part 1 outputs found at: {PART1_DIR}')
else:
    print('WARNING: Part 1 outputs NOT found -- falling back to defaults + '
          'Dirichlet re-partition. Results will NOT match Part 1 splits!')

# ── Locate Part 1 pseudo-masks (read-only, inside the Part 1 attachment) ──
# PART1_DIR points at .../fed_salgate_outputs; the masks live in the SIBLING
# .../masks/ folder of the same Part 1 attachment. On a fresh Part 2 Kaggle
# session /kaggle/working/masks is empty, so we MUST read the pseudo-masks
# from the attachment, otherwise unlabeled/val images would get all-zero masks.
PART1_ROOT = os.path.dirname(PART1_DIR) if PART1_DIR else None
if PART1_ROOT and os.path.isdir(os.path.join(PART1_ROOT, 'masks')):
    MASK_OUT_DIR = os.path.join(PART1_ROOT, 'masks')
    print(f'Part 1 pseudo-masks (read-only): {MASK_OUT_DIR}')
    for sub in ['unlabeled', 'val', 'test']:
        p = os.path.join(MASK_OUT_DIR, sub)
        n = len([f for f in os.listdir(p) if not f.startswith('.')]) if os.path.isdir(p) else 0
        print(f'    masks/{sub}: {n} files')
else:
    print(f'Using writable mask output dir: {MASK_OUT_DIR}')

# ── Load Part 1 config ───────────────────────────────────────────────────────
config_path = os.path.join(PART1_DIR, 'config.json') if PART1_DIR else None
if config_path and os.path.exists(config_path):
    with open(config_path) as f:
        CFG = json.load(f)
    print("Part 1 config loaded")
    for k, v in CFG.items():
        if not isinstance(v, list):
            print(f'  {k}: {v}')
else:
    print("WARNING: Part 1 config not found. Using default values.")
    CFG = {}

NUM_CLASSES   = int(CFG.get('NUM_CLASSES', 10))
CLASS_NAMES   = CFG.get('CLASS_NAMES', [
    'Background', 'Building-Flooded', 'Building-NonFlooded',
    'Road-Flooded', 'Road-NonFlooded', 'Water',
    'Tree', 'Vehicle', 'Pool', 'Grass'])
IMG_SIZE      = int(CFG.get('IMG_SIZE', 512))
BATCH_SIZE    = int(CFG.get('BATCH_SIZE', 4))
NUM_CLIENTS   = int(CFG.get('NUM_CLIENTS', 3))
NUM_ROUNDS    = int(CFG.get('NUM_ROUNDS', 50))
LOCAL_EPOCHS  = int(CFG.get('LOCAL_EPOCHS', 3))
LR            = float(CFG.get('LR', 1e-4))
PROXIMAL_MU   = float(CFG.get('PROXIMAL_MU', 0.01))

# ── Fed-SalGate v2 hyper-parameters ──────────────────────────────────────────
# CLIENT_SUBSAMPLE: images a client trains on per round (random draw). Keeps
# per-round cost bounded while preserving the federated dynamic (each round a
# fresh random subset of the client's local data is seen).
CLIENT_SUBSAMPLE = int(CFG.get('CLIENT_SUBSAMPLE', 96))
FL_LOCAL_EPOCHS  = int(CFG.get('FL_LOCAL_EPOCHS', 2))
GATE_BETA   = float(CFG.get('GATE_BETA', 10.0))   # sigmoid steepness
GATE_TAU    = float(CFG.get('GATE_TAU', 0.5))     # agreement threshold
PROBE_K     = int(CFG.get('PROBE_K', 8))          # val images for saliency consensus
VAL_LOG_N   = int(CFG.get('VAL_LOG_N', 128))      # val images for round tracking
DEBUG_SHORT = os.environ.get('PART2_DEBUG') == '1'
if DEBUG_SHORT:
    NUM_ROUNDS = 6
    print('!! DEBUG_SHORT mode: NUM_ROUNDS =', NUM_ROUNDS)

# ── Load Part 1 client indices (order: Flooded -> Non-Flooded -> Unlabeled) ──
client_indices_path = os.path.join(PART1_DIR, 'client_indices.pkl') if PART1_DIR else None
if client_indices_path and os.path.exists(client_indices_path):
    with open(client_indices_path, 'rb') as f:
        CLIENT_INDICES = pickle.load(f)
    print(f'Client indices loaded: {[len(c) for c in CLIENT_INDICES]}')
else:
    print("WARNING: client_indices.pkl not found. Re-partitioning (Dirichlet fallback).")
    CLIENT_INDICES = None

# ── Load Part 1 held-out real-GT indices (never trained on by any client) ────
GT_VAL_INDICES = CFG.get('gt_val_indices')
if GT_VAL_INDICES is None:
    gt_pkl = os.path.join(PART1_DIR, 'gt_val_indices.pkl') if PART1_DIR else None
    if gt_pkl and os.path.exists(gt_pkl):
        with open(gt_pkl, 'rb') as f:
            GT_VAL_INDICES = pickle.load(f)
if GT_VAL_INDICES is not None:
    print(f'Held-out real-GT indices loaded: {len(GT_VAL_INDICES)} photos')
else:
    print('WARNING: held-out GT indices not found -- final eval will use the pseudo-masked val split only.')

COLOR_MAP = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [128, 0, 0],
    3: [255, 255, 0], 4: [128, 128, 0], 5: [0, 0, 255],
    6: [0, 128, 0], 7: [128, 128, 128], 8: [0, 255, 255], 9: [0, 255, 0],
}

print('\nConfig + Part 1 outputs ready.')

class FloodNetDataset(Dataset):
    """FloodNet Track-1 dataset (identical ordering to Part 1 v2 so that the
    saved client partitions and held-out GT indices align 1:1)."""
    def __init__(self, base_path, split='train', transform=None, mask_transform=None,
                 require_mask=True, pseudo_mask_root=None, include_unlabeled=False, augment=False):
        self.transform        = transform
        self.mask_transform   = mask_transform
        self.require_mask     = require_mask
        self.pseudo_mask_root = pseudo_mask_root
        self.augment          = augment
        self.samples = []

        split_dir_map = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        split_folder  = os.path.join(base_path, split_dir_map[split])
        if not os.path.exists(split_folder):
            print(f'Not found: {split_folder}'); return

        if split == 'train':
            self._add_labeled(os.path.join(split_folder, 'Labeled', 'Flooded',     'image'),
                              os.path.join(split_folder, 'Labeled', 'Flooded',     'mask'), 'flooded')
            self._add_labeled(os.path.join(split_folder, 'Labeled', 'Non-Flooded', 'image'),
                              os.path.join(split_folder, 'Labeled', 'Non-Flooded', 'mask'), 'nonflooded')
            if include_unlabeled and pseudo_mask_root:
                self._add_unlabeled(os.path.join(split_folder, 'Unlabeled', 'image'),
                                    os.path.join(pseudo_mask_root, 'unlabeled'))
        else:
            pseudo_dir = os.path.join(pseudo_mask_root, split) if pseudo_mask_root else None
            self._add_unlabeled(os.path.join(split_folder, 'image'), pseudo_dir)
        print(f'[{split.upper()}] Loaded {len(self.samples)} samples')

    def _add_labeled(self, img_dir, msk_dir, label):
        if not os.path.exists(img_dir): return
        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg",".jpeg",".png",".tif"))
                       and not f.startswith(".")])
        mask_by_stem = {}
        if os.path.exists(msk_dir):
            for mf in os.listdir(msk_dir):
                if not mf.lower().endswith((".png",".jpg",".jpeg",".tif")) or mf.startswith("."): continue
                stem = os.path.splitext(mf)[0]
                mask_by_stem.setdefault(stem, mf)
                for suffix in ("_lab", "_mask", "_label", "_gt"):
                    if stem.endswith(suffix):
                        mask_by_stem.setdefault(stem[:-len(suffix)], mf)
        for fn in imgs:
            img_path  = os.path.join(img_dir, fn)
            base      = os.path.splitext(fn)[0]
            mask_path = os.path.join(msk_dir, mask_by_stem[base]) if base in mask_by_stem else None
            if self.require_mask and mask_path is None: continue
            self.samples.append({"img": img_path, "mask": mask_path,
                                 "is_pseudo": False, "label": label})

    def _add_unlabeled(self, img_dir, msk_dir):
        if not os.path.exists(img_dir): return
        exts = (".jpg",".jpeg",".png",".tif")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(exts) or fn.startswith('.'): continue
            img_path  = os.path.join(img_dir, fn)
            base      = os.path.splitext(fn)[0]
            mask_path = None
            if msk_dir and os.path.exists(msk_dir):
                for suffix in [f"{base}_mask.png", f"{base}.png"]:
                    cand = os.path.join(msk_dir, suffix)
                    if os.path.exists(cand):
                        mask_path = cand; break
            self.samples.append({"img": img_path, "mask": mask_path,
                                 "is_pseudo": True, "label": "unknown"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["img"]).convert("RGB")
        if s["mask"] and os.path.exists(s["mask"]):
            raw  = np.array(Image.open(s["mask"]))
            if raw.ndim == 3: raw = raw[:, :, 0]
            mask = np.clip(raw, 0, NUM_CLASSES - 1)
        else:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
        mask_pil = Image.fromarray(mask)

        if self.augment:
            if random.random() < 0.5:
                img = transforms.functional.hflip(img); mask_pil = transforms.functional.hflip(mask_pil)
            if random.random() < 0.3:
                img = transforms.functional.vflip(img); mask_pil = transforms.functional.vflip(mask_pil)
            angle = random.uniform(-15, 15)
            img      = transforms.functional.rotate(img,      angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask_pil = transforms.functional.rotate(mask_pil, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)

        if self.transform:        img  = self.transform(img)
        if self.mask_transform:
            mask_tensor = torch.from_numpy(np.array(self.mask_transform(mask_pil))).long()
        else:
            mask_tensor = torch.from_numpy(mask).long()
        return img, mask_tensor, os.path.basename(s["img"]), s["is_pseudo"]


# ── Transform factories ──────────────────────────────────────────────────────
def get_train_transforms(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_val_transforms(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_mask_transforms(img_size=512):
    return transforms.Compose([transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)])

print('FloodNetDataset + transforms defined.')

FLOODNET_BASE = CFG.get('FLOODNET_BASE', '')
if not FLOODNET_BASE or not os.path.exists(FLOODNET_BASE):
    if IN_KAGGLE:
        # Same recursive discovery as Part 1: /kaggle/input/.../<slug>/Train
        input_dir = "/kaggle/input"
        for root, dirs, files in os.walk(input_dir):
            if 'Train' in dirs:
                FLOODNET_BASE = root
                print(f'Found Train/ at: {root}')
                break
    if not FLOODNET_BASE or not os.path.exists(FLOODNET_BASE):
        raise FileNotFoundError(f'FloodNet dataset not found at {FLOODNET_BASE}')
print(f'FLOODNET_BASE: {FLOODNET_BASE}')

train_dataset = FloodNetDataset(
    FLOODNET_BASE, split='train',
    transform=get_train_transforms(IMG_SIZE),
    mask_transform=get_mask_transforms(IMG_SIZE),
    require_mask=True, pseudo_mask_root=MASK_OUT_DIR,
    include_unlabeled=True, augment=True)
val_dataset = FloodNetDataset(
    FLOODNET_BASE, split='val',
    transform=get_val_transforms(IMG_SIZE),
    mask_transform=get_mask_transforms(IMG_SIZE),
    require_mask=False, pseudo_mask_root=MASK_OUT_DIR, augment=False)
print(f'Train: {len(train_dataset)}  Val: {len(val_dataset)}')

# ── Fallback: Dirichlet Non-IID partition (identical to Part 1) ──────────────
if CLIENT_INDICES is None:
    print('Re-partitioning clients (Dirichlet fallback)...')
    np.random.seed(SEED)
    labels = []
    for i in tqdm(range(len(train_dataset))):
        _, msk, _, _ = train_dataset[i]
        labels.append(int(torch.bincount(msk.flatten(), minlength=NUM_CLASSES).argmax()))
    labels = np.array(labels)
    CLIENT_INDICES = [[] for _ in range(NUM_CLIENTS)]
    alpha = float(CFG.get('DIRICHLET_ALPHA', 0.5))
    for cls in range(NUM_CLASSES):
        idx_k = np.where(labels == cls)[0]
        if len(idx_k) == 0: continue
        np.random.shuffle(idx_k)
        props = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        splits = (np.cumsum(props / props.sum()) * len(idx_k)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx_k, splits)):
            CLIENT_INDICES[cid].extend(chunk.tolist())
    for ci in CLIENT_INDICES: np.random.shuffle(ci)
    if GT_VAL_INDICES is not None:
        CLIENT_INDICES = [[i for i in ci if i not in GT_VAL_INDICES] for ci in CLIENT_INDICES]
    print(f'  Fallback client sizes: {[len(c) for c in CLIENT_INDICES]}')

# ── Non-augmented copy of the train split for CLEAN per-client evaluation ─────
# (same deterministic ordering => the saved client indices align 1:1)
train_dataset_eval = FloodNetDataset(
    FLOODNET_BASE, split='train',
    transform=get_val_transforms(IMG_SIZE),
    mask_transform=get_mask_transforms(IMG_SIZE),
    require_mask=True, pseudo_mask_root=MASK_OUT_DIR,
    include_unlabeled=True, augment=False)

# ── Per-client eval loaders (used for cross-region generalization, Gap 3) ────
client_eval_loaders = []
for ci, idx in enumerate(CLIENT_INDICES):
    loader = DataLoader(Subset(train_dataset_eval, idx), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
    client_eval_loaders.append(loader)
    print(f'  Client {ci}: {len(idx)} samples, {len(loader)} batches')

# ── Round-logging val subset (pseudo-masked, fixed) ──────────────────────────
val_log_idx = list(range(0, len(val_dataset), max(1, len(val_dataset) // VAL_LOG_N)))[:VAL_LOG_N]
val_log_loader = DataLoader(Subset(val_dataset, val_log_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
print(f'Round-log val subset: {len(val_log_idx)} images')

# ── Saliency-consensus probe set (fixed K images, shared by all clients) ─────
probe_idx = list(range(0, len(val_dataset), max(1, len(val_dataset) // PROBE_K)))[:PROBE_K]
probe_loader = DataLoader(Subset(val_dataset, probe_idx), batch_size=1,
                          shuffle=False, num_workers=4)
print(f'Saliency probe set: {len(probe_idx)} images (shared across clients)')

# ── Honest held-out GT loader (real human-checked masks, never trained on) ──
if GT_VAL_INDICES is not None:
    gt_val_loader = DataLoader(Subset(train_dataset, GT_VAL_INDICES), batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4)
    print(f'Held-out GT eval set: {len(GT_VAL_INDICES)} real photos')
else:
    gt_val_loader = val_log_loader
    print('WARNING: using pseudo-masked val subset for final eval.')

# ── Full val loader (final full-val score) ───────────────────────────────────
full_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print('Loaders ready.')

# ── Fed-SalGate attention module (SPADANet-inspired, used inside LiteUNet) ──
class SaliencyGate(nn.Module):
    """Lightweight channel saliency gate -- spatial prior attention (SPADANet-inspired)."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, max(4, in_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, in_channels // reduction), in_channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        avg  = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        combined = torch.cat([avg, max_], dim=1)
        gate = self.fc(combined).view(b, c, 1, 1)
        return x * gate


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class LiteUNet(nn.Module):
    """Gap 4: lightweight U-Net with saliency gates -- drone/edge-suitable."""
    def __init__(self, in_channels=3, num_classes=10, features=(16, 32, 64, 128)):
        super().__init__()
        self.downs          = nn.ModuleList()
        self.ups            = nn.ModuleList()
        self.saliency_gates = nn.ModuleList()
        self.pool           = nn.MaxPool2d(kernel_size=2, stride=2)
        in_ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            self.saliency_gates.append(SaliencyGate(f))
            in_ch = f
        self.bottleneck      = DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_gate = SaliencyGate(features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
            self.saliency_gates.append(SaliencyGate(f))
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down, gate in zip(self.downs, self.saliency_gates[:len(self.downs)]):
            x = down(x)
            x = gate(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.bottleneck_gate(x)
        skips = skips[::-1]
        gate_idx = len(self.downs)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = transforms.functional.resize(x, s.shape[2:])
            x = torch.cat((s, x), dim=1)
            x = self.ups[i + 1](x)
            x = self.saliency_gates[gate_idx](x)
            gate_idx += 1
        return self.final_conv(x)


def make_deeplabv3p(in_channels=3, num_classes=10):
    """Part 1's DeepLabV3+ backbone (ResNet-34 + ASPP), ImageNet-pretrained."""
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(
        encoder_name='resnet34', encoder_weights='imagenet',
        in_channels=in_channels, classes=num_classes)


# ── Sanity checks + param counts ─────────────────────────────────────────────
m_lite = LiteUNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
m_dlv3 = make_deeplabv3p(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

n_lite = count_params(m_lite)
n_dlv3 = count_params(m_dlv3)
print(f'LiteUNet      params: {n_lite:,}  ({n_lite/1e6:.2f} M)  -- Gap 4 drone/edge variant')
print(f'DeepLabV3+    params: {n_dlv3:,}  ({n_dlv3/1e6:.2f} M)  -- Part 1 backbone')
print(f'Param ratio   : DeepLabV3+ is {n_dlv3/max(1, n_lite):.1f}x larger than LiteUNet')

# eval() is required: DeepLabV3+'s ASPP global-pool branch produces a 1x1
# map, and BatchNorm in train() mode rejects batch size 1.
m_lite.eval(); m_dlv3.eval()
x = torch.randn(1, 3, 128, 128).to(DEVICE)
with torch.no_grad(), torch.cuda.amp.autocast():
    yl = m_lite(x); yd = m_dlv3(x)
print(f'LiteUNet   out: {yl.shape}   DeepLabV3+ out: {yd.shape}')
del m_lite, m_dlv3, x, yl, yd
torch.cuda.empty_cache()

def compute_miou(model, loader, num_classes=NUM_CLASSES, max_batches=None):
    """Returns (mean IoU, per-class IoU dict, pixel accuracy)."""
    model.eval()
    ious  = {c: [] for c in range(num_classes)}
    pix_correct = 0
    pix_total   = 0
    n_batches   = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, msks, _, _ in loader:
            imgs  = imgs.to(DEVICE)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            gts   = msks.numpy()
            for p, g in zip(preds, gts):
                pix_correct += (p == g).sum()
                pix_total   += g.size
                for c in range(num_classes):
                    gt_c, pr_c = (g == c), (p == c)
                    union = (gt_c | pr_c).sum()
                    if union > 0:
                        ious[c].append((gt_c & pr_c).sum() / union)
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break
    vals = [float(np.mean(v)) for v in ious.values() if len(v) > 0]
    miou = float(np.mean(vals)) if vals else 0.0
    per_class = {c: float(np.mean(v)) if v else 0.0 for c, v in ious.items()}
    pix_acc = float(pix_correct) / float(pix_total) if pix_total > 0 else 0.0
    return miou, per_class, pix_acc


def generate_gradcam(model, image_tensor, target_layer=None):
    """Grad-CAM saliency map on the top predicted class."""
    model.eval()
    gradients   = None
    activations = None

    def forward_hook(m, i, o):
        nonlocal activations
        activations = o.detach()

    def backward_hook(m, gi, go):
        nonlocal gradients
        gradients = go[0].detach()

    if target_layer is None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
    if target_layer is None:
        return None

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        output = model(image_tensor.unsqueeze(0).to(DEVICE))
        # segmentation output is [1, C, H, W]; pick the dominant predicted class
        preds = output.argmax(dim=1).flatten()
        target_class = int(preds.bincount(minlength=output.shape[1]).argmax().item())
        model.zero_grad()
        output[0, target_class].sum().backward()
    except Exception:
        return None
    finally:
        h1.remove(); h2.remove()

    if gradients is None or activations is None:
        return None
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def avg_saliency_map(model, probe_loader):
    """Average normalized Grad-CAM over the shared probe set."""
    model.eval()
    maps = []
    for imgs, _, _, _ in probe_loader:
        cam = generate_gradcam(model, imgs[0])
        if cam is not None:
            maps.append(cam)
    if len(maps) == 0:
        return None
    return np.mean(np.stack(maps), axis=0)


def soft_iou(a, b, eps=1e-6):
    """Continuous Jaccard between two normalized maps."""
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum() + eps
    return float(inter / union)


def saliency_consensus_gates(client_models, probe_loader, beta=GATE_BETA, tau=GATE_TAU):
    """
    Gap 2 -- spatial saliency-consensus gating.
    For each client k: agreement = soft-IoU(client_k Grad-CAM,
                                           mean of OTHER clients' Grad-CAMs).
    Gate = sigmoid(beta*(agreement - tau)). Gates are normalized to sum to
    NUM_CLIENTS so that when all clients agree equally, Fed-SalGate == FedAvg.
    Returns (gates array, agreements array).
    """
    cams = []
    for m in client_models:
        cam = avg_saliency_map(m, probe_loader)
        if cam is None:
            cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        cams.append(cam)
    agreements = []
    for k in range(len(cams)):
        others = np.mean([c for i, c in enumerate(cams) if i != k], axis=0)
        agreements.append(soft_iou(cams[k], others))
    agg = np.array(agreements)
    gates = 1.0 / (1.0 + np.exp(-beta * (agg - tau)))
    gates = gates / (gates.sum() + 1e-8) * len(gates)
    return gates.tolist(), agreements


print('Metrics + Grad-CAM + saliency-consensus gating defined.')

def fed_aggregate(weights_list, client_weights):
    """Weighted aggregation. client_weights[0..K) -- FedAvg uses uniform."""
    w = torch.tensor(client_weights, dtype=torch.float32, device=DEVICE)
    w = w / (w.sum() + 1e-8)
    global_dict = {}
    for key in weights_list[0].keys():
        stacked = torch.stack([ww[key].float() for ww in weights_list])   # [K, ...]
        shape = [len(weights_list)] + [1] * (stacked.dim() - 1)
        global_dict[key] = (stacked * w.view(shape)).sum(dim=0)
    return global_dict


def _local_train(global_model, client_idx, backbone_fn, local_epochs, lr,
                 proximal_mu=0.0, saliency_lambda=0.0, round_idx=1):
    """Train one client model on a fresh random subset of its local data."""
    idx_all = np.array(CLIENT_INDICES[client_idx])
    rng = np.random.RandomState(SEED + client_idx * 7919 + round_idx * 104729)
    if len(idx_all) > CLIENT_SUBSAMPLE:
        idx = idx_all[rng.choice(len(idx_all), CLIENT_SUBSAMPLE, replace=False)]
    else:
        idx = idx_all
    sub_ds   = Subset(train_dataset, idx.tolist())
    loader   = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True,
                          drop_last=(len(sub_ds) > BATCH_SIZE))  # RUNPOD FIX: avoid a size-1 last
                          # batch crashing BatchNorm when a small client's sample count isn't a
                          # multiple of BATCH_SIZE (real issue with 5-client/alpha=0.3 split, e.g.
                          # a 33-sample client). Guarded so a client smaller than BATCH_SIZE still
                          # gets at least one (partial) batch instead of zero.

    client_model = backbone_fn().to(DEVICE)
    client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    global_params = list(global_model.parameters())
    client_model.train()
    total_loss = 0.0; n_seen = 0
    for _ in range(local_epochs):
        for imgs, msks, _, _ in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = client_model(imgs)
                seg_loss = criterion(logits, msks)
            loss = seg_loss
            if proximal_mu > 0:
                prox = 0.0
                for p, p0 in zip(client_model.parameters(), global_params):
                    prox += ((p - p0.detach()) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * prox
            if saliency_lambda > 0:
                gate_act = []
                for module in client_model.modules():
                    if isinstance(module, SaliencyGate):
                        gate_act.append(module.fc[-1].weight.abs().mean())
                if gate_act:
                    loss = loss + saliency_lambda * torch.stack(gate_act).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            n_seen     += imgs.size(0)
    avg_loss = total_loss / max(1, n_seen)
    del loader
    return client_model, avg_loss


def run_federated(name, backbone_fn, aggregator='fedavg',
                  rounds=NUM_ROUNDS, local_epochs=FL_LOCAL_EPOCHS, lr=LR,
                  proximal_mu=PROXIMAL_MU):
    """Generic federated runner.
    aggregator: 'fedavg' | 'fedprox' | 'fedsalgate'
    """
    global_model = backbone_fn().to(DEVICE)
    history = {"round": [], "avg_train_loss": [], "val_miou": [],
               "per_class_iou": [], "gates": [], "agreements": []}

    for round_idx in range(1, rounds + 1):
        local_weights  = []
        local_models   = []
        local_losses   = []

        for client_idx in range(NUM_CLIENTS):
            cm, al = _local_train(global_model, client_idx, backbone_fn,
                                  local_epochs, lr,
                                  proximal_mu if aggregator == 'fedprox' else 0.0,
                                  round_idx=round_idx)
            local_weights.append(copy.deepcopy(cm.state_dict()))
            local_models.append(cm)
            local_losses.append(al)

        # ── aggregation ──────────────────────────────────────────────────
        if aggregator == 'fedsalgate':
            gates, agreements = saliency_consensus_gates(local_models, probe_loader,
                                                        beta=GATE_BETA, tau=GATE_TAU)  # GATE SWEEP: read globals at call time, not def-time defaults
        else:
            gates, agreements = [1.0 / NUM_CLIENTS] * NUM_CLIENTS, [0.0] * NUM_CLIENTS
        global_dict = fed_aggregate(local_weights, gates)
        global_model.load_state_dict(global_dict)

        # ── round logging (fast subset of pseudo-masked val) ─────────────
        val_miou, per_class, _ = compute_miou(global_model, val_log_loader)

        history["round"].append(round_idx)
        history["avg_train_loss"].append(float(np.mean(local_losses)))
        history["val_miou"].append(val_miou)
        history["per_class_iou"].append(per_class)
        history["gates"].append(gates)
        history["agreements"].append(agreements)

        for m in local_models: del m
        torch.cuda.empty_cache()

        if round_idx % 5 == 0 or round_idx == 1:
            gate_str = ' | '.join(f'{g:.3f}' for g in gates)
            print(f'[{name}] Round {round_idx:3d}/{rounds}  loss={history["avg_train_loss"][-1]:.4f}'
                  f'  val_mIoU={val_miou:.4f}  gates=[{gate_str}]')

    return global_model, history


print('Federated training functions defined.')

# ══════════════════════════════════════════════════════════════════════════
# GATE HYPERPARAMETER SWEEP (patched run, 2026-08-10)
#
# Diagnosis: the completed 5-client/alpha=0.3 run showed final agreements in
# [0.769, 0.930] but final gates only in [0.972, 1.024] -- because tau=0.5 is
# far below the real agreement range, sigmoid(beta*(agreement-tau)) saturates
# near 1.0 for every client regardless of their *relative* differences.
# Recentering tau near the real agreement range should restore differentiation.
#
# FedAvg/FedProx are NOT retrained here -- their held-out mIoU is already
# computed for this exact 5-client/alpha=0.3 split (part2_v2_summary.json)
# and is reused as the comparison baseline below.
# ══════════════════════════════════════════════════════════════════════════

def final_eval(model, loader, tag=""):
    miou, per_class, pix_acc = compute_miou(model, loader)
    print(f'  {tag:>34}  mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}')
    return miou, per_class, pix_acc

CACHED_FEDAVG_MIOU  = 0.4145049642555153
CACHED_FEDPROX_MIOU = 0.4071457958136905

sweep_configs = [
    {"name": "tau0.85_beta10", "tau": 0.85, "beta": 10.0},
    {"name": "tau0.85_beta20", "tau": 0.85, "beta": 20.0},
    {"name": "tau0.80_beta15", "tau": 0.80, "beta": 15.0},
]

sweep_results = {}
sweep_out_path = os.path.join(OUTPUT_DIR, 'gate_sweep_results.json')

for cfg in sweep_configs:
    GATE_TAU  = cfg["tau"]
    GATE_BETA = cfg["beta"]
    print("=" * 70)
    print(f"GATE SWEEP: {cfg['name']}  (tau={GATE_TAU}, beta={GATE_BETA})")
    print("=" * 70)
    t0 = time.time()
    model, history = run_federated(f"Fed-SalGate-{cfg['name']}", make_deeplabv3p, aggregator='fedsalgate')
    elapsed = (time.time() - t0) / 60
    print(f'{cfg["name"]} done in {elapsed:.1f} min | best round mIoU = {max(history["val_miou"]):.4f}')
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'fedsalgate_{cfg["name"]}_final.pth'))

    print("=" * 70)
    print(f"FINAL EVALUATION (held-out real GT) -- {cfg['name']}")
    print("=" * 70)
    miou, per_class, pix_acc = final_eval(model, gt_val_loader, f"Fed-SalGate {cfg['name']}")
    final_gates = history["gates"][-1]
    final_agreements = history["agreements"][-1]
    print("  Final gates:", [f"{g:.4f}" for g in final_gates])
    print("  Final agreements:", [f"{a:.4f}" for a in final_agreements])

    sweep_results[cfg["name"]] = {
        "gate_tau": GATE_TAU, "gate_beta": GATE_BETA,
        "heldout_miou": miou, "pixel_acc": pix_acc,
        "best_round_miou": max(history["val_miou"]),
        "per_class_iou": {str(k): v for k, v in per_class.items()},
        "final_gates": final_gates, "final_agreements": final_agreements,
        "gain_vs_fedavg": miou - CACHED_FEDAVG_MIOU,
        "gain_vs_fedprox": miou - CACHED_FEDPROX_MIOU,
        "elapsed_min": elapsed,
    }

    with open(sweep_out_path, 'w') as f:
        json.dump({
            "num_clients": NUM_CLIENTS,
            "dirichlet_alpha": CFG.get("DIRICHLET_ALPHA"),
            "cached_baselines": {"fedavg_heldout_miou": CACHED_FEDAVG_MIOU,
                                  "fedprox_heldout_miou": CACHED_FEDPROX_MIOU},
            "original_gate_result": {"gate_beta": 10.0, "gate_tau": 0.5,
                                      "heldout_miou": 0.4086200788405887},
            "sweep": sweep_results,
        }, f, indent=2)
    print(f"Progress saved: {sweep_out_path}")

    del model
    torch.cuda.empty_cache()

print("=" * 70)
print("GATE SWEEP COMPLETE")
print("=" * 70)
print(f'{"config":<20} {"heldout_miou":>14} {"vs FedAvg":>12} {"vs FedProx":>12}')
for name, r in sweep_results.items():
    print(f'{name:<20} {r["heldout_miou"]:>14.4f} {r["gain_vs_fedavg"]:>+12.4f} {r["gain_vs_fedprox"]:>+12.4f}')
print(f'{"(original tau=0.5,beta=10)":<20} {0.4086200788405887:>14.4f} {0.4086200788405887-CACHED_FEDAVG_MIOU:>+12.4f} {0.4086200788405887-CACHED_FEDPROX_MIOU:>+12.4f}')
