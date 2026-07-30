"""
dataset.py — загрузчик датасета CDD-CESM по CSV-аннотациям.

Поддерживает 3 класса: Normal | Benign | Malignant.
Используется обоими пайплайнами (CNN→SHAP+SAM и CNN→SHAP→MASK).
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split

SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CesmCsvDataset(Dataset):
    """Датасет CDD-CESM с разметкой из CSV-файла."""

    CLASS_NAMES: Tuple[str, ...] = ("Normal", "Benign", "Malignant")

    def __init__(
        self,
        images_dir: Path,
        annotations_csv: Path,
        class_names: Tuple[str, ...] = CLASS_NAMES,
        transform=None,
    ):
        self.images_dir = images_dir
        self.annotations_csv = annotations_csv
        self.class_names = class_names
        self.class_to_index: Dict[str, int] = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self._load()

    # ── private ───────────────────────────────────────────────────────────────

    def _normalize_label(self, txt: str) -> Optional[str]:
        if not txt:
            return None
        s = txt.strip().split("/")[0].strip().lower()
        if s.startswith("malig"):
            return "Malignant"
        if s.startswith("benig"):
            return "Benign"
        if s.startswith("norm"):
            return "Normal"
        return None

    def _find_columns(self, header: List[str]) -> Tuple[int, int]:
        img_col = cls_col = -1
        for i, name in enumerate(header):
            ln = name.strip().lower()
            if img_col == -1 and ("image" in ln and "name" in ln):
                img_col = i
            if cls_col == -1 and ("pathology" in ln and "classification" in ln):
                cls_col = i
        return img_col, cls_col

    def _resolve_image_path(self, image_name: str) -> Optional[Path]:
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            p = self.images_dir / f"{image_name}{ext}"
            if _safe_exists(p):
                return p
        # prefix-match fallback (handles duplicates like *_2)
        candidates: List[Path] = []
        for pat in [f"{image_name}*.jpg", f"{image_name}*.JPG", f"{image_name}*.png"]:
            candidates.extend(self.images_dir.glob(pat))
        return sorted(candidates)[0] if candidates else None

    def _load(self) -> None:
        if not _safe_exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not _safe_exists(self.annotations_csv):
            raise FileNotFoundError(f"Annotations CSV not found: {self.annotations_csv}")
        missing = used = 0
        with open(self.annotations_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)
        if not rows:
            raise RuntimeError("Empty annotations CSV")
        header = rows[0]
        img_col, cls_col = self._find_columns(header)
        if img_col < 0 or cls_col < 0:
            raise RuntimeError(
                "Cannot find columns 'Image_name' / 'Pathology Classification' in CSV"
            )
        for row in rows[1:]:
            if not row or len(row) <= max(img_col, cls_col):
                continue
            image_name = row[img_col].strip()
            cls_norm = self._normalize_label(row[cls_col])
            if cls_norm is None or cls_norm not in self.class_to_index:
                continue
            path = self._resolve_image_path(image_name)
            if path is None:
                missing += 1
                continue
            self.image_paths.append(path)
            self.labels.append(self.class_to_index[cls_norm])
            used += 1
        print(f"[Dataset] used={used}, missing={missing}, classes={self.class_names}")
        if not self.image_paths:
            raise RuntimeError("No labeled images found after parsing annotations.")

    # ── public ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img_t = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        return img_t, label, str(path)


class _ValWrapper(Dataset):
    """Применяет val-трансформацию поверх уже загруженного датасета."""

    def __init__(self, base: CesmCsvDataset, indices: Iterable[int], transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.paths = [self.base.image_paths[i] for i in self.indices]
        self.labels = [self.base.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        orig_i = self.indices[i]
        path = self.base.image_paths[orig_i]
        label = self.base.labels[orig_i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img_t = self.transform(img)
        return img_t, label, str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Dataloaders factory
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_stats() -> Tuple[List[float], List[float]]:
    try:
        meta = getattr(ResNet18_Weights.IMAGENET1K_V1, "meta", {}) or {}
        mean = meta.get("mean", _IMAGENET_MEAN)
        std = meta.get("std", _IMAGENET_STD)
    except Exception:
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    return list(mean), list(std)


def build_dataloaders(
    images_dir: Path,
    annotations_csv: Path,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.2,
    class_names: Tuple[str, ...] = CesmCsvDataset.CLASS_NAMES,
) -> Tuple[DataLoader, DataLoader]:
    """Возвращает (train_loader, val_loader)."""
    mean, std = get_imagenet_stats()

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_ds = CesmCsvDataset(images_dir, annotations_csv, class_names=class_names, transform=train_tf)
    labels = np.array(full_ds.labels)
    idx = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, random_state=SEED, stratify=labels)

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = _ValWrapper(full_ds, val_idx.tolist(), val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
