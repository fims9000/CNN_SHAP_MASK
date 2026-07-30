"""
sam_utils.py — загрузка SAM ViT-B и два режима сегментации:
  • automatic  — SamAutomaticMaskGenerator (без подсказок)
  • guided     — SamPredictor с points+box из SHAP-карты
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Weights
# ──────────────────────────────────────────────────────────────────────────────

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_VIT_B_FILENAME = "sam_vit_b_01ec64.pth"


def ensure_sam_weights(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / SAM_VIT_B_FILENAME
    if ckpt.exists():
        return ckpt
    print(f"[SAM] Downloading ViT-B weights → {ckpt}")
    with requests.get(SAM_VIT_B_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ckpt, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=SAM_VIT_B_FILENAME) as pbar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helper
# ──────────────────────────────────────────────────────────────────────────────

def mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask.astype(np.float32)[..., None]
    overlay = overlay * (1 - alpha * m) + c * (alpha * m)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Automatic SAM (без подсказок)
# ──────────────────────────────────────────────────────────────────────────────

def run_sam_automatic(
    image_paths: List[Path],
    weights_dir: Path,
    device: str = "cpu",
) -> List[Optional[np.ndarray]]:
    """Сегментирует изображения автоматически (берёт маску с наибольшей площадью)."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("[SAM] segment_anything not installed")
        return [None] * len(image_paths)

    ckpt = ensure_sam_weights(weights_dir)
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device)
    gen = SamAutomaticMaskGenerator(sam)

    results = []
    for path in image_paths:
        with Image.open(path) as img:
            np_img = np.array(img.convert("RGB"))
        masks = gen.generate(np_img)
        if not masks:
            results.append(None)
            continue
        best = max(masks, key=lambda m: m.get("area", 0))
        results.append(mask_overlay(np_img, best["segmentation"].astype(np.uint8)))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Guided SAM (SHAP → points + box → SamPredictor)
# ──────────────────────────────────────────────────────────────────────────────

def _shap_to_prompts(
    shap_map: np.ndarray,
    box_threshold: float = 0.6,
    num_points: int = 10,
):
    H, W = shap_map.shape
    # bounding box from threshold
    coords = np.argwhere(shap_map > box_threshold)
    box = None
    if coords.size > 0:
        y0, x0 = coords[:, 0].min(), coords[:, 1].min()
        y1, x1 = coords[:, 0].max(), coords[:, 1].max()
        pad = 2
        box = np.array([max(0, x0-pad), max(0, y0-pad),
                         min(W-1, x1+pad), min(H-1, y1+pad)], dtype=np.float32)
    # top-N foreground points
    flat = shap_map.reshape(-1)
    k = min(num_points, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    ys, xs = idx // W, idx % W
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    labels = np.ones(points.shape[0], dtype=np.int32)
    return points, labels, box


def run_sam_guided(
    images_rgb: List[np.ndarray],          # uint8 HWC
    shap_maps: List[np.ndarray],           # float32 [0,1]
    weights_dir: Path,
    device: str = "cpu",
    box_threshold: float = 0.6,
    num_points: int = 10,
) -> List[Optional[np.ndarray]]:
    """SAM с подсказками из SHAP (points + bounding box)."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("[SAM] segment_anything not installed")
        return [None] * len(images_rgb)

    ckpt = ensure_sam_weights(weights_dir)
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device)
    predictor = SamPredictor(sam)

    results = []
    for np_img, shap_map in zip(images_rgb, shap_maps):
        predictor.set_image(np_img)
        points, labels, box = _shap_to_prompts(shap_map, box_threshold, num_points)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )
            best_i = int(np.argmax(scores))
            colored = mask_overlay(np_img, masks[best_i].astype(np.uint8))
            results.append(colored)
        except Exception as e:
            print(f"[SAM] guided failed: {e}")
            results.append(None)
    return results
