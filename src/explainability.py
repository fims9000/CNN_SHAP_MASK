"""
explainability.py — SHAP + Grad-CAM utils (общие для обоих пайплайнов).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    show_cam_on_image = None

try:
    import shap as _shap
except Exception:
    _shap = None

# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_rgb(t: np.ndarray) -> np.ndarray:
    """CHW float tensor → HWC float [0,1]."""
    img = np.transpose(t, (1, 2, 0))
    img = img * _STD[None, None, :] + _MEAN[None, None, :]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def robust_normalize(arr: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(arr, low_q), np.percentile(arr, high_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn, mx = arr.min(), arr.ptp()
        return (arr - mn) / (mx + 1e-8)
    return (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

def run_gradcam(model: nn.Module, images: torch.Tensor, device: torch.device) -> List[np.ndarray]:
    if GradCAM is None:
        raise RuntimeError("pytorch-grad-cam not installed")
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    with torch.enable_grad():
        grayscale = cam(input_tensor=images.to(device), targets=None)
    results = []
    for i in range(images.size(0)):
        rgb = tensor_to_rgb(images[i].cpu().numpy())
        overlay = show_cam_on_image(rgb, grayscale[i], use_rgb=True)
        results.append(overlay)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SHAP
# ──────────────────────────────────────────────────────────────────────────────

def _shap_wrapper(model: nn.Module, device: torch.device):
    """Оборачивает модель для SHAP: numpy HWC [0,1] → softmax probabilities."""
    def f(x_np: np.ndarray) -> np.ndarray:
        imgs = torch.from_numpy(np.transpose(x_np, (0, 3, 1, 2))).float()
        mean = torch.tensor(_MEAN)[None, :, None, None]
        std  = torch.tensor(_STD )[None, :, None, None]
        imgs = (imgs - mean) / std
        with torch.no_grad():
            return torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
    return f


def _reduce_to_map(values_i: np.ndarray, num_classes: int, pred_class: int,
                   low_q: float = 2.0, high_q: float = 98.0, gamma: float = 0.7) -> np.ndarray:
    """Превращает shap_values[i] (HWC×C или list) в 2-D importance map [0,1]."""
    # extract class slice
    if isinstance(values_i, list) or (hasattr(values_i, "dtype") and values_i.dtype == object):
        vals = values_i[pred_class]
    else:
        shape = list(values_i.shape)
        if num_classes in shape:
            vals = np.take(values_i, pred_class, axis=shape.index(num_classes))
        else:
            vals = values_i[..., pred_class]

    # aggregate channels → 2-D
    vals = np.squeeze(vals)
    if vals.ndim == 2:
        agg = np.abs(vals)
    elif vals.ndim == 3:
        if vals.shape[-1] != 3:
            dims = np.array(vals.shape)
            if np.any(dims == 3):
                ch = int(np.where(dims == 3)[0][0])
                vals = np.moveaxis(vals, ch, -1)
            else:
                vals = np.stack([vals, vals, vals], -1)
        agg = np.abs(vals).mean(axis=2)
    else:
        agg = np.abs(vals)

    agg = robust_normalize(agg, low_q, high_q)
    if gamma != 1.0:
        agg = np.power(np.clip(agg, 0, 1), gamma)
    return agg.astype(np.float32)


def compute_shap_heatmaps(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    class_names: Tuple[str, ...],
    masker_type: str = "inpaint",
    max_evals: int = 500,
    batch_size: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Возвращает:
      shap_overlays  — цветные наложения SHAP на RGB (uint8)
      shap_maps      — сырые importance maps float32 [0,1] для SAM
    """
    if _shap is None:
        raise RuntimeError("shap not installed")
    model.eval()

    data_hwcs = [tensor_to_rgb(images[i].cpu().numpy()) for i in range(images.size(0))]
    data_arr = np.stack(data_hwcs, axis=0)

    f = _shap_wrapper(model, device)
    hwc_shape = data_arr[0].shape
    masker = (_shap.maskers.Image("blur(50,50)", hwc_shape)
              if masker_type == "blur"
              else _shap.maskers.Image("inpaint_telea", hwc_shape))
    explainer = _shap.Explainer(f, masker, output_names=list(class_names))
    shap_values = explainer(data_arr, max_evals=max_evals,
                             batch_size=min(batch_size, images.size(0)))
    values = shap_values.values

    # predicted classes
    with torch.no_grad():
        imgs_t = torch.from_numpy(np.transpose(data_arr, (0, 3, 1, 2))).float()
        mean = torch.tensor(_MEAN)[None, :, None, None]
        std  = torch.tensor(_STD )[None, :, None, None]
        preds = model(((imgs_t - mean) / std).to(device)).argmax(dim=1).cpu().numpy().tolist()

    nc = len(class_names)
    overlays, maps = [], []
    cmap = plt.get_cmap("jet")
    for i in range(images.size(0)):
        imp_map = _reduce_to_map(values[i], nc, preds[i])
        maps.append(imp_map)
        colored = cmap(imp_map)[..., :3]
        overlay = np.clip(0.5 * data_arr[i] + 0.5 * colored, 0, 1)
        overlays.append((overlay * 255).astype(np.uint8))
    return overlays, maps


# ──────────────────────────────────────────────────────────────────────────────
# SHAP → binary mask
# ──────────────────────────────────────────────────────────────────────────────

def shap_map_to_mask(shap_map: np.ndarray) -> np.ndarray:
    """
    Превращает importance map [0,1] в бинарную маску uint8 через:
    1. Gaussian blur
    2. Адаптивный порог (0.5 → 0.4 → 0.3 → 0.2 → percentile-90)
    3. Морфологическое закрытие
    4. Convex hull наибольшего контура
    5. Дилатация
    """
    sh = shap_map.squeeze().astype(np.float32)
    if sh.ndim != 2:
        sh = sh.mean(-1)
    H, W = sh.shape
    sh_b = cv2.GaussianBlur(sh, (0, 0), 1.25)

    mask = None
    for t in [0.5, 0.4, 0.3, 0.2]:
        m = (sh_b >= t).astype(np.uint8)
        if m.sum() >= max(1, 0.001 * H * W):
            mask = m; break
    if mask is None:
        mask = (sh_b >= float(np.percentile(sh_b, 90))).astype(np.uint8)

    k = max(3, int(min(H, W) * 0.02)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

    cnts, _ = cv2.findContours((closed * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        full = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(full, hull, 1)
    else:
        full = closed

    k2 = max(3, int(min(H, W) * 0.05)) | 1
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    return (cv2.dilate(full, ker2) > 0).astype(np.uint8)


def shap_mask_overlay(
    image_rgb: np.ndarray,
    shap_map: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
    border: int = 2,
) -> np.ndarray:
    """Наносит SHAP-маску поверх RGB изображения."""
    img = np.clip(image_rgb, 0, 255).astype(np.uint8)
    mask = shap_map_to_mask(shap_map)
    overlay = img.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + c * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if border > 0:
        cnts, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, color, thickness=border)
    return overlay
