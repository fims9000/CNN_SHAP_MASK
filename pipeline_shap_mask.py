"""
pipeline_shap_mask.py
═════════════════════
Пайплайн  CNN → SHAP → MASK  (последовательный режим).

Схема:
  ResNet-18 ──► SHAP importance map ──► binary SHAP-mask
                                             │
                                             ├─► наложение маски на исходник  (SHAP Mask)
                                             │
                                             └─► SAM Predictor (points + bbox из маски)
                                                     │
                                                     └─► уточнённая сегментация (SAM ← SHAP mask)

Финальный вывод для каждого снимка:
  Original | SHAP heatmap | SHAP Mask | SAM (guided by SHAP mask)

Запуск:
  python pipeline_shap_mask.py --vis-samples 6 --epochs 10 --device auto
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import build_dataloaders, CesmCsvDataset
from model import build_resnet18, load_best_checkpoint, train, evaluate_and_save
from explainability import (
    run_gradcam, compute_shap_heatmaps,
    shap_map_to_mask, shap_mask_overlay,
    tensor_to_rgb,
)
from sam_utils import run_sam_guided, mask_overlay, ensure_sam_weights
from viz import save_comparison_grid, save_pipeline_grid

# ──────────────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent
_DATA        = _ROOT / "data"
_DATASET     = _DATA / "Low energy images of CDD-CESM"
_CSV         = _DATA / "Radiology-manual-annotations.csv"
_WEIGHTS     = _ROOT / "weights"
_OUTPUTS     = _ROOT / "outputs" / "shap_mask"
_PRETRAINED  = _ROOT / "checkpoints" / "resnet18_cesm"  # готовые веса

CLASS_NAMES = ("Normal", "Benign", "Malignant")


# ──────────────────────────────────────────────────────────────────────────────

def _pick_samples(loader, k: int):
    images, labels, paths = [], [], []
    for x, y, p in loader:
        for i in range(x.size(0)):
            images.append(x[i]); labels.append(int(y[i])); paths.append(Path(p[i]))
            if len(images) >= k:
                break
        if len(images) >= k:
            break
    if not images:
        raise RuntimeError("No samples in val loader")
    return torch.stack(images), labels, paths


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN → SHAP → MASK → SAM (последовательный пайплайн, CDD-CESM)"
    )
    parser.add_argument("--images-dir",       type=str, default=str(_DATASET))
    parser.add_argument("--annotations-csv",  type=str, default=str(_CSV))
    parser.add_argument("--weights-dir",      type=str, default=str(_WEIGHTS))
    parser.add_argument("--outputs-dir",      type=str, default=str(_OUTPUTS))
    parser.add_argument("--epochs",           type=int, default=10)
    parser.add_argument("--batch-size",       type=int, default=32)
    parser.add_argument("--num-workers",      type=int, default=2)
    parser.add_argument("--vis-samples",      type=int, default=6)
    parser.add_argument("--device",           type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-train",       action="store_true")
    parser.add_argument("--skip-shap",        action="store_true")
    parser.add_argument("--shap-max-evals",   type=int, default=500)
    parser.add_argument("--shap-batch",       type=int, default=8)
    parser.add_argument("--shap-masker",      type=str, default="inpaint",
                        choices=["inpaint", "blur"])
    parser.add_argument("--mask-alpha",       type=float, default=0.35,
                        help="Прозрачность наложения SHAP-маски (0–1)")
    parser.add_argument("--mask-color",       type=str, default="0,255,0",
                        help="Цвет маски RGB через запятую, напр. '0,255,0'")
    parser.add_argument("--sam-box-threshold", type=float, default=0.6)
    parser.add_argument("--sam-num-points",    type=int, default=10)
    parser.add_argument("--patience",         type=int, default=5)
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    mask_color = tuple(int(c) for c in args.mask_color.split(","))

    images_dir  = Path(args.images_dir)
    annotations = Path(args.annotations_csv)
    weights_dir = Path(args.weights_dir)
    out_dir     = Path(args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    train_loader, val_loader = build_dataloaders(
        images_dir, annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_names=CLASS_NAMES,
    )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("[2/5] Building ResNet-18 …")
    model = build_resnet18(num_classes=len(CLASS_NAMES))

    if args.skip_train:
        # Сначала ищем в outputs/, затем в checkpoints/ (предобученные веса)
        loaded = load_best_checkpoint(model, out_dir, device)
        if loaded is None:
            loaded = load_best_checkpoint(model, _PRETRAINED, device)
        if loaded is None:
            print("  No checkpoint found — training from scratch.")
            args.skip_train = False
        else:
            model = loaded

    if not args.skip_train:
        print("[2/5] Training …")
        train(model, train_loader, val_loader, device, out_dir,
              epochs=args.epochs, patience=args.patience, class_names=CLASS_NAMES)
        loaded = load_best_checkpoint(model, out_dir, device)
        if loaded is not None:
            model = loaded

    model.to(device).eval()

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("[3/5] Evaluating …")
    evaluate_and_save(model, val_loader, device, CLASS_NAMES, out_dir, tag="val")

    # ── 4. Pick samples ───────────────────────────────────────────────────────
    print(f"[4/5] Selecting {args.vis_samples} samples …")
    vis_images, vis_labels, vis_paths = _pick_samples(val_loader, args.vis_samples)

    originals = [(tensor_to_rgb(vis_images[i].cpu().numpy()) * 255).astype(np.uint8)
                 for i in range(vis_images.size(0))]
    titles    = [CLASS_NAMES[l] for l in vis_labels]

    # ── 5. Sequential: SHAP → mask → SAM ─────────────────────────────────────
    print("[5/5] Running SHAP → MASK → SAM …")

    shap_overlays: list = [None] * vis_images.size(0)
    shap_masks_img: list = [None] * vis_images.size(0)
    shap_raw_maps: list = []
    sam_guided_imgs: list = [None] * vis_images.size(0)

    if not args.skip_shap:
        try:
            shap_overlays, shap_raw_maps = compute_shap_heatmaps(
                model, vis_images, device, CLASS_NAMES,
                masker_type=args.shap_masker,
                max_evals=args.shap_max_evals,
                batch_size=args.shap_batch,
            )
            # SHAP map → binary mask → overlay
            for i in range(vis_images.size(0)):
                shap_masks_img[i] = shap_mask_overlay(
                    originals[i], shap_raw_maps[i],
                    color=mask_color, alpha=args.mask_alpha,
                )
        except Exception as e:
            print(f"  [WARN] SHAP failed: {e}")

    # SAM guided by SHAP mask
    if shap_raw_maps:
        try:
            sam_guided_imgs = run_sam_guided(
                originals, shap_raw_maps, weights_dir,
                device=device.type,
                box_threshold=args.sam_box_threshold,
                num_points=args.sam_num_points,
            )
        except Exception as e:
            print(f"  [WARN] SAM guided failed: {e}")

    # ── Save grids ────────────────────────────────────────────────────────────
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Общая сетка: Original | SHAP | SHAP Mask | SAM (guided)
    save_comparison_grid(
        vis_dir / "comparisons_shap_mask.png",
        originals,
        [None] * vis_images.size(0),   # нет Grad-CAM колонки в этом пайплайне
        shap_overlays,
        sam_guided_imgs,
        titles,
        shap_masks=shap_masks_img,
    )

    # Подробная сетка на каждый снимок
    for i in range(vis_images.size(0)):
        save_pipeline_grid(
            vis_dir / f"sample_{i:02d}_shap_mask.png",
            originals[i],
            [shap_overlays[i], shap_masks_img[i], sam_guided_imgs[i]],
            ["SHAP heatmap", "SHAP Mask", "SAM ← SHAP Mask"],
        )

    print(f"\n[DONE] Results → {out_dir}")


if __name__ == "__main__":
    main()
