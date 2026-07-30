"""
pipeline_shap_sam.py
════════════════════
Пайплайн  CNN → SHAP + SAM  (параллельный режим).

Схема:
  ResNet-18 ──┬── SHAP  ──────────────────────────────► наложение SHAP на исходник
              │
              └── SAM (automatic) ─────────────────────► автосегментация
                       │
                       └── SAM guided by SHAP ─────────► SHAP-направленная маска

Финальный вывод для каждого снимка:
  Original | Grad-CAM | SHAP | SAM (auto) | SAM (guided by SHAP)

Запуск:
  python pipeline_shap_sam.py --vis-samples 6 --epochs 10 --device auto
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Добавляем src/ в PYTHONPATH при прямом запуске
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import build_dataloaders, CesmCsvDataset
from model import build_resnet18, load_best_checkpoint, train, evaluate_and_save
from explainability import (
    run_gradcam, compute_shap_heatmaps, tensor_to_rgb
)
from sam_utils import run_sam_automatic, run_sam_guided, mask_overlay
from viz import save_comparison_grid, save_pipeline_grid

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (относительные пути от папки проекта)
# ──────────────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent
_DATA        = _ROOT / "data"
_DATASET     = _DATA / "Low energy images of CDD-CESM"
_CSV         = _DATA / "Radiology-manual-annotations.csv"
_WEIGHTS     = _ROOT / "weights"
_OUTPUTS     = _ROOT / "outputs" / "shap_sam"
_PRETRAINED  = _ROOT / "checkpoints" / "resnet18_cesm"  # готовые веса

CLASS_NAMES = ("Normal", "Benign", "Malignant")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_samples(loader, k: int):
    images, labels, paths = [], [], []
    for x, y, p in loader:
        for i in range(x.size(0)):
            images.append(x[i])
            labels.append(int(y[i]))
            paths.append(Path(p[i]))
            if len(images) >= k:
                break
        if len(images) >= k:
            break
    if not images:
        raise RuntimeError("No samples in val loader")
    return torch.stack(images), labels, paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN → SHAP + SAM (параллельный пайплайн, CDD-CESM)"
    )
    parser.add_argument("--images-dir",      type=str, default=str(_DATASET))
    parser.add_argument("--annotations-csv", type=str, default=str(_CSV))
    parser.add_argument("--weights-dir",     type=str, default=str(_WEIGHTS))
    parser.add_argument("--outputs-dir",     type=str, default=str(_OUTPUTS))
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--batch-size",      type=int, default=32)
    parser.add_argument("--num-workers",     type=int, default=2)
    parser.add_argument("--vis-samples",     type=int, default=6)
    parser.add_argument("--device",          type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-train",      action="store_true")
    parser.add_argument("--skip-shap",       action="store_true")
    parser.add_argument("--shap-max-evals",  type=int, default=500)
    parser.add_argument("--shap-batch",      type=int, default=8)
    parser.add_argument("--shap-masker",     type=str, default="inpaint",
                        choices=["inpaint", "blur"])
    parser.add_argument("--sam-box-threshold", type=float, default=0.6)
    parser.add_argument("--sam-num-points",    type=int, default=10)
    parser.add_argument("--patience",        type=int, default=5)
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

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

    # ── 4. Pick visual samples ────────────────────────────────────────────────
    print(f"[4/5] Selecting {args.vis_samples} samples for visualisation …")
    vis_images, vis_labels, vis_paths = _pick_samples(val_loader, args.vis_samples)

    # ── 5. SHAP + SAM (параллельно) ───────────────────────────────────────────
    print("[5/5] Running Grad-CAM, SHAP, SAM …")

    # Grad-CAM
    try:
        gradcam_imgs = run_gradcam(model, vis_images, device)
    except Exception as e:
        print(f"  [WARN] Grad-CAM: {e}")
        gradcam_imgs = [None] * vis_images.size(0)

    # SHAP
    shap_overlays, shap_maps = [], []
    if not args.skip_shap:
        try:
            shap_overlays, shap_maps = compute_shap_heatmaps(
                model, vis_images, device, CLASS_NAMES,
                masker_type=args.shap_masker,
                max_evals=args.shap_max_evals,
                batch_size=args.shap_batch,
            )
        except Exception as e:
            print(f"  [WARN] SHAP: {e}")
            shap_overlays = [None] * vis_images.size(0)
    else:
        shap_overlays = [None] * vis_images.size(0)

    # SAM automatic (без подсказок)
    sam_auto = run_sam_automatic(vis_paths, weights_dir, device=device.type)

    # SAM guided by SHAP (если SHAP удался)
    if shap_maps:
        imgs_rgb = [(tensor_to_rgb(vis_images[i].cpu().numpy()) * 255).astype(np.uint8)
                    for i in range(vis_images.size(0))]
        sam_guided = run_sam_guided(
            imgs_rgb, shap_maps, weights_dir,
            device=device.type,
            box_threshold=args.sam_box_threshold,
            num_points=args.sam_num_points,
        )
    else:
        sam_guided = [None] * vis_images.size(0)

    # ── Save comparison grids ─────────────────────────────────────────────────
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    originals = [(tensor_to_rgb(vis_images[i].cpu().numpy()) * 255).astype(np.uint8)
                 for i in range(vis_images.size(0))]
    titles    = [CLASS_NAMES[l] for l in vis_labels]

    # Основная сетка: Original | GradCAM | SHAP | SAM-auto | SAM-guided
    save_comparison_grid(
        vis_dir / "comparisons_shap_sam.png",
        originals, gradcam_imgs, shap_overlays, sam_auto, titles,
    )

    # Отдельная сетка CNN→SHAP+SAM для каждого снимка
    for i in range(vis_images.size(0)):
        save_pipeline_grid(
            vis_dir / f"sample_{i:02d}_shap_sam.png",
            originals[i],
            [gradcam_imgs[i], shap_overlays[i], sam_auto[i], sam_guided[i]],
            ["Grad-CAM", "SHAP", "SAM (auto)", "SAM (guided)"],
        )

    print(f"\n[DONE] Results → {out_dir}")


if __name__ == "__main__":
    main()
