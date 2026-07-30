"""
viz.py — функции сохранения сравнительных сеток визуализаций.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(ax, img, title):
    if img is not None:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")


def save_comparison_grid(
    output_path: Path,
    originals: List[np.ndarray],
    gradcams: List[Optional[np.ndarray]],
    shaps: List[Optional[np.ndarray]],
    sams: List[Optional[np.ndarray]],
    titles: List[str],
    shap_masks: Optional[List[Optional[np.ndarray]]] = None,
) -> None:
    """
    Сохраняет сетку: Original | Grad-CAM | SHAP | SAM [| SHAP Mask].
    """
    rows = len(originals)
    cols = 4 + (1 if shap_masks is not None else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, max(1, rows) * 4))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM", "SHAP", "SAM"]
    if shap_masks is not None:
        col_titles.append("SHAP Mask")

    for i in range(rows):
        _show(axes[i, 0], originals[i], f"{col_titles[0]}\n{titles[i]}")
        _show(axes[i, 1], gradcams[i] if gradcams else None, col_titles[1])
        _show(axes[i, 2], shaps[i] if shaps else None, col_titles[2])
        _show(axes[i, 3], sams[i] if sams else None, col_titles[3])
        if shap_masks is not None:
            _show(axes[i, 4], shap_masks[i], col_titles[4])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved → {output_path}")


def save_pipeline_grid(
    output_path: Path,
    original: np.ndarray,
    panels: List[Optional[np.ndarray]],
    panel_titles: List[str],
) -> None:
    """Универсальная сетка для одного изображения и произвольного набора панелей."""
    cols = 1 + len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
    _show(axes[0], original, "Original")
    for j, (img, title) in enumerate(zip(panels, panel_titles)):
        _show(axes[j + 1], img, title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Viz] Saved → {output_path}")
