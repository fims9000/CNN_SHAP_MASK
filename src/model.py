"""
model.py — ResNet-18 builder + training loop для CDD-CESM.

Характеристики:
  • Backbone: ResNet-18, pretrained ImageNet-1K (512-d features)
  • Head: Linear(512 → num_classes)
  • Optimizer: AdamW + CosineAnnealingLR
  • Loss: CrossEntropyLoss
  • Checkpoints: best-accuracy / best-macro-F1
  • Early stopping по val_loss
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── factory ───────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 с заменённой головой; inplace-ReLU выключены для хуков."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for m in model.modules():
        if isinstance(m, nn.ReLU) and getattr(m, "inplace", False):
            m.inplace = False
    return model


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    for m in model.modules():
        if isinstance(m, nn.ReLU) and getattr(m, "inplace", False):
            m.inplace = False
    print(f"[Model] Loaded: {path}")
    return model


def load_best_checkpoint(
    model: nn.Module, outputs_dir: Path, device: torch.device, prefer: str = "acc"
) -> Optional[nn.Module]:
    order = (["best_resnet18_acc.pth", "best_resnet18_f1.pth"]
             if prefer == "acc"
             else ["best_resnet18_f1.pth", "best_resnet18_acc.pth"])
    for name in order:
        p = outputs_dir / name
        if p.exists():
            return load_checkpoint(model, p, device)
    return None


# ── metrics ───────────────────────────────────────────────────────────────────

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).sum().item() / max(1, targets.numel())


@torch.no_grad()
def gather_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_l, all_t, all_p = [], [], []
    for inputs, targets, paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)
        all_l.append(model(inputs).cpu())
        all_t.append(targets.cpu())
        all_p.extend(paths)
    return torch.cat(all_l), torch.cat(all_t), all_p


def evaluate_and_save(
    model, loader, device, class_names, out_dir: Path, tag: str
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    logits, targets, paths = gather_logits(model, loader, device)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)
    y_true = targets.numpy()
    labels = list(range(len(class_names)))

    macro_f1 = float(f1_score(y_true, preds, average="macro", labels=labels))
    micro_f1 = float(f1_score(y_true, preds, average="micro", labels=labels))
    cm = confusion_matrix(y_true, preds, labels=labels)
    report = classification_report(y_true, preds, labels=labels,
                                   target_names=list(class_names), digits=4, zero_division=0)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(list(class_names), rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(list(class_names))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=150)
    plt.close(fig)

    (out_dir / f"classification_report_{tag}.txt").write_text(report, encoding="utf-8")
    with open(out_dir / f"predictions_{tag}.json", "w") as f:
        json.dump([{"path": paths[i], "true": int(y_true[i]), "pred": int(preds[i]),
                    "probs": probs[i].tolist()} for i in range(len(paths))], f, indent=2)
    metrics = {"macro_f1": macro_f1, "micro_f1": micro_f1, "n": len(paths)}
    with open(out_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Eval/{tag}] macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")
    return metrics


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    min_delta: float = 0.0,
    class_names: Tuple[str, ...] = ("Normal", "Benign", "Malignant"),
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc = best_f1 = 0.0
    best_vl = float("inf")
    worse = 0
    history: List[Dict] = []

    for epoch in range(epochs):
        # train
        model.train()
        tl = ta = n = 0.0
        for inputs, targets, _ in tqdm(train_loader, desc=f"E{epoch+1} train", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.as_tensor(targets, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward(); optimizer.step()
            bs = targets.size(0)
            tl += loss.item() * bs; ta += accuracy_from_logits(logits.detach(), targets) * bs; n += bs
        scheduler.step()

        # val
        model.eval()
        vl = va = nv = 0.0
        lgs, tgs = [], []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc=f"E{epoch+1} val", leave=False):
                inputs = inputs.to(device, non_blocking=True)
                targets = torch.as_tensor(targets, device=device)
                logits = model(inputs)
                bs = targets.size(0)
                vl += criterion(logits, targets).item() * bs
                va += accuracy_from_logits(logits, targets) * bs; nv += bs
                lgs.append(logits.cpu()); tgs.append(targets.cpu())

        avg_tl, avg_vl = tl / max(1, n), vl / max(1, nv)
        avg_ta, avg_va = ta / max(1, n), va / max(1, nv)
        pnp = torch.softmax(torch.cat(lgs), 1).argmax(1).numpy()
        tnp = torch.cat(tgs).numpy()
        mf1 = float(f1_score(tnp, pnp, average="macro", labels=list(range(len(class_names)))))

        print(f"Epoch {epoch+1}/{epochs}: tl={avg_tl:.4f} vl={avg_vl:.4f} ta={avg_ta:.3f} va={avg_va:.3f} f1={mf1:.3f}")
        history.append({"epoch": epoch+1, "train_loss": avg_tl, "val_loss": avg_vl,
                         "train_acc": avg_ta, "val_acc": avg_va, "macro_f1": mf1})

        if avg_va > best_acc:
            best_acc = avg_va
            torch.save({"model_state": model.state_dict(), "val_acc": best_acc},
                        out_dir / "best_resnet18_acc.pth")
            print(f"  ↑ saved best-acc={best_acc:.3f}")
        if mf1 > best_f1:
            best_f1 = mf1
            torch.save({"model_state": model.state_dict(), "macro_f1": best_f1},
                        out_dir / "best_resnet18_f1.pth")
            print(f"  ↑ saved best-f1={best_f1:.3f}")

        if avg_vl < best_vl - min_delta:
            best_vl = avg_vl; worse = 0
        else:
            worse += 1
            if worse >= patience:
                print("Early stopping.")
                break

    summary = {"best_val_acc": best_acc, "best_val_macro_f1": best_f1, "history": history}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
