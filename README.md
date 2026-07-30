# CNN_SHAP_MASK

**Статус:**
⚠️ Work in progress

## Что работает
- Обучение ResNet-18 на датасете CDD-CESM (Normal / Benign / Malignant)
- Grad-CAM визуализации
- SHAP heatmaps (Image Explainer, inpaint / blur masker)
- SHAP importance map → бинарная маска (морфология + convex hull)
- SAM ViT-B: автоматический режим (`SamAutomaticMaskGenerator`)
- SAM ViT-B: направляемый режим по SHAP-подсказкам (`SamPredictor` + points & bbox)

## Что не работает / в разработке
- Многомасштабный SHAP (TTA + multi-scale)
- Метрики качества сегментации (IoU vs. ручной разметки)
- Экспорт масок в DICOM-формат

---

## Архитектура

```
ResNet-18 (ImageNet pretrained)
 └─ fc: Linear(512 → 3)          # Normal / Benign / Malignant

Пайплайн A — CNN → SHAP + SAM (параллельный):
  CNN ──┬── SHAP overlay
        ├── SAM automatic
        └── SAM guided by SHAP points+bbox

Пайплайн B — CNN → SHAP → MASK (последовательный):
  CNN ──► SHAP map ──► binary mask ──► SAM Predictor (mask-guided)
```

---

## Структура проекта

```
CNN_SHAP_MASK/
├── pipeline_shap_sam.py      # Пайплайн A: CNN → SHAP + SAM
├── pipeline_shap_mask.py     # Пайплайн B: CNN → SHAP → MASK → SAM
├── src/
│   ├── dataset.py            # CesmCsvDataset + build_dataloaders
│   ├── model.py              # build_resnet18 + train + evaluate
│   ├── explainability.py     # Grad-CAM, SHAP, shap_map_to_mask
│   ├── sam_utils.py          # SAM ViT-B: automatic + guided
│   └── viz.py                # Функции сохранения сеток
├── weights/                  # SAM ViT-B веса (sam_vit_b_01ec64.pth)
├── outputs/                  # Артефакты (в .gitignore)
│   ├── shap_sam/             # best_resnet18_acc.pth, visualizations/…
│   └── shap_mask/            # best_resnet18_acc.pth, visualizations/…
├── notebooks/                # Jupyter-ноутбуки для экспериментов
├── data/                     # Символические ссылки / скрипты загрузки (в .gitignore)
├── requirements.txt
└── .gitignore
```

> **Данные** (Low energy images of CDD-CESM) и **веса SAM** находятся вне репозитория.  
> Укажите реальный путь через флаги CLI или измените константы `_DATASET` / `_WEIGHTS`  
> в заголовках пайплайнов.

---

## Датасет

**CDD-CESM** — маммографический датасет с CE-субтракционными снимками.

| Параметр | Значение |
|---|---|
| Изображения | Low energy (LE) CESM (~1 000 jpg) |
| Классы | Normal, Benign, Malignant |
| Разметка | `Radiology-manual-annotations.csv` |



---


### Использование готовых весов (без переобучения)

```bash
python pipeline_shap_sam.py  --skip-train --vis-samples 8 --device auto
python pipeline_shap_mask.py --skip-train --vis-samples 8 --device auto
```

Пайплайн автоматически найдёт `checkpoints/resnet18_cesm/best_resnet18_acc.pth`
и загрузит веса без дополнительных действий.

---

## Установка окружения

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

> Если CUDA не нужна, установите CPU-версию PyTorch первой:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

## Запуск

### Пайплайн A — CNN → SHAP + SAM

```bash
python pipeline_shap_sam.py \
    --epochs 10 \
    --batch-size 32 \
    --vis-samples 8 \
    --device auto \
    --shap-max-evals 500
```

### Пайплайн B — CNN → SHAP → MASK

```bash
python pipeline_shap_mask.py \
    --epochs 10 \
    --batch-size 32 \
    --vis-samples 8 \
    --device auto \
    --mask-alpha 0.4 \
    --mask-color "0,255,0"
```

### Быстрый прогон без обучения (CPU, нет SHAP)

```bash
python pipeline_shap_sam.py \
    --skip-train \
    --skip-shap \
    --vis-samples 4 \
    --device cpu
```

---

## Ключевые аргументы CLI

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--epochs` | Эпохи обучения | 10 |
| `--batch-size` | Размер батча | 32 |
| `--num-workers` | Потоки загрузчика (0–2 на Windows) | 2 |
| `--vis-samples` | Число снимков для визуализации | 6 |
| `--device` | `auto` / `cpu` / `cuda` | auto |
| `--skip-train` | Пропустить обучение | — |
| `--skip-shap` | Пропустить SHAP | — |
| `--shap-max-evals` | Макс. eval SHAP | 500 |
| `--shap-masker` | `inpaint` / `blur` | inpaint |
| `--sam-box-threshold` | Порог bbox из SHAP | 0.6 |
| `--sam-num-points` | Точки-подсказки SAM | 10 |
| `--patience` | Patience early stopping | 5 |

---

## Веса SAM

SAM ViT-B веса (`sam_vit_b_01ec64.pth`, ~357 МБ) скачиваются автоматически  
при первом запуске в папку `weights/`.  


## Артефакты (outputs/)

```
outputs/shap_sam/
├── best_resnet18_acc.pth          # лучшая модель по accuracy
├── best_resnet18_f1.pth           # лучшая модель по macro-F1
├── metrics.json                   # история обучения
├── metrics_val.json               # итоговые метрики
├── confusion_matrix_val.png
├── classification_report_val.txt
├── predictions_val.json
└── visualizations/
    ├── comparisons_shap_sam.png   # сводная сетка
    └── sample_NN_shap_sam.png     # по каждому снимку
```

---

## Лицензия

Для исследовательских и демонстрационных целей.  
Датасет CDD-CESM — см. [оригинальную публикацию](https://www.cancerimagingarchive.net/collection/cdd-cesm/).
