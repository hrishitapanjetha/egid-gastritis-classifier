from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from config import TASKS
from models import MultiTaskModel
from transforms import eval_transform

APP_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = APP_DIR / "checkpoints"
SAMPLES_DIR = APP_DIR / "samples"
BACKBONES = ["vit_b_16"]
FOLDS = [0, 1, 2, 3, 4]

# Model weights are hosted on the Hugging Face Hub rather than committed to
# this git repo, and downloaded (once, then cached) at startup.
HF_CHECKPOINTS_REPO = "Hrishita-P/egid-gastritis-checkpoints"

CLASS_LABELS = {
    "H_pylori_infection_status": ["Negative", "Positive"],
    "Presence_of_atrophy": ["Non-atrophic", "Atrophic"],
    "Distribution_of_atrophy": ["Normal", "Antrum-predominant", "Corpus-predominant", "Antrum & Corpus"],
    "Gastritis_type": ["Normal", "Type A (Autoimmune)", "Type B (H. pylori)"],
}

TASK_TITLES = {
    "H_pylori_infection_status": "H. pylori Infection Status",
    "Presence_of_atrophy": "Presence of Atrophy",
    "Distribution_of_atrophy": "Distribution of Atrophy",
    "Gastritis_type": "Gastritis Type",
}


def get_device():
    # Force CPU: HF Spaces ZeroGPU only grants CUDA access inside functions
    # wrapped in @spaces.GPU, which this app doesn't use.
    return torch.device("cpu")


def _resolve_checkpoint(backbone: str, fold: int) -> Path:
    """Prefer a local checkpoints/ copy (e.g. during local dev) if present;
    otherwise download from the Hugging Face model repo, cached locally by
    huggingface_hub after the first download."""
    filename = f"{backbone}_fold{fold}_best.pt"
    local_path = CHECKPOINTS_DIR / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(repo_id=HF_CHECKPOINTS_REPO, filename=filename))


@lru_cache(maxsize=1)
def load_ensemble():
    device = get_device()
    models = []
    for backbone in BACKBONES:
        for fold in FOLDS:
            ckpt_path = _resolve_checkpoint(backbone, fold)
            model = MultiTaskModel(backbone)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append((f"{backbone} (fold {fold})", model))
    return models, device


@torch.no_grad()
def predict(images: list[Image.Image]):
    """Patient-level prediction: average softmax probabilities across every
    image the caller provides AND across every ensemble model, then argmax.
    This mirrors exactly how patient-level metrics were computed during
    k-fold validation (average image-level softmax across a patient's full
    image set) rather than judging from a single image."""
    models, device = load_ensemble()
    xs = torch.stack([eval_transform(img.convert("RGB")) for img in images]).to(device)

    task_probs_sum = {task: None for task in TASKS}
    per_prediction = {task: [] for task in TASKS}  # every (model, image) vote

    for _name, model in models:
        outputs = model(xs)  # batched over all provided images
        for task in TASKS:
            probs = torch.softmax(outputs[task], dim=1).cpu().numpy()  # (n_images, n_classes)
            if task_probs_sum[task] is None:
                task_probs_sum[task] = probs.sum(axis=0)
            else:
                task_probs_sum[task] += probs.sum(axis=0)
            per_prediction[task].extend(probs.argmax(axis=1).tolist())

    n_models = len(models)
    n_images = len(images)
    total_votes = n_models * n_images
    results = {}
    for task in TASKS:
        avg_probs = task_probs_sum[task] / total_votes
        pred_idx = int(avg_probs.argmax())
        results[task] = {
            "predicted_class": CLASS_LABELS[task][pred_idx],
            "predicted_idx": pred_idx,
            "confidence": float(avg_probs[pred_idx]),
            "all_probs": {CLASS_LABELS[task][i]: float(p) for i, p in enumerate(avg_probs)},
            "model_agreement": sum(1 for p in per_prediction[task] if p == pred_idx) / total_votes,
        }
    return results, n_models, n_images
