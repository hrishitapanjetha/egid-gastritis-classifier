# EGID Gastritis Classifier

A Gradio app for multi-label gastritis classification from endoscopic images, built on the [EGID dataset](https://doi.org/10.1038/s41597-026-07666-6) (Endoscopic Gastritis Image Dataset).

Given one or more endoscopy images of a patient, the app predicts four clinical dimensions simultaneously:
- **H. pylori infection status** (positive / negative)
- **Presence of atrophy** (atrophic / non-atrophic)
- **Distribution of atrophy** (normal / antrum-predominant / corpus-predominant / antrum & corpus)
- **Gastritis type** (normal / Type A — autoimmune / Type B — H. pylori-associated)

## How it works

- **Ensemble of 5 ViT-B/16 models**, one per patient-level cross-validation fold. Predictions are averaged across every model and every image provided (matching how the underlying models were validated — patient-level softmax averaging, not judged from a single frame). ResNet50, DenseNet121, and EfficientNet-B0 were also trained and benchmarked during development; ViT-B/16 was the strongest performer on accuracy, AUROC, and AUPRC in cross-validation, so the deployed app uses it exclusively.
- **Class-weighted training**: inverse-frequency loss weighting was used during training to address label imbalance across the four tasks.

## Running locally

```bash
pip install -r requirements.txt
python3 app.py
```

Model checkpoints (`checkpoints/*.pt`) are not committed to this repository (see `.gitignore`) — they're hosted on the [Hugging Face Hub](https://huggingface.co/Hrishita-P/egid-gastritis-checkpoints) and downloaded automatically at startup, cached locally after the first run. The deployed version of this app runs on Hugging Face Spaces.

## Disclaimer

This is a research/educational prototype built on a public benchmark dataset. It is **not a diagnostic device** and has not been clinically validated. Predictions should never be used to inform real patient care or replace professional medical judgement.
