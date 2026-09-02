from pathlib import Path

import gradio as gr
from PIL import Image

from inference import predict, load_ensemble, TASK_TITLES, SAMPLES_DIR, BACKBONES, FOLDS

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except ImportError:
    def GPU_DECORATOR(fn):
        return fn


@GPU_DECORATOR
def _zerogpu_probe():
    """No-op: satisfies HF Spaces' ZeroGPU requirement that at least one
    @spaces.GPU function exist. Inference itself runs on CPU."""
    return None


TASK_ORDER = ["H_pylori_infection_status", "Presence_of_atrophy",
              "Distribution_of_atrophy", "Gastritis_type"]

PRIMARY = "#1E9A7C"
PRIMARY_DARK = "#147A63"
ACCENT = "#E8B84B"
BG_CARD = "#FFFFFF"
BG_PAGE = "#F4F7F6"
TEXT_MUTED = "#5B6B67"

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
:root {{ font-size: 18px; }}
.gradio-container, .gradio-container * {{
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif !important;
}}
html, body {{ background-color: {BG_PAGE} !important; }}
.gradio-container {{ background-color: {BG_PAGE} !important; }}
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container h5, .gradio-container p,
.gradio-container span, .gradio-container label, .gradio-container li,
.gradio-container i, .gradio-container em, .gradio-container strong,
.gradio-container b {{
    color: #1F2A27 !important;
}}
.hero {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(15,110,92,0.25);
}}
.hero h1 {{ margin: 0; font-size: 2.1rem; font-weight: 700; color: white !important; }}
.hero p {{ margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1.02rem; color: white !important; }}
.badge-row {{ margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
.hero .badge {{
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    color: white !important;
}}
.results-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}}
@media (max-width: 700px) {{ .results-grid {{ grid-template-columns: 1fr; }} }}
.task-card {{
    background: {BG_CARD};
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #E7ECEA;
}}
.task-title {{
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    font-weight: 600;
    margin-bottom: 0.3rem;
}}
.task-pred {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {PRIMARY_DARK};
    margin-bottom: 0.6rem;
}}
.conf-track {{
    background: #EAF1EF;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-bottom: 0.4rem;
}}
.conf-fill {{
    background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
    height: 100%;
    border-radius: 999px;
}}
.conf-label {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.7rem;
}}
.all-probs {{ font-size: 0.82rem; color: {TEXT_MUTED}; }}
.all-probs summary {{ cursor: pointer; color: {PRIMARY_DARK}; font-weight: 600; }}
.prob-row {{ display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; }}
.prob-row .prob-name {{ width: 40%; flex-shrink: 0; }}
.prob-row .prob-track {{
    flex-grow: 1; background: #EAF1EF; border-radius: 999px; height: 8px; overflow: hidden;
}}
.prob-row .prob-fill {{ background: {PRIMARY}; height: 100%; border-radius: 999px; }}
.prob-row .prob-pct {{ width: 3rem; text-align: right; flex-shrink: 0; }}
.results-caption {{ color: {TEXT_MUTED}; font-size: 0.9rem; margin-bottom: 0.8rem; }}
.disclaimer {{
    background: #FFF8E8;
    border: 1px solid #F0DDA6;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #6B5518 !important;
    margin-top: 1.5rem;
}}
.disclaimer b, .disclaimer strong {{ color: #6B5518 !important; }}
.about-panel {{
    background: {BG_CARD};
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #E7ECEA;
    margin-bottom: 1rem;
}}
.io-row {{ align-items: flex-start !important; }}
.left-pane {{
    position: sticky !important;
    top: 1rem !important;
    align-self: flex-start !important;
}}
.right-pane {{
    max-height: 82vh !important;
    overflow-y: auto !important;
    padding-right: 0.6rem !important;
}}
"""

ABOUT_HTML = f"""
<div class="about-panel">
<p>Gastritis, inflammation of the stomach lining, is one of the clearest early warning
signs doctors look for when assessing a patient's risk of developing stomach cancer down
the line. Left unaddressed, especially when driven by <i>H. pylori</i> infection, it can
progress through a well-documented sequence of increasingly serious changes to the
stomach lining. Catching and correctly characterizing it early matters.</p>
<p>This project explores whether a model can look at a stomach endoscopy image and answer
the same four questions a gastroenterologist would ask when reading one:</p>
<ul>
<li>Is <i>H. pylori</i> infection present?</li>
<li>Has the stomach lining started to thin out (atrophy)?</li>
<li>If so, where in the stomach is that thinning concentrated?</li>
<li>Which broader type of gastritis does the pattern most resemble, autoimmune,
or driven by <i>H. pylori</i>?</li>
</ul>
<p style="color:{TEXT_MUTED}; font-size:0.85rem;">Research prototype, not a diagnostic device.
Not clinically validated. Should never inform real patient care.</p>
</div>
"""

PATIENT_DIRS = sorted([d for d in SAMPLES_DIR.iterdir() if d.is_dir()]) if SAMPLES_DIR.exists() else []
PATIENT_CHOICES = [d.name for d in PATIENT_DIRS]


def _task_card_html(task, r):
    conf_pct = r["confidence"] * 100
    prob_rows = "".join(
        f"""<div class="prob-row">
              <div class="prob-name">{cls}</div>
              <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%"></div></div>
              <div class="prob-pct">{p*100:.1f}%</div>
            </div>"""
        for cls, p in r["all_probs"].items()
    )
    return f"""
    <div class="task-card">
        <div class="task-title">{TASK_TITLES[task]}</div>
        <div class="task-pred">{r['predicted_class']}</div>
        <div class="conf-track"><div class="conf-fill" style="width:{conf_pct:.1f}%"></div></div>
        <div class="conf-label">{conf_pct:.1f}% confidence</div>
        <details class="all-probs">
            <summary>See all class probabilities</summary>
            {prob_rows}
        </details>
    </div>
    """


def _results_html(results, caption):
    cards = "".join(_task_card_html(task, results[task]) for task in TASK_ORDER)
    return f"""
    <div class="results-caption">{caption}</div>
    <div class="results-grid">{cards}</div>
    """


def _run(images):
    results, n_models, n_images = predict(images)
    if n_models > 1:
        caption = f"Averaged across {n_models} models x {n_images} image(s) = {n_models * n_images} predictions per task."
    else:
        caption = f"Averaged across {n_images} image(s) per task."
    return _results_html(results, caption)


def analyze_sample(patient_name):
    if not patient_name:
        return "<p>Pick a sample patient first.</p>"
    patient_dir = SAMPLES_DIR / patient_name
    files = sorted(patient_dir.glob("*.bmp"))
    images = [Image.open(f) for f in files]
    return _run(images)


def preview_patient(patient_name):
    if not patient_name:
        return []
    patient_dir = SAMPLES_DIR / patient_name
    return [str(f) for f in sorted(patient_dir.glob("*.bmp"))]


def analyze_upload(files):
    if not files:
        return "<p>Upload at least one image first.</p>"
    images = [Image.open(f) for f in files]
    return _run(images)


THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="amber")

with gr.Blocks(title="EGID Gastritis Classifier", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(f"""
    <div class="hero">
      <h1>EGID Gastritis Classifier</h1>
      <p>Upload stomach endoscopy images for an instant multi-label read across four clinical dimensions,
      powered by an ensemble of Vision Transformer models.</p>
      <div class="badge-row">
        <span class="badge">H. pylori status</span>
        <span class="badge">Presence of atrophy</span>
        <span class="badge">Distribution of atrophy</span>
        <span class="badge">Gastritis type</span>
      </div>
    </div>
    """)

    gr.HTML(ABOUT_HTML)

    with gr.Row(elem_classes=["io-row"]):
        with gr.Column(scale=1, elem_classes=["left-pane"]):
            gr.Markdown("#### Input image(s)")
            gr.Markdown(
                "Predictions are averaged across every image you provide (like a real "
                "exam's multiple photos), matching how this model was validated, not "
                "judged from a single frame.",
            )
            with gr.Tabs():
                with gr.Tab("Try a sample patient"):
                    patient_dropdown = gr.Dropdown(
                        choices=PATIENT_CHOICES,
                        value=PATIENT_CHOICES[0] if PATIENT_CHOICES else None,
                        label="Sample patient",
                    )
                    sample_gallery = gr.Gallery(
                        value=preview_patient(PATIENT_CHOICES[0]) if PATIENT_CHOICES else [],
                        label="Patient images", columns=3, height=210,
                        object_fit="cover", preview=False,
                    )
                    sample_button = gr.Button("Analyze sample patient", variant="primary")

                with gr.Tab("Upload your own"):
                    upload = gr.File(
                        label="Upload endoscopy image(s)", file_count="multiple",
                        file_types=["image"], type="filepath",
                    )
                    upload_button = gr.Button("Analyze uploaded image(s)", variant="primary")

        with gr.Column(scale=2, elem_classes=["right-pane"]):
            gr.Markdown("#### Prediction results")
            results_html = gr.HTML("<p style='color:#5B6B67;'>Pick a sample patient or upload image(s), then click Analyze.</p>")

    gr.HTML("""
    <div class="disclaimer">
    <b>Research prototype, not a diagnostic device.</b> This tool is built for educational and research
    purposes on the EGID dataset and has not been clinically validated. Predictions should never be used
    to inform real patient care or replace professional medical judgement.
    </div>
    """)

    patient_dropdown.change(preview_patient, inputs=patient_dropdown, outputs=sample_gallery)
    sample_button.click(analyze_sample, inputs=patient_dropdown, outputs=results_html)
    upload_button.click(analyze_upload, inputs=upload, outputs=results_html)

if __name__ == "__main__":
    load_ensemble()
    demo.launch(ssr_mode=False)
