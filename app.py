"""
app.py
-------
SatQuery AI — interactive GUI (Gradio), wired to the agentic controller.

Design: a satellite mission-control terminal aesthetic — dark panels, neon
cyan/magenta telemetry accents, monospace data readouts — chosen because the
product IS a remote-sensing command console, not decoration for its own sake.

Run:  python app.py
Then open the printed local URL (default http://127.0.0.1:7860).
"""
import os
import numpy as np
import gradio as gr

from controller.agentic_controller import SatQueryController
from utils.spectral_indices import BAND_PRESETS
from utils.evaluation import evaluate_landcover

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
controller = SatQueryController(reports_dir=REPORTS_DIR)

BAND_PRESET_CHOICES = ["(none — RGB proxy only)"] + list(BAND_PRESETS.keys())

EXAMPLE_QUERIES = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "Is there a significant built-up area in this image?",
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
]

# --------------------------------------------------------------------------- #
# Mission-control neon theme
# --------------------------------------------------------------------------- #
NEON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --sq-bg-deep:    #05070d;
  --sq-bg-panel:   #0b1220;
  --sq-bg-panel-2: #0e1728;
  --sq-cyan:       #00e5ff;
  --sq-magenta:    #ff2ec4;
  --sq-green:      #39ff88;
  --sq-amber:      #ffb020;
  --sq-text:       #e8f4ff;
  --sq-text-muted: #7b8aa3;
  --sq-border:     rgba(0, 229, 255, 0.28);
}

.gradio-container {
  background: radial-gradient(ellipse at top left, #0a1424 0%, var(--sq-bg-deep) 55%) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--sq-text) !important;
}

/* scanline / grid texture over the whole app, subtle */
.gradio-container::before {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(0,229,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.035) 1px, transparent 1px);
  background-size: 42px 42px;
  opacity: 0.5;
}

#sq-header {
  border: 1px solid var(--sq-border);
  border-radius: 4px;
  background: linear-gradient(180deg, var(--sq-bg-panel-2) 0%, var(--sq-bg-panel) 100%);
  padding: 18px 24px;
  margin-bottom: 18px;
  box-shadow: 0 0 24px rgba(0,229,255,0.08), inset 0 0 40px rgba(0,229,255,0.03);
  position: relative;
}
#sq-header::after {
  content: "";
  position: absolute; left: 0; right: 0; top: 0; height: 2px;
  background: linear-gradient(90deg, var(--sq-cyan), var(--sq-magenta), transparent);
}
#sq-header h1 {
  font-family: 'JetBrains Mono', monospace !important;
  letter-spacing: 0.02em;
  font-size: 1.55rem !important;
  margin: 0 0 4px 0 !important;
  color: var(--sq-text) !important;
  text-shadow: 0 0 18px rgba(0,229,255,0.35);
}
#sq-header p { color: var(--sq-text-muted) !important; margin: 0 !important; font-size: 0.92rem; }
#sq-header .sq-accent { color: var(--sq-cyan); }

.sq-panel, .gr-block.gr-box, .block {
  background: var(--sq-bg-panel) !important;
  border: 1px solid rgba(0,229,255,0.14) !important;
  border-radius: 4px !important;
}

label span, .gr-form label span, label.svelte-1f354aw {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.04em;
  color: var(--sq-cyan) !important;
  text-transform: uppercase;
}

textarea, input[type=text], input[type=number] {
  background: #060b14 !important;
  color: var(--sq-text) !important;
  border: 1px solid rgba(0,229,255,0.2) !important;
  border-radius: 3px !important;
}
textarea:focus, input[type=text]:focus {
  border-color: var(--sq-cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}

button.primary, .gr-button-primary {
  background: linear-gradient(90deg, var(--sq-cyan) 0%, #00b8d9 100%) !important;
  color: #001018 !important;
  border: none !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 0.82rem !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.45) !important;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
button.primary:hover { box-shadow: 0 0 30px rgba(0,229,255,0.75) !important; transform: translateY(-1px); }

button.secondary, .gr-button-secondary {
  background: var(--sq-bg-panel-2) !important;
  color: var(--sq-magenta) !important;
  border: 1px solid rgba(255,46,196,0.4) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
}

#sq-answer-box textarea {
  border-color: rgba(57,255,136,0.35) !important;
  color: var(--sq-green) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.95rem !important;
}

#sq-header-md p { color: var(--sq-text) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
#sq-header-md strong { color: var(--sq-magenta) !important; }

#sq-audit textarea {
  background: #04070c !important;
  color: #8fe9ff !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.76rem !important;
  border-color: rgba(0,229,255,0.15) !important;
}

.gr-accordion, details.accordion {
  border: 1px solid rgba(255,46,196,0.25) !important;
  border-radius: 4px !important;
  background: var(--sq-bg-panel-2) !important;
}

#sq-status-strip {
  display: flex; gap: 18px; font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; color: var(--sq-text-muted); margin-top: 10px;
}
#sq-status-strip .dot { color: var(--sq-green); }

table { color: var(--sq-text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }
thead { color: var(--sq-cyan) !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.25); border-radius: 4px; }
"""

_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#05070d",
    block_background_fill="#0b1220",
    block_border_color="rgba(0,229,255,0.14)",
    block_label_text_color="#00e5ff",
    input_background_fill="#060b14",
    button_primary_background_fill="#00e5ff",
    button_primary_text_color="#001018",
)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

def run_query(image_a, image_b, pair_type, band_preset, query):
    if image_a is None:
        return None, "AWAITING INPUT :: upload at least one image to begin.", "", ""

    tmp_dir = os.path.join(REPORTS_DIR, "_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []

    from PIL import Image as PILImage
    a_path = os.path.join(tmp_dir, "image_a.png")
    PILImage.fromarray(image_a).save(a_path)
    paths.append(a_path)

    if image_b is not None:
        b_path = os.path.join(tmp_dir, "image_b.png")
        PILImage.fromarray(image_b).save(b_path)
        paths.append(b_path)

    declared = None if pair_type == "auto-detect" else pair_type.replace("-", "_")
    preset = None if band_preset == BAND_PRESET_CHOICES[0] else band_preset

    result = controller.run(paths, query, declared_pair_type=declared, band_preset=preset)

    audit_text = "\n".join(f"[{i:02d}] {line}" for i, line in enumerate(result.audit_trail))
    if not result.success:
        return None, f"ERROR :: {result.error}", "", audit_text

    viz = None
    if result.visual_evidence_path and os.path.exists(result.visual_evidence_path):
        viz = np.array(PILImage.open(result.visual_evidence_path))

    conf_str = f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
    header = (f"**TASK** `{result.task}`&nbsp;&nbsp;&nbsp;**SCENARIO** `{result.scenario}`"
              f"&nbsp;&nbsp;&nbsp;**CONFIDENCE** `{conf_str}`")

    return viz, result.answer, header, audit_text


def run_evaluation_demo():
    """Runs the built-in synthetic accuracy benchmark (tests/test_accuracy_fix.py's
    scene) live, comparing the RGB-only proxy vs the NIR-based fix, so the
    evaluation harness is visible in the UI even before real labeled data is
    supplied."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
    from test_accuracy_fix import build_synthetic_scene
    from utils.spectral_indices import classify_landcover, LAND_COVER_CLASSES

    arr, gt = build_synthetic_scene()
    pred_proxy, _, _ = classify_landcover(arr[..., :3], band_roles=None)
    pred_fixed, _, _ = classify_landcover(arr, band_roles=BAND_PRESETS["rgb_nir (R,G,B,NIR)"])

    report_proxy = evaluate_landcover(pred_proxy, gt, LAND_COVER_CLASSES)
    report_fixed = evaluate_landcover(pred_fixed, gt, LAND_COVER_CLASSES)

    text = (
        "SYNTHETIC BENCHMARK SCENE — water / shadow / vegetation / built-up\n"
        "(shadow is deliberately spectrally similar to water in RGB — this is\n"
        "the exact failure mode reported on real Planetary Computer imagery)\n\n"
        "── RGB-ONLY PROXY (no band_roles) ──────────────────────────────\n"
        f"{report_proxy.pretty_print()}\n\n"
        "── NIR-BASED FIX (band_roles=rgb_nir) ──────────────────────────\n"
        f"{report_fixed.pretty_print()}\n\n"
        f"SUMMARY  overall accuracy {report_proxy.overall_accuracy*100:.1f}% -> "
        f"{report_fixed.overall_accuracy*100:.1f}%   |   mean IoU "
        f"{report_proxy.mean_iou*100:.1f}% -> {report_fixed.mean_iou*100:.1f}%\n\n"
        "Run this against YOUR labeled data once available:\n"
        "  from utils.evaluation import evaluate_landcover\n"
        "  report = evaluate_landcover(pred_class_map, gt_class_map, LAND_COVER_CLASSES)\n"
        "  print(report.pretty_print())"
    )
    return text


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="SatQuery AI", theme=_theme, css=NEON_CSS) as demo:
    with gr.Column(elem_id="sq-header"):
        gr.Markdown(
            "# 🛰️ SATQUERY <span class='sq-accent'>AI</span>\n"
            "Agentic vision-language terminal for single, cross-modal "
            "(optical + SAR), and bi-temporal remote-sensing analysis."
        )
        gr.HTML(
            "<div id='sq-status-strip'>"
            "<span><span class='dot'>●</span> CONTROLLER: ONLINE</span>"
            "<span><span class='dot'>●</span> BACKEND: HEURISTIC-SPECTRAL-V0</span>"
            "<span><span class='dot'>●</span> INDEX ENGINE: NDVI / NDWI / MNDWI / NDBI</span>"
            "</div>"
        )

    with gr.Tabs():
        with gr.Tab("Query console"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="sq-panel"):
                    image_a = gr.Image(label="Image A (required)", type="numpy")
                    image_b = gr.Image(label="Image B (optional — cross-modal / bi-temporal pair)", type="numpy")
                    pair_type = gr.Radio(
                        ["auto-detect", "cross-modal", "bi-temporal"],
                        value="auto-detect", label="Pair type"
                    )
                    band_preset = gr.Dropdown(
                        BAND_PRESET_CHOICES, value=BAND_PRESET_CHOICES[0],
                        label="Band role preset (critical for real multispectral accuracy)",
                        info="Match this to how your source (e.g. Planetary Computer Sentinel-2) "
                             "stacks bands. Wrong/no preset = classifier falls back to unreliable "
                             "RGB color proxies — see the Evaluation tab for why this matters.",
                    )
                    query = gr.Textbox(label="Natural-language query", lines=2,
                                        placeholder="e.g. Has the built-up area increased, decreased, or remained unchanged?")
                    gr.Examples(examples=EXAMPLE_QUERIES, inputs=query, label="Example queries")
                    run_btn = gr.Button("▶ EXECUTE QUERY", variant="primary")

                with gr.Column(scale=1, elem_classes="sq-panel"):
                    output_image = gr.Image(label="Visual evidence")
                    header_md = gr.Markdown(elem_id="sq-header-md")
                    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False, elem_id="sq-answer-box")
                    with gr.Accordion("Execution trace / audit log", open=False):
                        audit_box = gr.Textbox(label="", lines=14, interactive=False, elem_id="sq-audit")

            run_btn.click(
                run_query,
                inputs=[image_a, image_b, pair_type, band_preset, query],
                outputs=[output_image, answer_box, header_md, audit_box],
            )

demo.launch()
