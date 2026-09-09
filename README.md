# SatQuery AI

SatQuery AI is a CPU-runnable remote-sensing query console with transparent
specialists for captioning, VQA, grounding, optical/SAR fusion, change
analysis, and spectral land-cover classification. Optional heavyweight CNN
backends can be added without changing the controller contract.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the Gradio console with `python app.py`. The base heuristic models require
no model download and work with the sample images in `data/sample/`.

## Public launch checklist

This repository does not configure hosting or DNS. Before a public launch,
connect and verify the custom domain with TLS, serve `favicon.svg`, publish
the Privacy and Terms tabs with deployment-specific legal details, and verify
that no platform attribution is injected by the hosting provider. Do not
represent the local demo as a production service until those checks are
complete.

Answers are written for non-specialists: they use plain words such as
“buildings and roads” instead of “built-up class,” explain where a highlighted
area is, and describe changes as simple increases or decreases. Percentages
are estimates from the image, not measurements from a survey.

## Dataset-backed captioning

The supplied captioning archive is an airport remote-sensing corpus. Extract it
outside this repository (for example, `data/captioning/`) so the images and CSV
files are not committed. The expected layout is:

```text
data/captioning/
  train.csv
  valid.csv
  test.csv
  train/*.jpg
  valid/*.jpg
  test/*.jpg
```

Each CSV needs `filepath` and `captions` columns. `captions` may contain a
normal Python list or the NumPy-style list emitted by the supplied dataset.
The loader resolves paths safely, validates rows, and supports `--limit` for
quick smoke runs.

Train the reproducible CPU-friendly retrieval checkpoint:

```bash
python training/train_captioning.py \
  --dataset data/captioning \
  --output models/captioning.json \
  --seed 0
```

Use it from the controller by passing `caption_checkpoint=...` or setting
`SATQUERY_CAPTION_CHECKPOINT`. Caption queries then retrieve a caption from the
dataset using compact normalized image features. Without a checkpoint, the
controller explicitly reports and uses the spectral heuristic captioner.

## Optional Qwen3-VL BigEarthNet adapter

The colleague's adapter can be used for single-image VQA and captions when
its files are downloaded locally and the machine has the required runtime.
Set `SATQUERY_QWEN_ADAPTER` to the adapter folder before starting the app:

```bash
set SATQUERY_QWEN_ADAPTER=C:\models\satquery-qwen3vl-bigearthnet-txt-lora
python app.py
```

The adapter loads the pinned `Qwen/Qwen3-VL-2B-Instruct` base revision from
`base_revision.txt`. It is not used for bi-temporal change analysis, SAR,
12/13-band imagery, or optical/SAR fusion; those workflows continue to use
the specialist models designed for them. If the optional packages, weights,
GPU, or inference runtime are unavailable, the controller records the reason
and uses the CPU-safe specialist fallback instead of inventing an answer.
Do not put model weights, Hugging Face tokens, or notebook service tokens in
this repository.

## Other model modules

`RemoteSensingVLM` provides the shared evidence seam. It supports the
zero-download spectral backend by default, optional trained segmentation and
scene checkpoints, and forwards band-role presets through all workflows.
`tests/smoke_test.py` exercises VQA, captioning, grounding, cross-modal fusion,
and bi-temporal change analysis.

## Tests

```bash
python -m unittest discover -s tests -p "test*.py"
python -m py_compile app.py controller/*.py models/*.py training/*.py utils/*.py
```

The lightweight captioner is a retrieval model rather than a generative VLM;
quality is bounded by the supplied image/caption coverage and visual feature
similarity. It does not replace a fine-tuned transformer for open-ended
descriptions. Generated reports, uploads, checkpoints, datasets, and caches
should remain outside version control.
