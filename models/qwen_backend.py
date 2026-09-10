"""Optional Qwen3-VL BigEarthNet adapter backend.

The adapter is deliberately lazy: importing SatQuery does not download weights.
Set ``SATQUERY_QWEN_ADAPTER`` to a local adapter directory to enable it.
The existing spectral specialists remain the safe fallback for unsupported
modalities and when the optional runtime is unavailable.
"""
from __future__ import annotations

import os
from typing import Any, Dict


class QwenBackend:
    def __init__(self, adapter_path: str):
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.adapter_path):
            raise FileNotFoundError(
                f"Qwen adapter directory does not exist: {self.adapter_path}"
            )
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen backend requires peft, transformers, accelerate, "
                "qwen-vl-utils, and a torch build with model support."
            ) from exc

        revision_file = os.path.join(self.adapter_path, "base_revision.txt")
        if not os.path.isfile(revision_file):
            raise FileNotFoundError("Qwen adapter is missing base_revision.txt")
        with open(revision_file, encoding="utf-8") as handle:
            revision = handle.read().strip()
        if not revision:
            raise ValueError("Qwen base revision is empty")

        config = PeftConfig.from_pretrained(self.adapter_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.adapter_path, trust_remote_code=False
        )
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            revision=revision,
            trust_remote_code=False,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            base = base.to(device)
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    def answer(self, image, question: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen backend is not loaded")
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        batch = self.processor(
            text=[prompt], images=images, videos=videos, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) if hasattr(value, "to") else value
                 for key, value in batch.items()}
        output = self.model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        generated = output[:, batch["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        if not answer:
            raise RuntimeError("Qwen backend returned an empty answer")
        return {
            "answer": answer,
            "confidence": None,
            "backend": "qwen3-vl-bigearthnet-lora",
            "warning": "The adapter does not provide calibrated confidence.",
        }


def load_configured_backend() -> QwenBackend | None:
    path = os.environ.get("SATQUERY_QWEN_ADAPTER", "").strip()
    model_id = os.environ.get("SATQUERY_QWEN_MODEL_ID", "").strip()
    if not path and not model_id:
        return None
    if not path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "SATQUERY_QWEN_MODEL_ID requires the huggingface_hub package."
            ) from exc
        path = snapshot_download(
            repo_id=model_id,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
    return QwenBackend(path)
