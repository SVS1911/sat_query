"""
models/vqa.py
--------------
Single-image visual question answering (mandatory baseline).

Answers are grounded in the ImageEvidence produced by RemoteSensingVLM.encode()
(land-cover proportions today; swap in a fine-tuned VQA head later and this
module's public interface, answer(), doesn't need to change).
"""
from __future__ import annotations

import re
from typing import Dict, Any

from models.base_vlm import ImageEvidence

_PRESENCE_WORDS = ["is there", "are there", "does this", "any ", "presence of"]
_PERCENT_WORDS = ["how much", "percentage", "percent", "proportion", "fraction", "how large"]
_COUNT_WORDS = ["how many"]

_CLASS_ALIASES = {
    "water": ["water", "river", "lake", "pond", "reservoir", "flood", "coast"],
    "vegetation": ["vegetation", "forest", "tree", "crop", "farmland", "green area", "plants"],
    "built_up": ["built-up", "built up", "urban", "building", "road", "settlement", "infrastructure", "city"],
    "bare_soil": ["bare soil", "barren", "sand", "soil", "bare land"],
}


def _find_target_class(text: str):
    for cls, kws in _CLASS_ALIASES.items():
        if any(kw in text for kw in kws):
            return cls
    return None


def answer(evidence: ImageEvidence, question: str) -> Dict[str, Any]:
    text = question.lower()
    target = _find_target_class(text)
    proportions = evidence.proportions

    if target:
        pct = proportions.get(target, 0.0) * 100
        is_presence_q = any(w in text for w in _PRESENCE_WORDS)
        is_percent_q = any(w in text for w in _PERCENT_WORDS)

        if is_presence_q:
            present = pct > 3.0
            answer_text = (
                f"Yes. I can see {target.replace('_', ' ')} in about {pct:.1f}% of the image."
                if present else
                f"No large {target.replace('_', ' ')} area is visible. The estimate is {pct:.1f}% of the image."
            )
            confidence = min(0.95, 0.55 + pct / 100)
        elif is_percent_q:
            answer_text = f"About {pct:.1f}% of the image appears to be {target.replace('_', ' ')}."
            confidence = 0.65
        else:
            answer_text = f"The image contains about {pct:.1f}% {target.replace('_', ' ')}."
            confidence = 0.6
        return {"answer": answer_text, "confidence": round(confidence, 2), "target_class": target,
                "proportions": proportions}

    # Generic fallback: describe dominant land cover types.
    ranked = sorted(proportions.items(), key=lambda kv: kv[1], reverse=True)
    dom = [c for c, f in ranked if f > 0.05][:2]
    dom_str = " and ".join(d.replace("_", " ") for d in dom) if dom else "mixed land cover"
    answer_text = (
        f"Most of the image shows {dom_str}. I could not identify the exact subject of the question. "
        f"Try asking about water, plants, buildings, roads, or bare ground."
    )
    return {"answer": answer_text, "confidence": 0.4, "target_class": None, "proportions": proportions}
