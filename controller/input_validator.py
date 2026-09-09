"""
controller/input_validator.py
-------------------------------
Checks number, modality, format, metadata, and compatibility of input images
before the agentic controller selects a workflow. This is deliberately a
separate step so its checks show up verbatim in the auditable execution trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from utils.image_io import LoadedImage, same_footprint


@dataclass
class ValidationResult:
    ok: bool
    scenario: str            # "single" | "cross_modal" | "bi_temporal" | "invalid"
    messages: List[str] = field(default_factory=list)


def validate(images: List[LoadedImage], declared_pair_type: Optional[str] = None) -> ValidationResult:
    """
    declared_pair_type: optional user hint - "cross_modal" or "bi_temporal" - used
    to disambiguate two-image inputs when modality auto-detection is uncertain
    (e.g. two optical images from different sensors both being mislabeled).
    """
    msgs = []

    if len(images) == 0:
        return ValidationResult(False, "invalid", ["No images supplied."])

    if len(images) == 1:
        img = images[0]
        msgs.append(f"Single image accepted: {img.width}x{img.height}, {img.bands} band(s), "
                     f"modality={img.modality_guess}, geo_metadata={img.has_geo_metadata}.")
        if img.bands == 0:
            return ValidationResult(False, "invalid", msgs + ["Image failed to decode."])
        return ValidationResult(True, "single", msgs)

    if len(images) == 2:
        a, b = images
        ok_fp, fp_msg = same_footprint(a, b)
        msgs.append(f"Footprint/size compatibility check: {fp_msg}")
        if not ok_fp:
            return ValidationResult(False, "invalid", msgs)

        modalities = {a.modality_guess, b.modality_guess}
        if declared_pair_type in ("cross_modal", "bi_temporal"):
            scenario = declared_pair_type
            msgs.append(f"Pair type set by user declaration: {declared_pair_type}.")
        elif "sar" in modalities and "optical" in modalities:
            scenario = "cross_modal"
            msgs.append("Auto-detected one SAR + one optical image -> cross-modal pair.")
        else:
            scenario = "bi_temporal"
            msgs.append("Auto-detected two same-modality images -> treating as bi-temporal pair. "
                        "If these are actually two different sensors, set pair type explicitly.")

        msgs.append(f"Image A: {a.width}x{a.height}, {a.bands}b, modality={a.modality_guess}")
        msgs.append(f"Image B: {b.width}x{b.height}, {b.bands}b, modality={b.modality_guess}")
        return ValidationResult(True, scenario, msgs)

    return ValidationResult(False, "invalid", [f"Unsupported number of images: {len(images)} "
                                                f"(expected 1 for single-image tasks or 2 for "
                                                f"cross-modal/bi-temporal pairs)."])
