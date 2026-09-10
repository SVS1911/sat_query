"""
controller/agentic_controller.py
-----------------------------------
SatQueryController: the agentic orchestrator described in the spec.

For each request it:
  1. interprets the query and classifies the requested task     (query_parser)
  2. checks image count/modality/format/compatibility            (input_validator)
  3. selects and executes the appropriate specialist model(s)    (models/*)
  4. combines outputs, estimates confidence, returns visual evidence
  5. returns an auditable execution summary (task, tools, params, checks)

Internal reasoning text (if any is added later, e.g. via an LLM planner) is
never returned to the caller — only the observable execution trace is, per
the spec's evaluation scope.
"""
from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from utils.image_io import LoadedImage, load_image, to_display_rgb
from utils.visualization import overlay_class_map, overlay_binary_mask, draw_bboxes
from utils.spectral_indices import BandRoles, BAND_PRESETS
from controller.input_validator import validate
from controller.query_parser import parse, TaskPlan, set_llm_backend
from controller.llm_query import configured_planner
from models.base_vlm import RemoteSensingVLM
from models import vqa, captioning, grounding, change_analysis, optical_sar_fusion
from models.qwen_backend import load_configured_backend


@dataclass
class ExecutionResult:
    success: bool
    task: str
    scenario: str
    answer: str
    confidence: Optional[float]
    visual_evidence_path: Optional[str]
    audit_trail: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


MODEL_REGISTRY = {
    "vqa": "models.vqa.answer",
    "caption": "models.captioning.caption",
    "grounding": "models.grounding.ground",
    "fusion_analysis": "models.optical_sar_fusion.fuse",
    "change_description": "models.change_analysis.describe_change",
    "change_vqa": "models.change_analysis.change_vqa",
}


class SatQueryController:
    def __init__(self, vlm: Optional[RemoteSensingVLM] = None, reports_dir: str = "reports",
                 caption_checkpoint: Optional[str] = None):
        self.vlm = vlm or RemoteSensingVLM()
        self.qwen_backend = None
        self.qwen_backend_error = None
        try:
            self.qwen_backend = load_configured_backend()
        except (FileNotFoundError, ImportError, RuntimeError, OSError, ValueError) as exc:
            self.qwen_backend_error = str(exc)
        self.llm_planner_error = None
        try:
            planner = configured_planner()
            if planner is not None:
                set_llm_backend(planner)
        except (ValueError, OSError) as exc:
            self.llm_planner_error = str(exc)
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.caption_model = None
        self.caption_model_error = None
        checkpoint = caption_checkpoint or os.environ.get("SATQUERY_CAPTION_CHECKPOINT")
        if checkpoint:
            try:
                self.caption_model = captioning.CaptionModel.load(checkpoint)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.caption_model_error = f"Caption checkpoint '{checkpoint}' could not be loaded: {exc}"

    def run(self, image_paths: List[str], query: str,
             declared_pair_type: Optional[str] = None,
             band_preset: Optional[str] = None) -> ExecutionResult:
        audit = []
        t0 = time.time()
        audit.append(f"Received request: {len(image_paths)} image(s), query='{query}'.")

        band_roles = BAND_PRESETS.get(band_preset) if band_preset else None
        if band_preset:
            audit.append(f"Band role preset: '{band_preset}' -> {band_roles.as_dict()}.")
        else:
            audit.append("No band-role preset supplied — spectral indices requiring NIR/SWIR "
                         "will fall back to RGB color proxies (less reliable on real multiband "
                         "imagery; select a preset matching your source for accurate results).")

        # --- Step 1: load + validate inputs ---
        try:
            loaded = [load_image(p) for p in image_paths]
        except Exception as e:
            return ExecutionResult(False, "invalid", "invalid", "", None, None,
                                    audit + [f"Input loading failed: {e}"], {}, str(e))

        for img in loaded:
            audit.append(f"Loaded '{os.path.basename(img.path)}': {img.width}x{img.height}, "
                         f"{img.bands} band(s), modality={img.modality_guess}, "
                         f"geo_metadata={img.has_geo_metadata}.")
            for w in img.warnings:
                audit.append(f"  warning: {w}")

        validation = validate(loaded, declared_pair_type=declared_pair_type)
        audit.extend(validation.messages)
        if not validation.ok:
            return ExecutionResult(False, "invalid", validation.scenario, "", None, None,
                                    audit, {}, "Input validation failed.")

        scenario = validation.scenario
        audit.append(f"Selected input scenario: {scenario}.")
        if self.llm_planner_error:
            audit.append(f"LLM query planner unavailable; using rule-based parser: {self.llm_planner_error}")
        elif os.environ.get("SATQUERY_LLM_ENDPOINT"):
            audit.append("LLM query planner enabled; specialist execution remains local.")

        # --- Step 2: interpret query / classify task ---
        plan: TaskPlan = parse(query, scenario)
        audit.append(f"Query classified as task='{plan.task}' (rationale: {plan.rationale})")
        if plan.task == "invalid":
            return ExecutionResult(False, "invalid", scenario, "", None, None, audit, {},
                                    "Could not map query to a supported task for this input scenario.")

        model_ref = MODEL_REGISTRY.get(plan.task)
        audit.append(f"Selected model/tool: {model_ref} (RS-adapted backend: "
                     f"{self.vlm.backend.__class__.__name__ if self.vlm.using_finetuned else 'heuristic-spectral-v0'}).")
        if self.qwen_backend is not None and plan.task in {"vqa", "caption"}:
            audit.append("Qwen3-VL BigEarthNet adapter is enabled for single-image answers.")
        elif self.qwen_backend_error:
            audit.append(f"Qwen3-VL backend unavailable; using specialist fallback: {self.qwen_backend_error}")
        if plan.task == "caption":
            if self.caption_model is not None:
                audit.append("Caption backend: dataset-nearest-neighbor checkpoint.")
            elif self.caption_model_error:
                audit.append(f"Caption backend warning: {self.caption_model_error}; using heuristic fallback.")
            else:
                audit.append("Caption backend: heuristic spectral fallback (no checkpoint configured).")

        # --- Step 3: execute the selected workflow ---
        try:
            result, viz_rgb = self._execute(plan, loaded, scenario, audit, band_roles)
        except Exception as e:
            return ExecutionResult(False, plan.task, scenario, "", None, None,
                                    audit + [f"Execution error: {e}"], {}, str(e))

        # --- Step 4: save visual evidence + report ---
        viz_path = None
        if viz_rgb is not None:
            from PIL import Image
            viz_path = os.path.join(self.reports_dir, f"evidence_{int(t0)}.png")
            Image.fromarray(viz_rgb).save(viz_path)
            audit.append(f"Saved visual evidence to {viz_path}.")

        elapsed = time.time() - t0
        audit.append(f"Execution completed in {elapsed:.2f}s.")

        report_path = os.path.join(self.reports_dir, f"report_{int(t0)}.json")
        report = {
            "task": plan.task,
            "scenario": scenario,
            "query": query,
            "answer": result.get("answer") or result.get("caption"),
            "confidence": result.get("confidence"),
            "audit_trail": audit,
            "elapsed_seconds": elapsed,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        audit.append(f"Saved execution report to {report_path}.")

        answer_text = result.get("answer") or result.get("caption") or ""
        return ExecutionResult(True, plan.task, scenario, answer_text, result.get("confidence"),
                                viz_path, audit, result)

    # ------------------------------------------------------------------ #
    def _execute(self, plan: TaskPlan, loaded: List[LoadedImage], scenario: str, audit: List[str],
                 band_roles: Optional[BandRoles] = None):
        if scenario == "single":
            img = loaded[0]
            evidence = self.vlm.encode(img.array, band_roles=band_roles)
            audit.extend(f"  index method: {n}" for n in evidence.notes)
            rgb = to_display_rgb(img)

            if plan.task == "vqa":
                if self.qwen_backend is not None:
                    try:
                        result = self.qwen_backend.answer(img.array, plan.query)
                        return result, overlay_class_map(
                            rgb, evidence.class_map, self.vlm.class_names(), alpha=0.25
                        )
                    except (RuntimeError, ValueError, OSError) as exc:
                        audit.append(f"Qwen3-VL inference failed; using VQA fallback: {exc}")
                result = vqa.answer(evidence, plan.query)
                viz = overlay_class_map(rgb, evidence.class_map, self.vlm.class_names(), alpha=0.25)
                return result, viz

            if plan.task == "caption":
                if self.qwen_backend is not None:
                    try:
                        result = self.qwen_backend.answer(
                            img.array,
                            "Describe the land cover and major objects visible in this image.",
                        )
                        result["caption"] = result["answer"]
                        return result, overlay_class_map(
                            rgb, evidence.class_map, self.vlm.class_names(), alpha=0.25
                        )
                    except (RuntimeError, ValueError, OSError) as exc:
                        audit.append(f"Qwen3-VL inference failed; using caption fallback: {exc}")
                result = captioning.caption(evidence, image=img.array, model=self.caption_model)
                viz = overlay_class_map(rgb, evidence.class_map, self.vlm.class_names(), alpha=0.25)
                return result, viz

            if plan.task == "grounding":
                target = plan.params.get("target_class") or "built_up"
                result = grounding.ground(evidence, target)
                viz = overlay_binary_mask(rgb, result["mask"])
                if result["bboxes"]:
                    viz = draw_bboxes(viz, result["bboxes"], labels=[target] * len(result["bboxes"]))
                return result, viz

        if scenario == "cross_modal":
            optical = next((im for im in loaded if im.modality_guess != "sar"), loaded[0])
            sar = next((im for im in loaded if im is not optical), loaded[1])
            audit.append(f"Assigned roles: optical='{os.path.basename(optical.path)}', "
                         f"sar='{os.path.basename(sar.path)}'.")
            evidence = self.vlm.encode(optical.array, band_roles=band_roles)
            audit.extend(f"  index method: {n}" for n in evidence.notes)
            result = optical_sar_fusion.fuse(evidence, sar.array, target_class=plan.params.get("target_class"))
            rgb = to_display_rgb(optical)
            combined_mask = result["refined_built_up_mask"] | result["refined_water_mask"]
            viz = overlay_binary_mask(rgb, combined_mask, color=(255, 165, 0))
            return result, viz

        if scenario == "bi_temporal":
            before, after = loaded[0], loaded[1]
            ev_before = self.vlm.encode(before.array, band_roles=band_roles)
            ev_after = self.vlm.encode(after.array, band_roles=band_roles)
            audit.extend(f"  index method (before): {n}" for n in ev_before.notes)
            change = change_analysis.compute_change_map(before.array, after.array)
            change["water_reliable"] = any(
                "true NDWI" in note or "true MNDWI" in note
                for note in ev_before.notes + ev_after.notes
            )
            audit.append(f"Change map computed: {change['changed_fraction']*100:.1f}% of pixels flagged as changed.")

            if plan.task == "change_vqa":
                result = change_analysis.change_vqa(ev_before, ev_after, change, plan.query,
                                                     plan.params.get("target_class"))
            else:
                result = change_analysis.describe_change(ev_before, ev_after, change)

            rgb_after = to_display_rgb(after)
            viz = overlay_binary_mask(rgb_after, result["mask"], color=(255, 0, 255))
            return result, viz

        raise RuntimeError(f"No execution path for scenario={scenario}, task={plan.task}")
