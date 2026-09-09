import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller.agentic_controller import SatQueryController

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
opt1 = os.path.join(SAMPLE, "optical_t1.png")
opt2 = os.path.join(SAMPLE, "optical_t2.png")
sar1 = os.path.join(SAMPLE, "sar_t1.png")

ctrl = SatQueryController(reports_dir=os.path.join(os.path.dirname(__file__), "..", "reports"))

tests = [
    ("single VQA (water presence)", [opt1], "Is there water visible in this image?", None),
    ("single VQA (built-up %)", [opt1], "How much of the image is built-up area?", None),
    ("single caption", [opt1], "Describe the land-cover and major objects visible in this image.", None),
    ("single grounding", [opt1], "Highlight the water body referred to in the query.", None),
    ("cross-modal fusion", [opt1, sar1], "Use the optical and SAR images together to identify built-up and water-covered regions.", None),
    ("bi-temporal change description", [opt1, opt2], "What changed between these two dates, and where did the change occur?", None),
    ("bi-temporal change VQA", [opt1, opt2], "Has the built-up area increased, decreased, or remained unchanged?", None),
]

all_ok = True
for name, imgs, query, pair_type in tests:
    result = ctrl.run(imgs, query, declared_pair_type=pair_type)
    status = "OK" if result.success else "FAIL"
    if not result.success:
        all_ok = False
    print(f"\n=== {name} [{status}] ===")
    print(f"task={result.task} scenario={result.scenario}")
    print(f"answer: {result.answer}")
    print(f"confidence: {result.confidence}")
    print(f"visual evidence: {result.visual_evidence_path}")
    if result.error:
        print(f"error: {result.error}")

print("\n\nALL TESTS PASSED" if all_ok else "\n\nSOME TESTS FAILED")
