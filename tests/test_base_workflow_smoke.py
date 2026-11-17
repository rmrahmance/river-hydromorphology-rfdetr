"""
Live smoke test: one real call to the Base Workflow on Roboflow serverless.

Skipped when ROBOFLOW_API_KEY is not set (environment or .env). Uses 1 credit.

    set ROBOFLOW_API_KEY=<private key>            (Windows)
    python -m unittest tests.test_base_workflow_smoke -v

Sample image, first match wins:
  1. HYDROMORPH_SMOKE_IMAGE=<local path or https URL>
  2. first image in the sibling COCO export: ../River Feature Object Detection.coco/{valid,test,train}/
  3. a valid-split image hosted by Roboflow (needs the URL to be reachable from this machine)
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import base_workflow as bw  # noqa: E402

COCO_EXPORT = ROOT.parent / "River Feature Object Detection.coco"
FALLBACK_URL = "https://source.roboflow.com/Etdqth3F7AeWtJQaSxkKk2kwKtt2/0ZIR9s5bpG2ZhDhjSJJ8/original.jpg"


def sample_image() -> str:
    if os.environ.get("HYDROMORPH_SMOKE_IMAGE"):
        return os.environ["HYDROMORPH_SMOKE_IMAGE"]
    for split in ("valid", "test", "train"):
        folder = COCO_EXPORT / split
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return str(p)
    return FALLBACK_URL


class BaseWorkflowLiveSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api_key = bw.load_api_key()
        except bw.MissingApiKeyError:
            raise unittest.SkipTest("ROBOFLOW_API_KEY not set")

    def test_one_image_returns_declared_outputs(self):
        image = sample_image()
        res = bw.detect_river_features(image, api_key=self.api_key, timeout_s=90)
        missing = bw.EXPECTED_OUTPUTS - res.output_keys
        self.assertFalse(missing, f"missing outputs {sorted(missing)}; got {sorted(res.output_keys)}")
        self.assertIsInstance(res.detections, tuple)
        self.assertGreater(res.image_width or 0, 0)
        self.assertTrue(res.model_id and res.model_id.startswith("river-feature-object-detection-wbzft/"))
        print(f"\n  model {res.model_id}  {res.image_width}x{res.image_height}  {res.counts() or 'no detections'}")


if __name__ == "__main__":
    unittest.main()
