"""
Offline tests for base_workflow.py. No network, no API key needed.

The fixture is a real response from the published Base Workflow
(tests/fixtures/base_workflow_response.json). The wire-level test drives the
real inference-sdk client and only stubs requests.post.

Run from the Roboflow folder:
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import base_workflow as bw  # noqa: E402

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "base_workflow_response.json").read_text())
FAKE_KEY = "test-key-not-real"
URL_IMAGE = "https://example.com/river.jpg"


class FakeClient:
    """Stands in for InferenceHTTPClient.run_workflow; replays a script of results/errors."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def run_workflow(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step()
        return step


def http_error(code):
    from inference_sdk.http.errors import HTTPCallErrorError
    return HTTPCallErrorError(description=f"HTTP {code}", status_code=code, api_message="test")


class BaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        bw._build_client.cache_clear()

    def run_with(self, script, **kw):
        fake = FakeClient(script)
        with mock.patch.object(bw, "_build_client", return_value=fake):
            kw.setdefault("api_key", FAKE_KEY)
            kw.setdefault("backoff_s", 0)
            return bw.detect_river_features(URL_IMAGE, **kw), fake

    # -- parsing the real response
    def test_parses_real_response(self):
        res, _ = self.run_with([FIXTURE])
        self.assertTrue(bw.EXPECTED_OUTPUTS <= res.output_keys)
        self.assertEqual((res.image_width, res.image_height), (1600, 1200))
        self.assertEqual(res.model_id, "river-feature-object-detection-wbzft/1")
        self.assertEqual(res.inference_id, "8d870391-dbd3-4676-885a-1d5ea1cf4fb6")
        self.assertEqual(res.counts(), {"woody_debris_jam": 1, "fallen_tree": 2})
        d = res.detections[0]
        self.assertEqual((d.class_name, d.class_id), ("woody_debris_jam", 12))
        self.assertAlmostEqual(d.confidence, 0.6195, places=4)
        self.assertEqual(d.xyxy, (535.0, 442.0, 1239.0, 826.0))
        self.assertNotIn("parent_id", d.to_dict())  # only fields we use are kept

    # -- request shape
    def test_sends_ids_and_only_set_parameters(self):
        _, fake = self.run_with([FIXTURE], confidence=0.35, max_detections=200)
        call = fake.calls[0]
        self.assertEqual(call["workspace_name"], "rs-workspace-o0dgc")
        self.assertEqual(call["workflow_id"], "river-feature-object-detection-wbzft")
        self.assertEqual(call["images"], {"image": URL_IMAGE})
        self.assertEqual(call["parameters"], {"confidence": 0.35, "max_detections": 200})

    def test_wire_request_uses_bearer_header(self):
        """Real SDK client on a real local JPEG; only requests.post is stubbed."""
        from PIL import Image

        captured = {}

        def fake_post(url, json=None, headers=None, **kw):
            captured.update(url=url, json=json, headers=headers)
            resp = mock.Mock(status_code=200)
            resp.json.return_value = {"outputs": FIXTURE}
            return resp

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "river.jpg"
            Image.new("RGB", (64, 48), (40, 90, 60)).save(img_path)
            with mock.patch("requests.post", side_effect=fake_post):
                res = bw.detect_river_features(img_path, confidence=0.35, api_key=FAKE_KEY)
        self.assertEqual(captured["url"],
                         "https://serverless.roboflow.com/rs-workspace-o0dgc/workflows/river-feature-object-detection-wbzft")
        self.assertEqual(captured["headers"].get("Authorization"), f"Bearer {FAKE_KEY}")
        self.assertNotIn("api_key", captured["json"])
        self.assertNotIn(FAKE_KEY, captured["url"])
        image_in = captured["json"]["inputs"]["image"]
        self.assertEqual(image_in["type"], "base64")  # the SDK encodes files (and downloads URLs) client-side
        self.assertTrue(image_in["value"])
        self.assertEqual(captured["json"]["inputs"]["confidence"], 0.35)
        self.assertEqual(len(res.detections), 3)

    # -- retries and errors
    def test_retries_transient_then_succeeds(self):
        res, fake = self.run_with([http_error(503), http_error(429), FIXTURE], retries=2)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(res.detections), 3)

    def test_gives_up_after_retries(self):
        with self.assertRaises(bw.WorkflowServerError) as cm:
            self.run_with([http_error(503)] * 3, retries=2)
        self.assertEqual(cm.exception.status_code, 503)

    def test_auth_error_not_retried(self):
        fake = FakeClient([http_error(401), FIXTURE])
        with mock.patch.object(bw, "_build_client", return_value=fake):
            with self.assertRaises(bw.WorkflowAuthError):
                bw.detect_river_features(URL_IMAGE, api_key=FAKE_KEY, backoff_s=0)
        self.assertEqual(len(fake.calls), 1)

    def test_not_found_and_bad_request_typed(self):
        with self.assertRaises(bw.WorkflowNotFoundError):
            self.run_with([http_error(404)])
        with self.assertRaises(bw.WorkflowRequestError):
            self.run_with([http_error(400)])

    def test_connection_error_retried_then_typed(self):
        from inference_sdk.http.errors import HTTPClientError
        with self.assertRaises(bw.WorkflowConnectionError):
            self.run_with([HTTPClientError("Error with server connection")] * 2, retries=1)

    def test_timeout(self):
        slow = lambda: (time.sleep(0.5), FIXTURE)[1]  # noqa: E731
        t0 = time.monotonic()
        with self.assertRaises(bw.WorkflowTimeoutError):
            self.run_with([slow], timeout_s=0.1, retries=0)
        self.assertLess(time.monotonic() - t0, 0.4)

    # -- input validation (no request made)
    def test_rejects_http_url_and_missing_file(self):
        with self.assertRaises(bw.InvalidImageInputError):
            bw.detect_river_features("http://example.com/a.jpg", api_key=FAKE_KEY)
        with self.assertRaises(bw.InvalidImageInputError):
            bw.detect_river_features("does_not_exist.jpg", api_key=FAKE_KEY)

    def test_rejects_bad_parameters(self):
        for kw in ({"confidence": 1.5}, {"iou_threshold": "0.3"}, {"max_detections": 0},
                   {"class_agnostic_nms": "yes"}, {"model_id": "no-version"}):
            with self.subTest(kw=kw), self.assertRaises(bw.InvalidParameterError):
                bw.detect_river_features(URL_IMAGE, api_key=FAKE_KEY, **kw)

    def test_missing_output_is_reported(self):
        with self.assertRaises(bw.WorkflowResponseError) as cm:
            self.run_with([[{"model_id": "x/1"}]])
        self.assertIn("model_id", str(cm.exception))

    def test_malformed_detection_is_reported(self):
        bad = [{"predictions": {"image": {}, "predictions": [{"class": "root"}]}}]
        with self.assertRaises(bw.WorkflowResponseError):
            self.run_with([bad])

    def test_missing_api_key(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": ""}), \
                mock.patch.object(bw, "__file__", str(Path(tmp) / "base_workflow.py")):
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(bw.MissingApiKeyError):
                    bw.load_api_key()
            finally:
                os.chdir(cwd)


class RunnerIntegrationTests(unittest.TestCase):
    """hydromorph_detect.py --image-workflow base writes detections JSON + inventory CSV."""

    def test_runner_base_workflow(self):
        import hydromorph_detect as hd
        from PIL import Image

        fake = FakeClient([FIXTURE])
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "erft.jpg"
            Image.new("RGB", (32, 32)).save(img)
            out = Path(tmp) / "results"
            argv = ["hydromorph_detect.py", str(img), "--image-workflow", "base", "--out", str(out)]
            with mock.patch.object(bw, "_build_client", return_value=fake), \
                    mock.patch.dict(os.environ, {"ROBOFLOW_API_KEY": FAKE_KEY}), \
                    mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                hd.main()
            dets = json.loads((out / "erft_detections.json").read_text())
            inv = (out / "image_inventory.csv").read_text().splitlines()
        self.assertEqual(fake.calls[0]["parameters"], {"confidence": hd.DEFAULT_CONF})
        self.assertEqual([d["class"] for d in dets], ["woody_debris_jam", "fallen_tree", "fallen_tree"])
        self.assertEqual(inv[0], "image,class,count,mean_conf,min_conf,max_conf")
        self.assertIn("erft.jpg,fallen_tree,2,0.442,0.421,0.463", inv)


if __name__ == "__main__":
    unittest.main()
