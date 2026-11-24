"""
Offline tests for evaluation/sweep.py.

sweep.py runs on import and writes thresholds.json next to itself, so the script
and its two input files are copied to a temporary folder first.

    python -m unittest discover -s tests -p test_sweep.py -v
"""

import contextlib
import io
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "evaluation"


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("sweep.py", "gt.txt", "preds.txt"):
                shutil.copy(EVAL / name, tmp)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cls.ns = runpy.run_path(str(Path(tmp) / "sweep.py"))
            cls.stdout = out.getvalue()
            cls.written = json.loads((Path(tmp) / "thresholds.json").read_text())

    def test_thresholds_json_is_reproduced(self):
        self.assertEqual(self.written, json.loads((EVAL / "thresholds.json").read_text()))

    def test_printed_results_are_reproduced(self):
        saved = (EVAL / "sweep_results.txt").read_text()
        self.assertEqual(self.stdout.split(), saved.split())

    def test_prf_by_hand(self):
        # 6 TP, 2 FP, 4 FN: P = 6/8, R = 6/10, F1 = 2*0.75*0.6/1.35 = 2/3
        p, r, f = self.ns["prf"](6, 2, 4)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.6)
        self.assertAlmostEqual(f, 2 / 3)

    def test_prf_without_any_boxes_is_zero(self):
        self.assertEqual(self.ns["prf"](0, 0, 0), (0.0, 0.0, 0.0))

    def test_counts_one_image_by_hand(self):
        # Image 1 has 3 labelled woody debris jams, predicted at .85 .66 .66 .32 .24
        counts = self.ns["counts"]
        self.assertEqual(counts([1], "WD", 0.70), (1, 0, 2))
        self.assertEqual(counts([1], "WD", 0.66), (3, 0, 0))  # threshold itself is included
        self.assertEqual(counts([1], "WD", 0.30), (3, 1, 0))


if __name__ == "__main__":
    unittest.main()
