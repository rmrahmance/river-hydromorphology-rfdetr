# River hydromorphology detector: Roboflow setup and how to run it

## What is deployed (workspace `rs-workspace-o0dgc`)

| Item | ID / value |
|---|---|
| Project | `river-feature-object-detection-wbzft`: 436 images, split 361 / 50 / 25 |
| Dataset version | v1: auto-orient, fit within 1280 px, 3× augmentation (flip, 0–20 % crop, ±20 % brightness, ±10 % exposure, ±10° hue, ±20 % saturation, 1 px blur, 1 % noise) |
| Model | RF-DETR Medium at 1280 px, model ID `rs-workspace-o0dgc/river-feature-object-detection-wbzft-1-rfdetr-medium-t1` |
| Image Workflow | `hydromorph-image-detection`: detector, confidence filter, annotated image with class and confidence |
| Video Workflow | `hydromorph-video-unique-count`: detector, ByteTrack, class lock, event log (each object counted once) |
| Base Workflow | `river-feature-object-detection-wbzft`: detector only, raw detections, no annotated image. Uses `river-feature-object-detection-wbzft/1`, which is the same RF-DETR Medium model (the only training on version 1) |
| Default confidence | **0.35** |

Training metrics (Roboflow, validation split): mAP@50 57.5 %, precision 66.4 %, recall 51.3 %.

## Why the confidence threshold is 0.35

I ran the model on 74 of the 75 held-out images (49 valid, 25 test; one valid image is excluded) and compared predicted counts with labeled counts for each class and image. This count-matching check scores counts only, not box overlap.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.65 | 0.82 | 0.73 |
| **0.35** | **0.74** | **0.77** | **0.75** |
| 0.40 | 0.80 | 0.70 | 0.75 |
| 0.50 | 0.87 | 0.60 | 0.71 |

- At 0.35, precision is about equal to recall, so the total count comes out nearly unbiased (871 predicted, 838 labeled). Single classes can be further off, see the table below. That matters most for an inventory.
- Per-class thresholds fitted on the valid split did worse on test (F1 0.712) than one global threshold (0.735). With this little data they overfit, so one global threshold is used. The image Workflow still accepts a `class_thresholds` dict if you want to override individual classes later.
- Tiled (SAHI-style) inference on 4K stills was tested and did not help, so it is not used.

Count F1 per class at 0.35 (labeled vs predicted totals over the 74 images):

| Class | Labeled | Predicted | F1 |
|---|---|---|---|
| building | 177 | 226 | 0.80 |
| road | 75 | 73 | 0.80 |
| wood_stump | 118 | 112 | 0.78 |
| woody_debris_jam | 102 | 113 | 0.78 |
| riffle | 57 | 56 | 0.78 |
| Island | 38 | 42 | 0.78 |
| fallen_tree | 87 | 96 | 0.74 |
| Erosion | 73 | 65 | 0.68 |
| longitudinal_bar | 35 | 31 | 0.64 |
| root | 52 | 44 | 0.54 |

building over-counts at 0.35; pass `class_thresholds={"building": 0.40}` to fix that.
Not reliable yet, with only 8 held-out labels each: Bank_protection, bifurcation, organic_bed.
Not trained, with under 10 labels in total: bed_protection, bed_fall, transverse_bar, ramp, bedrock.

## Run it on your own images and videos

```bash
pip install "inference-sdk[webrtc]>=1.7" opencv-python
set ROBOFLOW_API_KEY=<private key from app.roboflow.com/rs-workspace-o0dgc/settings/api>
python hydromorph_detect.py "D:\path\to\photo.jpg"
python hydromorph_detect.py "D:\path\to\folder"            # all images and videos in it
python hydromorph_detect.py clip.mp4 --min-frames 5          # unique-object inventory
python hydromorph_detect.py clip.mp4 --engine local          # run on your PC (pip install inference)
python hydromorph_detect.py "D:\path\to\folder" --image-workflow base   # images via the Base Workflow
```

Outputs go to `hydromorph_results/`:
- **Images:** `*_annotated.jpg`, `*_detections.json`, and `image_inventory.csv` (class, count, mean/min/max confidence). With `--image-workflow base` there is no `*_annotated.jpg`.
- **Videos:** `*_annotated.mp4`, `*_tracks.csv` (one row per physical object: track ID, class, frames seen, first seen in seconds, confidence), and `*_inventory.csv` (unique objects per class).

CSV columns:

| File | Columns |
|---|---|
| `image_inventory.csv` | `image`, `class`, `count`, `mean_conf`, `min_conf`, `max_conf` (one row per image and class) |
| `*_tracks.csv` | `track_id`, `class`, `frames_seen`, `first_frame`, `last_frame`, `first_seen_s`, `mean_conf`, `max_conf` |
| `*_inventory.csv` | `class`, `unique_objects` |

In `*_tracks.csv` the class is the one the tracker saw most often for that track. Tracks seen in fewer than `--min-frames` frames (default 5) are dropped from both video CSVs. `first_seen_s` stays empty if the video has no frame rate.

The first run takes about a minute while the model warms up.

## Base Workflow from Python (`base_workflow.py`)

`detect_river_features()` runs the Base Workflow on one image and returns typed detections. Use it in your own scripts or notebooks.

```python
from base_workflow import detect_river_features, BaseWorkflowError

res = detect_river_features(r"D:\path\to\photo.jpg", confidence=0.35)
res.counts()               # {'woody_debris_jam': 1, 'fallen_tree': 2}
for d in res.detections:   # class_name, class_id, confidence, x, y, width, height (box centre, px)
    print(d.class_name, round(d.confidence, 2), d.xyxy)
```

| Parameter | Workflow default | Note |
|---|---|---|
| `confidence` | 0.4 | Pass 0.35 to match the threshold evaluated above (the runner does) |
| `iou_threshold` | 0.3 | NMS overlap |
| `class_agnostic_nms` | False | |
| `max_detections` | 1000 | |
| `model_id` | `river-feature-object-detection-wbzft/1` | Point at a newer version after retraining |

- Parameters left as `None` fall back to the Workflow defaults.
- Image: local path, `https://` URL, numpy array (BGR) or PIL image. `http://` is rejected. The SDK downloads URLs and sends every image as base64.
- Auth: key from `ROBOFLOW_API_KEY` or `.env`, sent only in the `Authorization: Bearer` header. Needs `inference-sdk>=1.7`.
- Timeout 60 s per attempt, then up to 2 retries (1 s, 2 s backoff) on timeouts, connection errors and HTTP 408/429/5xx. A timed-out request can still finish on Roboflow and use a credit.
- Errors all subclass `BaseWorkflowError`: `MissingApiKeyError`, `SdkVersionError`, `InvalidImageInputError`, `InvalidParameterError`, `WorkflowAuthError` (401/403), `WorkflowNotFoundError` (404), `WorkflowRequestError` (other 4xx), `WorkflowServerError` (429/5xx), `WorkflowTimeoutError`, `WorkflowConnectionError`, `WorkflowResponseError` (response no longer matches the Workflow's outputs).

Tests (run from this folder):

```bash
python -m unittest discover -s tests -v                # offline, no key; replays a recorded real response
python -m unittest tests.test_base_workflow_smoke -v   # live, 1 credit; skipped without ROBOFLOW_API_KEY
```

The live test uses the first image in `..\River Feature Object Detection.coco\valid\`. Set `HYDROMORPH_SMOKE_IMAGE` to use another.

## Next steps that would raise accuracy most

1. Label more examples of the weak and rare classes (root, longitudinal_bar, Erosion, Bank_protection, bifurcation, organic_bed, plus the 5 untrained classes). Aim for at least 50 instances each.
2. Add more full-resolution drone stills, such as the Erft Gnadental Mavic photos. On one test still the model found 0 of 15 wood stumps at the default threshold.
3. Remove `Annotated_Erft_Gnadental_250617_Foto_MavicPro_0013_D.JPG`. It has boxes drawn into the pixels and no labels.
4. Split train/valid/test by video rather than by frame. Neighbouring frames of the same clip now sit in different splits, which inflates the scores.

## Files in this folder

| File | Contents |
|---|---|
| `hydromorph_detect.py` | Runner for images, folders and videos (unique-object inventory) |
| `base_workflow.py` | Python client for the Base Workflow: `detect_river_features()` |
| `tests/` | Offline tests and live smoke test for `base_workflow.py`; `fixtures/` holds a recorded real response |
| `project_info.json` | All Roboflow IDs, version settings, training recipe and metrics |
| `workflows/*.json` | Exact definitions of the two deployed Workflows; re-import in the Roboflow Workflow editor if needed |
| `evaluation/gt.txt` | Labeled per-class counts of the 74 held-out images |
| `evaluation/preds.txt` | Model detections (class and confidence ≥ 0.20) for the same images |
| `evaluation/sweep.py` | Confidence-sweep script (`python sweep.py`) |
| `evaluation/sweep_results.txt`, `thresholds.json` | Sweep output |
