"""
Hydromorphological feature detection + unique-object inventory (Roboflow, RF-DETR).

Images  -> per-image detections with confidence, annotated JPG, per-class counts.
Videos  -> ByteTrack-tracked detections; each physical object counted ONCE
           (unique track IDs seen in >= --min-frames frames), annotated MP4.

Setup (Python 3.10-3.12):
    pip install "inference-sdk[webrtc]" opencv-python
    set ROBOFLOW_API_KEY=<your private key>        (Windows)
    export ROBOFLOW_API_KEY=<your private key>     (macOS/Linux)

Usage:
    python hydromorph_detect.py path/to/photo.jpg
    python hydromorph_detect.py path/to/folder_of_images_and_videos
    python hydromorph_detect.py path/to/drone_clip.mp4 --min-frames 5
    python hydromorph_detect.py clip.mp4 --engine local   # run on your own machine (pip install inference)
    python hydromorph_detect.py photo.jpg --image-workflow base   # raw detections, Base Workflow (base_workflow.py)

Outputs go to ./hydromorph_results/ (change with --out).
"""

import argparse
import base64
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = "rs-workspace-o0dgc"
IMAGE_WORKFLOW = "hydromorph-image-detection"
VIDEO_WORKFLOW = "hydromorph-video-unique-count"
API_URL = "https://serverless.roboflow.com"
DEFAULT_CONF = 0.35  # best-F1 global threshold from held-out sweep; counts ~unbiased (P~R)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv"}


def api_key() -> str:
    """Roboflow key from ROBOFLOW_API_KEY, or from a .env file in the current folder."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ROBOFLOW_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY (environment variable or .env file).")
    return key


def preds_list(obj) -> list[dict]:
    """Normalise a serialised detections output to a list of prediction dicts."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.get("predictions", [])
    if isinstance(obj, list):
        return obj
    return []


# ----------------------------------------------------------------- images
def run_images(paths: list[Path], out_dir: Path, conf: float) -> None:
    """Run the hydromorph image workflow on each image.

    Writes <stem>_annotated.jpg, <stem>_detections.json and image_inventory.csv to out_dir.
    """
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key())
    rows = []
    for p in paths:
        res = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=IMAGE_WORKFLOW,
            images={"image": str(p)},
            parameters={"confidence": conf},
        )[0]
        preds = preds_list(res.get("predictions"))
        img = res.get("annotated_image")
        if isinstance(img, dict) and img.get("type") == "base64":
            img = img["value"]
        if isinstance(img, str):
            (out_dir / f"{p.stem}_annotated.jpg").write_bytes(base64.b64decode(img))
        (out_dir / f"{p.stem}_detections.json").write_text(json.dumps(preds, indent=2))
        rows.extend(_inventory_rows(p.name, preds))
        print(f"{p.name}: {dict(Counter(d['class'] for d in preds)) or 'no features detected'}")
    _write_image_inventory(out_dir, rows)


def run_images_base(paths: list[Path], out_dir: Path, conf: float) -> None:
    """Raw detections from the Base Workflow (no annotated image, no per-class filter)."""
    from base_workflow import (BaseWorkflowError, SdkVersionError, WorkflowAuthError,
                               WorkflowNotFoundError, detect_river_features)

    key = api_key()
    rows, failed = [], 0
    for p in paths:
        try:
            res = detect_river_features(p, confidence=conf, api_key=key)
        except (WorkflowAuthError, WorkflowNotFoundError, SdkVersionError) as e:
            sys.exit(f"{type(e).__name__}: {e}")  # same failure for every image: stop
        except BaseWorkflowError as e:
            failed += 1
            print(f"{p.name}: FAILED ({type(e).__name__}: {e})")
            continue
        preds = [d.to_dict() for d in res.detections]
        (out_dir / f"{p.stem}_detections.json").write_text(json.dumps(preds, indent=2))
        rows.extend(_inventory_rows(p.name, preds))
        print(f"{p.name}: {res.counts() or 'no features detected'}")
    _write_image_inventory(out_dir, rows)
    if failed:
        print(f"{failed} of {len(paths)} images failed; see messages above.")


def _inventory_rows(image_name: str, preds: list[dict]) -> list[list]:
    """One row per class: image, class, count, mean/min/max confidence."""
    counts = Counter(d["class"] for d in preds)
    rows = []
    for cls, n in sorted(counts.items()):
        confs = [d["confidence"] for d in preds if d["class"] == cls]
        rows.append([image_name, cls, n, round(sum(confs) / n, 3), round(min(confs), 3), round(max(confs), 3)])
    return rows


def _write_image_inventory(out_dir: Path, rows: list[list]) -> None:
    """Write image_inventory.csv (columns as in docs/USAGE.md)."""
    with open(out_dir / "image_inventory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "count", "mean_conf", "min_conf", "max_conf"])
        w.writerows(rows)


# ----------------------------------------------------------------- videos
class TrackBook:
    """Accumulates per-track evidence; one row per physical object."""

    def __init__(self):
        self.frames = defaultdict(int)
        self.first = {}
        self.last = {}
        self.cls_votes = defaultdict(Counter)
        self.conf = defaultdict(list)

    def add(self, frame_id: int, tid, cls: str, conf: float) -> None:
        """Record one tracked detection. Detections without a track ID are ignored."""
        if tid is None or tid < 0:
            return
        self.frames[tid] += 1
        self.first.setdefault(tid, frame_id)
        self.last[tid] = frame_id
        self.cls_votes[tid][cls] += 1
        self.conf[tid].append(conf)

    def write(self, out_dir: Path, stem: str, min_frames: int, fps: float) -> Counter:
        """Write <stem>_tracks.csv and <stem>_inventory.csv and return unique objects per class.

        Tracks seen in fewer than min_frames frames are dropped. A track's class is the one
        it was given most often.
        """
        tracks = []
        for tid, n in self.frames.items():
            if n < min_frames:
                continue
            cls = self.cls_votes[tid].most_common(1)[0][0]
            c = self.conf[tid]
            tracks.append([tid, cls, n, self.first[tid], self.last[tid],
                           round(self.first[tid] / fps, 2) if fps else "",
                           round(sum(c) / len(c), 3), round(max(c), 3)])
        tracks.sort(key=lambda r: (r[1], r[3]))
        with open(out_dir / f"{stem}_tracks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "class", "frames_seen", "first_frame", "last_frame",
                        "first_seen_s", "mean_conf", "max_conf"])
            w.writerows(tracks)
        inv = Counter(r[1] for r in tracks)
        with open(out_dir / f"{stem}_inventory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "unique_objects"])
            for cls, n in sorted(inv.items()):
                w.writerow([cls, n])
        return inv


def run_video_serverless(path: Path, out_dir: Path, min_frames: int, plan, conf: float) -> Counter:
    """Track objects in a video on Roboflow serverless (WebRTC) and write the annotated MP4 and CSVs."""
    import cv2
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.webrtc import StreamConfig, VideoFileSource

    client = InferenceHTTPClient.init(api_url=API_URL, api_key=api_key())
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    config = StreamConfig(
        stream_output=["annotated_image"],
        data_output=["tracked_predictions"],
        realtime_processing=False,  # process every frame, no dropping
        processing_timeout=3600,
        requested_plan=plan,
        workflow_parameters={"confidence": conf},
    )
    session = client.webrtc.stream(
        source=VideoFileSource(str(path), realtime_processing=False),
        workflow=VIDEO_WORKFLOW,
        workspace=WORKSPACE,
        image_input="image",
        config=config,
    )
    book = TrackBook()
    writer = {"w": None}

    @session.on_frame
    def on_frame(frame, metadata):
        if writer["w"] is None:
            h, w = frame.shape[:2]
            writer["w"] = cv2.VideoWriter(str(out_dir / f"{path.stem}_annotated.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), fps or 25, (w, h))
        writer["w"].write(frame)

    counter = {"n": 0}

    @session.on_data("tracked_predictions")
    def on_tracks(data, metadata):
        counter["n"] += 1
        fid = metadata.frame_id if metadata is not None else counter["n"]
        for d in preds_list(data):
            book.add(fid, d.get("tracker_id"), d.get("class"), d.get("confidence", 0))

    session.run()
    if writer["w"] is not None:
        writer["w"].release()
    return book.write(out_dir, path.stem, min_frames, fps)


def run_video_local(path: Path, out_dir: Path, min_frames: int, conf: float) -> Counter:
    """Same as run_video_serverless, but runs the workflow on this machine (needs `pip install inference`)."""
    import cv2
    from inference import InferencePipeline

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    book = TrackBook()
    writer = {"w": None}

    def on_prediction(result, video_frame):
        dets = result.get("tracked_predictions")
        if dets is not None and len(dets) and dets.tracker_id is not None:
            names = dets.data.get("class_name", [])
            for tid, cls, conf in zip(dets.tracker_id, names, dets.confidence):
                book.add(video_frame.frame_id, int(tid), str(cls), float(conf))
        img = result.get("annotated_image")
        if img is not None:
            arr = img.numpy_image
            if writer["w"] is None:
                h, w = arr.shape[:2]
                writer["w"] = cv2.VideoWriter(str(out_dir / f"{path.stem}_annotated.mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), fps or 25, (w, h))
            writer["w"].write(arr)

    pipeline = InferencePipeline.init_with_workflow(
        api_key=api_key(),
        workspace_name=WORKSPACE,
        workflow_id=VIDEO_WORKFLOW,
        video_reference=str(path),
        image_input_name="image",
        on_prediction=on_prediction,
        workflows_parameters={"confidence": conf},
    )
    pipeline.start()
    pipeline.join()
    if writer["w"] is not None:
        writer["w"].release()
    return book.write(out_dir, path.stem, min_frames, fps)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image, video, or folder")
    ap.add_argument("--out", default="hydromorph_results")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="a track must be seen in at least this many frames to be counted (default 5)")
    ap.add_argument("--engine", choices=["serverless", "local"], default="serverless",
                    help="video engine: Roboflow serverless GPU (default) or local InferencePipeline")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONF,
                    help="detection / track-activation threshold (default 0.35)")
    ap.add_argument("--image-workflow", choices=["hydromorph", "base"], default="hydromorph",
                    help="images only: hydromorph = filtered detections + annotated JPG (default); "
                         "base = raw detections from the Base Workflow, no annotated image")
    ap.add_argument("--plan", default=None,
                    help="serverless WebRTC plan, e.g. webrtc-gpu-small (default: account default)")
    a = ap.parse_args()

    src = Path(a.input)
    files = sorted(src.iterdir()) if src.is_dir() else [src]
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXT]
    if not images and not videos:
        sys.exit("No supported images or videos found.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if images:
        run = run_images_base if a.image_workflow == "base" else run_images
        run(images, out, a.confidence)
    for v in videos:
        print(f"Processing {v.name} (first run warms up the model; this can take a minute)...")
        if a.engine == "local":
            inv = run_video_local(v, out, a.min_frames, a.confidence)
        else:
            inv = run_video_serverless(v, out, a.min_frames, a.plan, a.confidence)
        print(f"{v.name} unique objects: {dict(inv) or 'none'}")
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
