"""
Client for the Roboflow Base Workflow "river-feature-object-detection-wbzft".

The Base Workflow is the plain detector Roboflow created for this project: one
object-detection step, no per-class filter, no annotated image. Use it when you
only need detections (class, confidence, box). For filtered detections plus an
annotated JPG, use the `hydromorph-image-detection` Workflow instead.

Workflow definition (read with the Roboflow MCP `workflows_get`):
    inputs : image
             confidence=0.4, iou_threshold=0.3, class_agnostic_nms=False,
             max_detections=1000, model_id="river-feature-object-detection-wbzft/1"
    outputs: predictions, inference_id, model_id

Usage:
    from base_workflow import detect_river_features
    result = detect_river_features("photo.jpg", confidence=0.35)
    print(result.counts())                      # {'fallen_tree': 2, ...}
    for d in result.detections:
        print(d.class_name, round(d.confidence, 2), d.xyxy)

Requires inference-sdk >= 1.7 (header API-key transport) and ROBOFLOW_API_KEY
in the environment or a .env file.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple, Union

WORKSPACE = "rs-workspace-o0dgc"
WORKFLOW_ID = "river-feature-object-detection-wbzft"
API_URL = "https://serverless.roboflow.com"
IMAGE_INPUT = "image"

# Output names declared by the published Workflow. Parsing reads only these.
OUT_PREDICTIONS = "predictions"
OUT_INFERENCE_ID = "inference_id"
OUT_MODEL_ID = "model_id"
EXPECTED_OUTPUTS: FrozenSet[str] = frozenset({OUT_PREDICTIONS, OUT_INFERENCE_ID, OUT_MODEL_ID})

DEFAULT_TIMEOUT_S = 60.0   # per attempt; first call after idle can take ~20-30 s (cold start)
DEFAULT_RETRIES = 2        # extra attempts after the first, for transient failures only
DEFAULT_BACKOFF_S = 1.0    # 1 s, 2 s, 4 s ...
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

ImageInput = Union[str, "os.PathLike[str]", Any]  # path, https URL, numpy array (BGR) or PIL image


# ----------------------------------------------------------------- errors
class BaseWorkflowError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingApiKeyError(BaseWorkflowError):
    """ROBOFLOW_API_KEY is not set."""


class SdkVersionError(BaseWorkflowError):
    """inference-sdk is missing or too old."""


class InvalidImageInputError(BaseWorkflowError, ValueError):
    """The image cannot be sent (missing file, http:// URL, undecodable data)."""


class InvalidParameterError(BaseWorkflowError, ValueError):
    """A Workflow parameter is out of range or has the wrong type."""


class WorkflowTimeoutError(BaseWorkflowError):
    """No response within the per-attempt timeout (after all retries)."""


class WorkflowConnectionError(BaseWorkflowError):
    """Could not reach Roboflow (after all retries)."""


class WorkflowHTTPError(BaseWorkflowError):
    """Roboflow answered with an HTTP error status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class WorkflowAuthError(WorkflowHTTPError):
    """HTTP 401/403: key invalid, revoked, or not in this workspace."""


class WorkflowNotFoundError(WorkflowHTTPError):
    """HTTP 404: workspace or workflow slug not found, or not published."""


class WorkflowRequestError(WorkflowHTTPError):
    """Other HTTP 4xx: the request itself was rejected (bad input or parameter)."""


class WorkflowServerError(WorkflowHTTPError):
    """HTTP 429/5xx after all retries."""


class WorkflowResponseError(BaseWorkflowError):
    """The response does not have the shape this Workflow declares."""


_TRANSIENT = (WorkflowTimeoutError, WorkflowConnectionError, WorkflowServerError)


# ----------------------------------------------------------------- results
@dataclass(frozen=True)
class Detection:
    class_name: str
    class_id: Optional[int]
    confidence: float
    x: float        # box centre, pixels
    y: float
    width: float
    height: float
    detection_id: Optional[str]

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        hw, hh = self.width / 2, self.height / 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name, "class_id": self.class_id,
            "confidence": self.confidence,
            "x": self.x, "y": self.y, "width": self.width, "height": self.height,
            "detection_id": self.detection_id,
        }


@dataclass(frozen=True)
class RiverFeatureResult:
    detections: Tuple[Detection, ...]
    image_width: Optional[int]
    image_height: Optional[int]
    inference_id: Optional[str]
    model_id: Optional[str]
    output_keys: FrozenSet[str]  # output names the server actually returned

    def counts(self) -> Dict[str, int]:
        return dict(Counter(d.class_name for d in self.detections))


# ----------------------------------------------------------------- config
def load_api_key() -> str:
    """ROBOFLOW_API_KEY from the environment, else from .env (cwd, then this folder)."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
    raise MissingApiKeyError(
        "Set ROBOFLOW_API_KEY (environment variable or .env file). "
        "Private key: app.roboflow.com/settings/api"
    )


@lru_cache(maxsize=4)
def _build_client(api_key: str):
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError as e:
        raise SdkVersionError('inference-sdk is not installed: pip install "inference-sdk>=1.7"') from e
    try:
        # Key goes in "Authorization: Bearer"; never in the URL or body.
        # SDK-internal retries are off so this module owns the retry policy.
        config = InferenceConfiguration(api_key_transport="header", workflow_run_retries_enabled=False)
    except TypeError as e:
        raise SdkVersionError(
            'Installed inference-sdk has no header auth: pip install -U "inference-sdk>=1.7"'
        ) from e
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key).configure(config)


# ----------------------------------------------------------------- validation
def _check_image(image: ImageInput) -> Any:
    if isinstance(image, (str, os.PathLike)):
        s = os.fspath(image)
        low = s.lower()
        if low.startswith("http://"):
            raise InvalidImageInputError("Roboflow serverless rejects http:// image URLs; use https:// or a local file.")
        if low.startswith("https://"):
            return s
        p = Path(s)
        if not p.is_file():
            raise InvalidImageInputError(f"Image file not found: {p}")
        return str(p)
    if image is None:
        raise InvalidImageInputError("image is None")
    return image  # numpy array / PIL image: the SDK base64-encodes it


def _build_parameters(confidence, iou_threshold, class_agnostic_nms, max_detections, model_id) -> Dict[str, Any]:
    """Only parameters the caller set; everything else falls back to the Workflow defaults."""
    params: Dict[str, Any] = {}
    for name, value in (("confidence", confidence), ("iou_threshold", iou_threshold)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise InvalidParameterError(f"{name} must be a number in [0, 1], got {value!r}")
        params[name] = float(value)
    if class_agnostic_nms is not None:
        if not isinstance(class_agnostic_nms, bool):
            raise InvalidParameterError(f"class_agnostic_nms must be bool, got {class_agnostic_nms!r}")
        params["class_agnostic_nms"] = class_agnostic_nms
    if max_detections is not None:
        if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections < 1:
            raise InvalidParameterError(f"max_detections must be a positive int, got {max_detections!r}")
        params["max_detections"] = max_detections
    if model_id is not None:
        if not isinstance(model_id, str) or "/" not in model_id:
            raise InvalidParameterError(f"model_id must look like 'project/version', got {model_id!r}")
        params["model_id"] = model_id
    return params


# ----------------------------------------------------------------- transport helpers
def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run fn in a daemon thread; the SDK has no per-request timeout of its own."""
    box: Dict[str, Any] = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in caller thread
            box["error"] = e

    t = threading.Thread(target=target, name="roboflow-base-workflow", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise WorkflowTimeoutError(f"No response from Roboflow within {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _classify(err: BaseException) -> Optional[BaseWorkflowError]:
    """Map SDK / requests exceptions to this module's typed errors (None = unknown)."""
    if isinstance(err, BaseWorkflowError):
        return err
    try:
        from inference_sdk.http.errors import (
            EncodingError, HTTPCallErrorError, HTTPClientError, InvalidInputFormatError,
        )
    except ImportError:  # pragma: no cover
        return None
    if isinstance(err, HTTPCallErrorError):
        code = err.status_code
        detail = err.api_message or err.description
        where = f"{WORKSPACE}/{WORKFLOW_ID}"
        if code in (401, 403):
            return WorkflowAuthError(code, f"HTTP {code}: API key rejected for {where}. "
                                           f"Check ROBOFLOW_API_KEY belongs to workspace {WORKSPACE}. ({detail})")
        if code == 404:
            return WorkflowNotFoundError(code, f"HTTP 404: workflow {where} not found or not published. ({detail})")
        if code in RETRYABLE_STATUS or code >= 500:
            return WorkflowServerError(code, f"HTTP {code} from Roboflow serverless. ({detail})")
        return WorkflowRequestError(code, f"HTTP {code}: request rejected. ({detail})")
    if isinstance(err, (InvalidInputFormatError, EncodingError)):
        return InvalidImageInputError(str(err))
    if isinstance(err, HTTPClientError):  # SDK wraps connection failures in the base class
        return WorkflowConnectionError(str(err))
    try:
        import requests
        if isinstance(err, (requests.ConnectionError, requests.Timeout)):
            return WorkflowConnectionError(str(err))
    except ImportError:  # pragma: no cover
        pass
    if isinstance(err, ConnectionError):
        return WorkflowConnectionError(str(err))
    return None


def _parse(raw: Any) -> RiverFeatureResult:
    if not isinstance(raw, list) or not raw:
        raise WorkflowResponseError(f"Expected a non-empty list (one entry per image), got {type(raw).__name__}")
    entry = raw[0]
    if not isinstance(entry, dict):
        raise WorkflowResponseError(f"Expected a dict of outputs, got {type(entry).__name__}")
    keys = frozenset(entry)
    if OUT_PREDICTIONS not in entry:
        raise WorkflowResponseError(
            f"Output '{OUT_PREDICTIONS}' missing; server returned {sorted(keys)}. "
            "The published Workflow may have changed."
        )
    block = entry[OUT_PREDICTIONS]
    if isinstance(block, dict):        # serialised detections: {"image": {...}, "predictions": [...]}
        img = block.get("image") or {}
        items = block.get("predictions") or []
    elif isinstance(block, list):
        img, items = {}, block
    elif block is None:
        img, items = {}, []
    else:
        raise WorkflowResponseError(f"'{OUT_PREDICTIONS}' has unexpected type {type(block).__name__}")

    detections = []
    for i, p in enumerate(items):
        try:
            detections.append(Detection(
                class_name=str(p["class"]),
                class_id=int(p["class_id"]) if p.get("class_id") is not None else None,
                confidence=float(p["confidence"]),
                x=float(p["x"]), y=float(p["y"]),
                width=float(p["width"]), height=float(p["height"]),
                detection_id=p.get("detection_id"),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise WorkflowResponseError(f"Detection #{i} is malformed ({type(e).__name__}: {e})") from None

    def _opt_int(v):
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def _opt_str(v):
        return v if isinstance(v, str) else None

    return RiverFeatureResult(
        detections=tuple(detections),
        image_width=_opt_int(img.get("width")) if isinstance(img, dict) else None,
        image_height=_opt_int(img.get("height")) if isinstance(img, dict) else None,
        inference_id=_opt_str(entry.get(OUT_INFERENCE_ID)),
        model_id=_opt_str(entry.get(OUT_MODEL_ID)),
        output_keys=keys,
    )


# ----------------------------------------------------------------- public API
def detect_river_features(
    image: ImageInput,
    *,
    confidence: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    class_agnostic_nms: Optional[bool] = None,
    max_detections: Optional[int] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> RiverFeatureResult:
    """Run the Base Workflow on one image and return typed detections.

    image: local path, https URL, numpy array (BGR) or PIL image.
    Parameters left as None use the Workflow defaults (confidence 0.4, IoU 0.3,
    class-agnostic NMS off, max 1000 detections, model river-feature-object-detection-wbzft/1).

    Retries timeouts, connection errors and HTTP 408/429/5xx with exponential
    backoff. Raises a BaseWorkflowError subclass on failure.
    """
    image = _check_image(image)
    params = _build_parameters(confidence, iou_threshold, class_agnostic_nms, max_detections, model_id)
    if timeout_s <= 0 or retries < 0:
        raise InvalidParameterError("timeout_s must be > 0 and retries >= 0")
    client = _build_client(api_key or load_api_key())

    def call():
        return client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={IMAGE_INPUT: image},
            parameters=params,
        )

    for attempt in range(retries + 1):
        try:
            raw = _call_with_timeout(call, timeout_s)
            break
        except Exception as e:  # noqa: BLE001 - classified below
            err = _classify(e)
            if err is None:
                raise
            if not isinstance(err, _TRANSIENT) or attempt == retries:
                if err is e:
                    raise
                raise err from e
            time.sleep(backoff_s * (2 ** attempt))
    return _parse(raw)
