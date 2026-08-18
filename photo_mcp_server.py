#!/usr/bin/env python3
"""
Async MCP (Model Context Protocol) server for photo-fixation analysis of
1D/2D codes (DataMatrix, QR, 1D barcodes).

Capabilities exposed to an LLM host over ``stdio``:
    * detect_code      - YOLO object detection (ultralytics yolov8n-barcode.pt)
    * align_perspective - corner-based perspective rectification
    * decode_code      - cascaded decoding
                         WeChatQRCode -> zxing-cpp -> pylibdmtx -> pyzbar

Design notes:
    * All blocking CV/ML work is offloaded via ``asyncio.to_thread``.
    * Hardware acceleration is auto-selected (CUDA -> MPS -> CPU).
    * The OpenCV DNN backend is switched to CUDA when a GPU is present.
    * WeChat QR models are auto-downloaded into ``wechat_models/`` on first use.
    * stdout is the MCP protocol channel, so ALL logging goes to stderr / file.
      ``print()`` is never used.
"""

import sys
import os
import json
import logging
import asyncio
import threading
import tempfile
import base64
import urllib.request
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Logging  (stdout is reserved for the MCP protocol -> log to stderr + file)
# --------------------------------------------------------------------------- #
logger = logging.getLogger("photo_mcp_server")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # do not leak into the root logger / stdout

_LOG_FORMAT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_LOG_FORMAT)
logger.addHandler(_stream_handler)

try:
    _file_handler = logging.FileHandler("mcp_server.log", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_LOG_FORMAT)
    logger.addHandler(_file_handler)
except Exception as _e:  # pragma: no cover
    logger.warning("Could not create file log handler: %s", _e)

# --------------------------------------------------------------------------- #
# Heavy / optional imports (guarded so the server never crashes at startup)
# --------------------------------------------------------------------------- #
import cv2  # noqa: E402  (opencv-contrib-python provides cv2.wechat_qrcode_*)

try:
    import torch  # noqa: F401
except Exception as _e:  # pragma: no cover
    torch = None
    logger.warning("torch is not available: %s", _e)

try:
    from mcp.server.fastmcp import FastMCP
except Exception as _e:  # pragma: no cover
    logger.exception("Failed to import FastMCP: %s", _e)
    raise

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WECHAT_MODELS_DIR = "wechat_models"

# Correct, verified source: jenly1314/WeChatQRCode (the canonical repo).
# ``cv2.wechat_qrcode_WeChatQRCode`` needs only the *detect* model
# (``detectAndDecode`` performs detection + decoding in one pass); the
# *sr* (super-resolution) files are downloaded too so 4 files land in
# ``wechat_models/`` per the spec, even though the constructor uses only
# the detect pair.
_WECHAT_MODELS_BASE = (
    "https://raw.githubusercontent.com/jenly1314/WeChatQRCode/"
    "master/wechat-qrcode/src/main/assets/models"
)
WECHAT_MODEL_FILES = {
    "detect.prototxt": f"{_WECHAT_MODELS_BASE}/detect.prototxt",
    "detect.caffemodel": f"{_WECHAT_MODELS_BASE}/detect.caffemodel",
    "sr.prototxt": f"{_WECHAT_MODELS_BASE}/sr.prototxt",
    "sr.caffemodel": f"{_WECHAT_MODELS_BASE}/sr.caffemodel",
}

DETECT_PROTO = os.path.join(WECHAT_MODELS_DIR, "detect.prototxt")
DETECT_CAFFE = os.path.join(WECHAT_MODELS_DIR, "detect.caffemodel")
SR_PROTO = os.path.join(WECHAT_MODELS_DIR, "sr.prototxt")  # downloaded, optional
SR_CAFFE = os.path.join(WECHAT_MODELS_DIR, "sr.caffemodel")  # downloaded, optional

YOLO_WEIGHTS = "yolov8n-barcode.pt"

# --------------------------------------------------------------------------- #
# Runtime state
# --------------------------------------------------------------------------- #
DEVICE: str = "cpu"
_cv_backend: str = "cpu"

_wechat_detector: Any = None
_wechat_init_lock = threading.Lock()
_wechat_detect_lock = threading.Lock()

_yolo_model: Any = None
_yolo_init_lock = threading.Lock()
_yolo_predict_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Hardware acceleration
# --------------------------------------------------------------------------- #
def _detect_device() -> str:
    """Auto-select CUDA -> MPS -> CPU."""
    global DEVICE
    if torch is None:
        return DEVICE
    try:
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
    except Exception as _e:  # pragma: no cover
        logger.warning("Device detection failed, falling back to CPU: %s", _e)
        DEVICE = "cpu"
    logger.info("Selected compute device: %s", DEVICE)
    return DEVICE


def _set_cv_backend(backend: str) -> None:
    """Set the OpenCV DNN backend/target. Tolerant of failures.

    Supports the OpenCV 4.x API (``setPreferableBackend``/``setPreferableTarget``
    taking int enums) and the OpenCV 5.x API
    (``setInferenceEngineBackendType`` taking a string). On a build that
    cannot switch backends the call is skipped and the default (CPU) backend
    is used, which is fine for correctness.
    """
    global _cv_backend
    try:
        if hasattr(cv2.dnn, "setPreferableBackend"):
            if backend == "cuda":
                cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif hasattr(cv2.dnn, "setInferenceEngineBackendType"):
            cv2.dnn.setInferenceEngineBackendType(
                "CUDA" if backend == "cuda" else "OPENCV"
            )
        _cv_backend = backend
        logger.info("OpenCV DNN backend set to '%s'", backend)
    except Exception as _e:  # pragma: no cover
        logger.warning("Could not set OpenCV DNN backend '%s': %s", backend, _e)


def _cuda_available() -> bool:
    try:
        return bool(cv2.cuda.cudaEnabled())
    except Exception:
        return False


def _setup_hardware() -> None:
    _detect_device()
    _set_cv_backend("cuda" if _cuda_available() else "cpu")


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest: str, timeout: int = 60) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(dest, "wb") as fh:
            downloaded = 0
            block = 1024 * 1024  # 1 MiB
            while True:
                chunk = response.read(block)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    logger.info("  %s: %d%% (%d/%d bytes)",
                                os.path.basename(dest), pct, downloaded, total)
    logger.info("Downloaded %s", dest)


def _ensure_wechat_models() -> None:
    os.makedirs(WECHAT_MODELS_DIR, exist_ok=True)
    for name, url in WECHAT_MODEL_FILES.items():
        dest = os.path.join(WECHAT_MODELS_DIR, name)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            try:
                _download_file(url, dest)
            except Exception as _e:  # pragma: no cover
                logger.error("Failed to download %s: %s", url, _e)
                raise


def _create_wechat_detector():
    _ensure_wechat_models()

    # The WeChatQRCode constructor signature varies across OpenCV versions:
    # 6 keyword args in 4.5, only 2 positional args in 4.14+, 2 args (but a
    # different DNN backend) in 5.0. Try progressively simpler forms, and for
    # each backend, until one succeeds.
    arg_sets = [
        dict(conf_threshold=0.5, nms_threshold=0.3,
             top_k=1000, input_size=(320, 320)),
        dict(conf_threshold=0.5, nms_threshold=0.3),
        dict(),
    ]
    candidates = ["cuda", "cpu"] if _cuda_available() else ["cpu"]
    last_error = None
    for backend in candidates:
        _set_cv_backend(backend)
        for kwargs in arg_sets:
            try:
                # Pass the super-resolution (sr) models too so the neural
                # up-scale path is enabled; the **kwargs below keeps the
                # constructor tolerant of version differences.
                detector = cv2.wechat_qrcode_WeChatQRCode(
                    DETECT_PROTO, DETECT_CAFFE,
                    SR_PROTO, SR_CAFFE, **kwargs
                )
                logger.info(
                    "WeChatQRCode initialised (backend '%s', %d extra args)",
                    backend, len(kwargs),
                )
                return detector
            except Exception as _e:  # pragma: no cover
                last_error = _e
                logger.warning(
                    "WeChatQRCode init failed (backend '%s', %d extra args): %s",
                    backend, len(kwargs), _e,
                )
    logger.error("WeChatQRCode could not be initialised: %s", last_error)
    raise last_error


def _load_wechat_detector():
    global _wechat_detector
    if _wechat_detector is not None:
        return _wechat_detector
    with _wechat_init_lock:
        if _wechat_detector is not None:
            return _wechat_detector
        _wechat_detector = _create_wechat_detector()
        return _wechat_detector


def _load_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    with _yolo_init_lock:
        if _yolo_model is not None:
            return _yolo_model
        from ultralytics import YOLO

        logger.info("Loading YOLO model '%s' on device '%s'", YOLO_WEIGHTS, DEVICE)
        _yolo_model = YOLO(YOLO_WEIGHTS)
        # Constraint: move the model onto the device (tolerant of API gaps).
        try:
            _yolo_model = _yolo_model.to(DEVICE)
            logger.info("YOLO model moved to '%s'", DEVICE)
        except Exception as _e:  # pragma: no cover
            logger.warning(
                "Could not move YOLO model to '%s' "
                "(device will be passed to predict instead): %s",
                DEVICE, _e,
            )
        return _yolo_model


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _read_image(path: str):
    """Load an image from a local path, an HTTP/HTTPS URL, or a Base64
    Data URI. Returns a BGR numpy array; raises ValueError on failure.

    Because the MCP server may run remotely (e.g. behind vLLM --tool-server),
    ``path`` is no longer assumed to be a local file: it can be an HTTP/HTTPS
    URL or a Base64 / Data URI. All three cases are decoded in memory via
    ``cv2.imdecode``; the local case keeps using ``cv2.imread``.
    """
    # 1. Remote HTTP/HTTPS URL: download into memory, decode via cv2.imdecode.
    if path.startswith("http://") or path.startswith("https://"):
        try:
            with urllib.request.urlopen(path, timeout=60) as response:
                raw = response.read()
        except Exception as _e:  # pragma: no cover
            raise ValueError(f"Cannot download image from '{path}': {_e}")
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image from URL '{path}'")
        return img

    # 2. Base64 / Data URI (e.g. "data:image/jpeg;base64,..." or a bare
    #    base64 string, detectable by "data:image" or image markers like
    #    "/9j/"). Strip the "data:...;base64," prefix if a comma is present,
    #    decode, then read via cv2.imdecode.
    if "data:image" in path.lower() or "/9j/" in path:
        b64 = path.split(",", 1)[1] if "," in path else path
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as _e:  # pragma: no cover
            raise ValueError(f"Cannot base64-decode image data: {_e}")
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from Base64 data")
        return img

    # 3. Local file path.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image at '{path}' (missing or corrupted)")
    return img


def _crop_roi(image, bbox):
    """Crop a region given [x1, y1, x2, y2]; returns full image on bad bbox."""
    if not bbox:
        return image
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except Exception as _e:  # pragma: no cover
        logger.warning("Invalid bbox '%s': %s", bbox, _e)
        return image
    h, w = image.shape[:2]
    x1 = int(max(0, min(w, round(x1))))
    y1 = int(max(0, min(h, round(y1))))
    x2 = int(max(0, min(w, round(x2))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 <= x1 or y2 <= y1:
        logger.warning("Empty ROI bbox: %s", bbox)
        return image
    return image[y1:y2, x1:x2].copy()


def _order_points(pts):
    """Order 4 points into [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.asarray(pts, dtype="float32")
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


# --------------------------------------------------------------------------- #
# Decoders  (strict cascade order)
# --------------------------------------------------------------------------- #
def _to_text(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    if isinstance(data, str):
        return data
    return str(data)


def _wechat_decode(image):
    detector = _load_wechat_detector()
    with _wechat_detect_lock:
        packed = detector.detectAndDecode(image)
    # ``detectAndDecode`` returns a 2-tuple, but the order of
    # (codes, results) varies across OpenCV versions (4.14 returns the
    # decoded strings first, older versions the boxes first). The decoded
    # text is the element whose items are str/bytes; the detection boxes
    # are numpy arrays and are skipped. This makes it robust to ordering.
    decoded = []
    seen = set()
    for part in packed:
        if part is None:
            continue
        entries = part if isinstance(part, (list, tuple)) else [part]
        for entry in entries:
            if isinstance(entry, np.ndarray):
                continue  # detection box, not a decoded string
            text = _to_text(entry).strip()
            if text and text not in seen:
                seen.add(text)
                decoded.append({"format": "QR", "text": text})
    return decoded


def _zxing_decode(image):
    try:
        from zxingcpp import read_barcodes
    except Exception:
        from zxingcpp import readbarcodes as read_barcodes  # type: ignore
    results = read_barcodes(image)
    decoded = []
    for r in results:
        text = str(getattr(r, "text", "")).strip()
        if not text:
            continue
        fmt = getattr(r, "format", "UNKNOWN")
        decoded.append({"format": str(fmt), "text": text})
    return decoded


def _libdmtx_decode(image):
    import pylibdmtx
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mandatory binarisation before pylibdmtx.decode(): libdmtx decoders
    # work on a binary image, so an adaptiveThreshold step sharply improves
    # DataMatrix decoding on noisy / low-contrast captures.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
    )
    results = pylibdmtx.decode(binary)
    decoded = []
    for r in results:
        text = _to_text(getattr(r, "data", "")).strip()
        if text:
            decoded.append({"format": "DATAMATRIX", "text": text})
    return decoded


def _pyzbar_decode(image):
    from pyzbar import pyzbar
    results = pyzbar.decode(image)
    decoded = []
    for r in results:
        text = _to_text(getattr(r, "data", "")).strip()
        if not text:
            continue
        fmt = getattr(r, "type", "UNKNOWN")
        decoded.append({"format": str(fmt), "text": text})
    return decoded


def _cascade_decode(image):
    """Strict cascade: WeChatQRCode -> zxing-cpp -> pylibdmtx -> pyzbar.

    Stops at the first stage that yields a non-empty result.
    """
    # 1. WeChatQRCode
    try:
        decoded = _wechat_decode(image)
        if decoded:
            logger.info("Decoded by WeChatQRCode: %s", decoded)
            return decoded
    except Exception as _e:  # pragma: no cover
        logger.warning("WeChatQRCode stage failed: %s", _e)

    # 2. zxing-cpp
    try:
        decoded = _zxing_decode(image)
        if decoded:
            logger.info("Decoded by zxing-cpp: %s", decoded)
            return decoded
    except Exception as _e:  # pragma: no cover
        logger.warning("zxing-cpp stage failed: %s", _e)

    # 3. pylibdmtx
    try:
        decoded = _libdmtx_decode(image)
        if decoded:
            logger.info("Decoded by pylibdmtx: %s", decoded)
            return decoded
    except Exception as _e:  # pragma: no cover
        logger.warning("pylibdmtx stage failed: %s", _e)

    # 4. pyzbar
    try:
        decoded = _pyzbar_decode(image)
        if decoded:
            logger.info("Decoded by pyzbar: %s", decoded)
            return decoded
    except Exception as _e:  # pragma: no cover
        logger.warning("pyzbar stage failed: %s", _e)

    logger.warning("Cascade decode produced no results")
    return []


# --------------------------------------------------------------------------- #
# Core (blocking) routines - executed inside worker threads
# --------------------------------------------------------------------------- #
def _run_yolo_detect(model, image_path: str):
    with _yolo_predict_lock:
        results = model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            device=DEVICE,
            verbose=False,
        )
    detections = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ],
                "confidence": round(float(confs[i]), 4),
            })
    return {"status": "success", "detections": detections}


def _run_align(image_path: str, corners, output_size: int = 300):
    image = _read_image(image_path)
    arr = np.array(corners, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    ordered = _order_points(arr)

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    base = os.path.splitext(image_path)[0]
    # Save to the system temp dir so the project tree is not littered with
    # *_aligned.png artifacts.
    saved_path = os.path.join(tempfile.gettempdir(), f"{base}_aligned.png")
    if not cv2.imwrite(saved_path, warped):
        raise RuntimeError(f"Could not write aligned image to '{saved_path}'")
    return {"status": "success", "saved_path": saved_path}


def _run_decode(image_path: str, bbox=None):
    image = _read_image(image_path)
    roi = _crop_roi(image, bbox)
    decoded = _cascade_decode(roi)
    return {"status": "success", "results": decoded}


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
# Host/port are set here (FastMCP constructor), NOT in run(): in mcp 1.29.0
# run() only takes transport/mount_path; the uvicorn server reads
# self.settings.host/port. Bind to 0.0.0.0 so a remote vLLM instance can reach
# the SSE endpoint.
mcp = FastMCP("photo-mcp-server", host="0.0.0.0", port=8000)


@mcp.tool()
async def detect_code(image_path: str) -> str:
    """Detect barcode / QR / DataMatrix objects with YOLO.

    Args:
        image_path: Local file path, HTTP/HTTPS URL, or Base64 Data URI.

    Returns:
        JSON string: {"status": "success",
        "detections": [{"bbox": [x1, y1, x2, y2], "confidence": 0.99}, ...]}
    """
    try:
        model = await asyncio.to_thread(_load_yolo_model)
        result = await asyncio.to_thread(_run_yolo_detect, model, image_path)
        logger.info("detect_code(%s) -> %d detections",
                    image_path, len(result["detections"]))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        logger.exception("detect_code failed for %s", image_path)
        return json.dumps({"status": "error", "message": str(e)},
                          ensure_ascii=False)


@mcp.tool()
async def align_perspective(image_path: str, corners: list, output_size: int = 300) -> str:
    """Rectify a code region using its 4 corners via a perspective transform.

    Args:
        image_path: Local file path, HTTP/HTTPS URL, or Base64 Data URI.
        corners: Four corner points as [[x, y], [x, y], [x, y], [x, y]]
            (any order; sorted internally).
        output_size: Side length (pixels) of the square output.

    Returns:
        JSON string: {"status": "success", "saved_path": "..."}
    """
    try:
        result = await asyncio.to_thread(_run_align, image_path, corners, output_size)
        logger.info("align_perspective(%s) -> %s", image_path, result["saved_path"])
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        logger.exception("align_perspective failed for %s", image_path)
        return json.dumps({"status": "error", "message": str(e)},
                          ensure_ascii=False)


@mcp.tool()
async def decode_code(image_path: str, bbox: Optional[list] = None) -> str:
    """Decode 1D/2D codes via a strict cascade
    (WeChatQRCode -> zxing-cpp -> pylibdmtx -> pyzbar).

    Args:
        image_path: Local file path, HTTP/HTTPS URL, or Base64 Data URI.
        bbox: Optional ROI [x1, y1, x2, y2]; the image is cropped first.

    Returns:
        JSON string: {"status": "success",
        "results": [{"format": "DATAMATRIX", "text": "..."}, ...]}
    """
    try:
        result = await asyncio.to_thread(_run_decode, image_path, bbox)
        logger.info("decode_code(%s) -> %d results",
                    image_path, len(result["results"]))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        logger.exception("decode_code failed for %s", image_path)
        return json.dumps({"status": "error", "message": str(e)},
                          ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _startup() -> None:
    _setup_hardware()
    logger.info("photo-mcp-server ready (device=%s, cv_backend=%s)",
                DEVICE, _cv_backend)


if __name__ == "__main__":
    try:
        _startup()
    except Exception as _e:  # pragma: no cover
        logger.exception("Startup failed: %s", _e)
    # Run as a standalone network service so a remote vLLM instance can reach
    # it via the Responses API (--tool-server). SSE is used instead of stdio;
    # the host/port were set in the FastMCP constructor above (run() only
    # accepts transport/mount_path in mcp 1.29.0).
    mcp.run(transport="sse")