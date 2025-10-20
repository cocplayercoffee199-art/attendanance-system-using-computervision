import os
import sys
import cv2
import numpy as np
import inspect

SILENTFACE_PATH = "/Users/macbook/Developer/face-recognition/Silent-Face-Anti-Spoofing"

# Force the working dir to the repo root so any './resources/...' resolves
os.chdir(SILENTFACE_PATH)

# Ensure Python can import the repo modules
if SILENTFACE_PATH not in sys.path:
    sys.path.insert(0, SILENTFACE_PATH)
SRC_PATH = os.path.join(SILENTFACE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Sanity checks: make sure detector files actually exist
proto = os.path.join(SILENTFACE_PATH, "resources", "detection_model", "deploy.prototxt")
caffemodel = os.path.join(SILENTFACE_PATH, "resources", "detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
if not (os.path.exists(proto) and os.path.exists(caffemodel)):
    raise FileNotFoundError(f"Missing detector files:\n{proto}\n{caffemodel}")

# Now import AFTER paths are set and CWD is correct
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Print where AntiSpoofPredict actually comes from (to ensure you aren’t importing a different copy)
import src.anti_spoof_predict as asp
print("[DEBUG] Using anti_spoof_predict from:", inspect.getfile(asp))


# --- Paths ---
SILENTFACE_PATH = "/Users/macbook/Developer/face-recognition/Silent-Face-Anti-Spoofing"
SRC_PATH = os.path.join(SILENTFACE_PATH, "src")

# Add repo root and src to Python path
if SILENTFACE_PATH not in sys.path:
    sys.path.insert(0, SILENTFACE_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# --- SilentFace imports ---
try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name
except ImportError:
    from anti_spoof_predict import AntiSpoofPredict
    from generate_patches import CropImage
    from utility import parse_model_name

# --- Model setup ---
_MODEL_DIR = os.path.join(SILENTFACE_PATH, "resources", "anti_spoof_models")
_PREDICTOR = AntiSpoofPredict(device_id=0)   # device_id=0 = CPU
_CROPPER = CropImage()

def anti_spoofing_score(image_rgb: np.ndarray) -> float:
    """Return the aggregated Real probability in [0,1].

    Averages the class-1 probability across all available models.
    Returns 0.0 on error.
    """
    if image_rgb is None or image_rgb.size == 0:
        return 0.0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    bbox = [0, 0, w, h]

    probs = []
    try:
        for model_name in os.listdir(_MODEL_DIR):
            if not model_name.endswith('.pth'):
                continue
            h_in, w_in, model_type, scale = parse_model_name(model_name)
            patch = _CROPPER.crop(
                org_img=image_bgr,
                bbox=bbox,
                scale=scale if scale is not None else 1.0,
                out_w=w_in,
                out_h=h_in,
                crop=(scale is not None),
            )
            model_path = os.path.join(_MODEL_DIR, model_name)
            pred = _PREDICTOR.predict(patch, model_path)  # (1,3) softmax
            probs.append(float(pred[0][1]))  # class 1 = Real
    except Exception:
        return 0.0

    if not probs:
        return 0.0
    return float(sum(probs) / len(probs))

def anti_spoofing_score_on_frame(image_bgr: np.ndarray, bbox_xywh: list[int] | None = None) -> float:
    """Return the aggregated Real probability using the original frame.

    If `bbox_xywh` is None, uses SilentFace's own detector to get a bbox.
    Expects `image_bgr` in BGR color space (as from OpenCV capture).
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0

    if bbox_xywh is None:
        try:
            bbox = _PREDICTOR.get_bbox(image_bgr)
        except Exception:
            return 0.0
    else:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return 0.0
        bbox = [int(x), int(y), int(w), int(h)]

    probs = []
    try:
        for model_name in os.listdir(_MODEL_DIR):
            if not model_name.endswith('.pth'):
                continue
            h_in, w_in, model_type, scale = parse_model_name(model_name)
            patch = _CROPPER.crop(
                org_img=image_bgr,
                bbox=bbox,
                scale=scale if scale is not None else 1.0,
                out_w=w_in,
                out_h=h_in,
                crop=(scale is not None),
            )
            model_path = os.path.join(_MODEL_DIR, model_name)
            pred = _PREDICTOR.predict(patch, model_path)
            probs.append(float(pred[0][1]))
    except Exception:
        return 0.0

    if not probs:
        return 0.0
    return float(sum(probs) / len(probs))

def anti_spoofing(image_rgb: np.ndarray) -> int:
    """
    Run SilentFace anti-spoofing on a face crop (RGB).

    Aggregates probabilities across all models in resources/anti_spoof_models
    and returns 1 for Real, 0 for Fake.
    """
    if image_rgb is None or image_rgb.size == 0:
        return 0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    bbox = [0, 0, w, h]

    # Sum predictions across all available models
    prediction = np.zeros((1, 3), dtype=np.float32)
    count = 0
    try:
        for model_name in os.listdir(_MODEL_DIR):
            if not model_name.endswith('.pth'):
                continue
            h_in, w_in, model_type, scale = parse_model_name(model_name)
            patch = _CROPPER.crop(
                org_img=image_bgr,
                bbox=bbox,
                scale=scale if scale is not None else 1.0,
                out_w=w_in,
                out_h=h_in,
                crop=(scale is not None),
            )
            model_path = os.path.join(_MODEL_DIR, model_name)
            prediction += _PREDICTOR.predict(patch, model_path)
            count += 1
    except Exception:
        return 0

    if count == 0:
        return 0
    # Keep legacy label behavior: class 1 highest
    cls = int(np.argmax(prediction))
    return 1 if cls == 1 else 0
