import base64
import io
import json
import os

import joblib
import numpy as np
from PIL import Image

from image_features import compute_image_features_from_array

MODEL = None
SELECTED_FEATURES = None


def _load_model_and_metadata():
    global MODEL, SELECTED_FEATURES

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")

    MODEL = joblib.load(model_path)

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        SELECTED_FEATURES = meta.get("selected_features")
    else:
        SELECTED_FEATURES = None


def init():
    _load_model_and_metadata()


def _decode_image_from_request(data: dict) -> np.ndarray:
    """
    Expected input formats:
    1) { "image_base64": "<base64_string>" }
    2) { "image_bytes": "<base64_string>" }  # alias
    """
    if "image_base64" in data:
        b64 = data["image_base64"]
    elif "image_bytes" in data:
        b64 = data["image_bytes"]
    else:
        raise ValueError("Request JSON must contain 'image_base64' or 'image_bytes'")

    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    return arr


def run(raw_data):
    global MODEL, SELECTED_FEATURES

    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        img_array = _decode_image_from_request(data)

        # Compute features with EXACT same logic as Silver
        feat_dict = compute_image_features_from_array(img_array)

        if SELECTED_FEATURES is None:
            # Fallback: use all features in sorted key order
            feature_names = sorted(feat_dict.keys())
        else:
            feature_names = SELECTED_FEATURES

        # Build feature vector in correct order
        vec = [feat_dict[name] for name in feature_names]
        X = np.array(vec, dtype=float).reshape(1, -1)

        # Predict
        proba = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred = MODEL.classes_[pred_idx]
            pred_prob = float(proba[pred_idx])
        else:
            pred = MODEL.predict(X)[0]
            pred_prob = None

        # Map to "tumor" / "no_tumor"
        label = str(pred)
        if isinstance(pred, (np.bool_, bool)):
            label = "tumor" if pred else "no_tumor"
        elif str(pred) in ["1", "tumor", "yes", "True"]:
            label = "tumor"
        elif str(pred) in ["0", "no_tumor", "no", "False"]:
            label = "no_tumor"

        response = {"prediction": label}
        if pred_prob is not None:
            response["probability"] = pred_prob

        return response

    except Exception as e:
        return {"error": str(e)}