import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from skimage import io, img_as_ubyte, exposure
from skimage.filters import rank, gaussian, sobel, gabor, prewitt
from skimage.morphology import disk
from skimage.feature import (
    graycomatrix,
    graycoprops,
    hessian_matrix,
    hessian_matrix_eigvals,
)

import mlflow


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to mounted tumor_images_raw (folder with yes/ and no/).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output folder for features.parquet.",
    )
    return parser.parse_args()


def to_8bit(image: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] then convert to uint8."""
    img_rescaled = exposure.rescale_intensity(image, in_range="image", out_range=(0, 1))
    return img_as_ubyte(img_rescaled)


def compute_glcm_features(image_u8: np.ndarray, prefix: str) -> dict:
    """
    Compute GLCM features for a uint8 image.
    Angles: 0°, 45°, 90°, 135°
    Properties: contrast, dissimilarity, homogeneity, ASM, energy, correlation
    """
    distances = [1]
    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    angle_degs = [0, 45, 90, 135]
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

    glcm = graycomatrix(
        image_u8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in props:
        values = graycoprops(glcm, prop)[0]  # shape: (len(angles),)
        for val, deg in zip(values, angle_degs):
            key = f"{prefix}_{prop}_{deg}"
            feats[key] = float(val)
    return feats


def process_image(item):
    """
    Worker for one image.
    item: (path_str, label_str) where label_str is 'yes' or 'no'.
    """
    path_str, label_str = item
    path = Path(path_str)

    # Load grayscale
    gray = io.imread(str(path), as_gray=True).astype(np.float32)

    # Base 8-bit grayscale
    gray_ubyte = to_8bit(gray)

    # Filters
    filters = {}

    # Original
    filters["orig"] = gray_ubyte

    # Entropy (needs uint8 + structuring element)
    entropy_img = rank.entropy(gray_ubyte, disk(3))
    filters["entropy"] = entropy_img

    # Gaussian
    gaussian_img = gaussian(gray, sigma=1)
    filters["gaussian"] = gaussian_img

    # Sobel
    sobel_img = sobel(gray)
    filters["sobel"] = sobel_img

    # Gabor (magnitude from real+imag)
    gabor_real, gabor_imag = gabor(gray, frequency=0.6)
    gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)
    filters["gabor"] = gabor_mag

    # Hessian (largest eigenvalue)
    H_elems = hessian_matrix(gray, sigma=1)
    eigvals = hessian_matrix_eigvals(H_elems)
    hessian_img = eigvals[0]
    filters["hessian"] = hessian_img

    # Prewitt
    prewitt_img = prewitt(gray)
    filters["prewitt"] = prewitt_img

    # Feature row: id + label
    feature_row = {
        "image_id": path.stem,
        "label": 1 if label_str == "yes" else 0,
    }

    # GLCM features for each filter image
    for name, img in filters.items():
        if img.dtype != np.uint8:
            img_u8 = to_8bit(img)
        else:
            img_u8 = img
        feats = compute_glcm_features(img_u8, prefix=name)
        feature_row.update(feats)

    return feature_row


def main():
    args = parse_args()
    input_root = Path(args.input_path)
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Collect images and labels
    items = []
    for label in ("yes", "no"):
        label_dir = input_root / label
        if not label_dir.is_dir():
            continue
        for p in label_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                items.append((str(p), label))

    if not items:
        raise RuntimeError(f"No images found under {input_root}")

    # Multiprocessing
    cpu_count = os.cpu_count() or 1
    max_workers = min(cpu_count, len(items)) or 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(process_image, items))

    df = pd.DataFrame(rows)

    # Basic stats
    num_images = len(df)
    num_features = df.shape[1] - 2  # minus image_id and label
    extraction_time_seconds = time.time() - start_time
    compute_sku = os.environ.get("AZUREML_COMPUTE", "unknown")

    # Output parquet
    out_file = output_root / "features.parquet"
    df.to_parquet(out_file, index=False)

    # Log to Azure ML (MLflow)
    mlflow.log_metric("num_images", num_images)
    mlflow.log_metric("num_features", num_features)
    mlflow.log_metric("extraction_time_seconds", extraction_time_seconds)
    mlflow.log_param("compute_sku", compute_sku)

    # Also print for plain logs
    print(f"num_images={num_images}")
    print(f"num_features={num_features}")
    print(f"extraction_time_seconds={extraction_time_seconds:.3f}")
    print(f"compute_sku={compute_sku}")
    print(f"output_parquet={out_file}")


if __name__ == "__main__":
    main()
