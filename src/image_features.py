import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.filters import gaussian, sobel, prewitt, gabor
from skimage.filters.rank import entropy as entropy_filter
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops


def _glcm_features_matrix(img, name_prefix):
    """Compute GLCM features for one 2D greyscale image."""
    img_uint8 = img_as_ubyte(img)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = greycomatrix(
        img_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    feats = {}

    for prop in props:
        vals = greycoprops(glcm, prop)[0]  # shape (1, n_angles) -> (n_angles,)
        for i, angle in enumerate(["0", "45", "90", "135"]):
            feats[f"{name_prefix}_glcm_{prop}_{angle}"] = float(vals[i])

    return feats


def compute_image_features_from_array(img_array):
    """
    Input: img_array: 2D or 3D numpy array
    Output: dict {feature_name: value}
    This **must** match your Silver-layer feature logic and naming.
    """

    # Convert to grayscale float [0,1]
    if img_array.ndim == 3:
        gray = color.rgb2gray(img_array)
    else:
        gray = img_array.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    features = {}

    # ------------------------------------------------------------------
    # REPLACE / EXTEND THIS BLOCK TO MATCH YOUR SILVER LOGIC EXACTLY.
    # Copy the filters+GLCM you used in src/extract_features.py.
    # ------------------------------------------------------------------

    # 1) Entropy (rank filter needs uint8 + structuring element)
    gray_u8 = img_as_ubyte(gray)
    ent = entropy_filter(gray_u8, disk(3))
    features["entropy_mean"] = float(np.mean(ent))
    features["entropy_std"] = float(np.std(ent))

    # 2) Gaussian
    gau = gaussian(gray, sigma=1)
    features["gaussian_mean"] = float(np.mean(gau))
    features["gaussian_std"] = float(np.std(gau))

    # 3) Sobel
    sob = sobel(gray)
    features["sobel_mean"] = float(np.mean(sob))
    features["sobel_std"] = float(np.std(sob))

    # 4) Prewitt
    pre = prewitt(gray)
    features["prewitt_mean"] = float(np.mean(pre))
    features["prewitt_std"] = float(np.std(pre))

    # 5) Gabor (magnitude only)
    gab_real, gab_imag = gabor(gray, frequency=0.2)
    gab_mag = np.sqrt(gab_real ** 2 + gab_imag ** 2)
    features["gabor_mean"] = float(np.mean(gab_mag))
    features["gabor_std"] = float(np.std(gab_mag))

    # GLCM features on base gray
    features.update(_glcm_features_matrix(gray, "gray"))

    # ------------------------------------------------------------------
    # END OF TEMPLATE. Ensure the resulting feature names (keys) match
    # the columns in your Silver parquet exactly.
    # ------------------------------------------------------------------

    return features