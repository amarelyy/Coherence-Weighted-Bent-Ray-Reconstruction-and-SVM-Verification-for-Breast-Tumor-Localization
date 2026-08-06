"""
src/blob_detection.py

Multi-scale LoG blob enhancement, Otsu thresholding, largest connected
component, intensity-weighted centroid with front-surface bias correction.
"""

import numpy as np
from scipy.ndimage import label, gaussian_laplace


def select_blob_threshold_adaptive(img):
    """Otsu threshold if scikit-image is available, else 90th-percentile fallback."""
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(img)
    except Exception:
        return np.percentile(img, 90)


def select_blob_mask_log(img, sigma_list=(3, 4, 5, 6, 7)):
    """
    Multi-scale LoG blob enhancement before thresholding.
    Sigma range tuned for UM-BMID tumor sizes (7.5-15mm radius).
    """
    try:
        response = np.zeros_like(img)
        for s in sigma_list:
            blob_r = -gaussian_laplace(img, sigma=s) * (s ** 2)
            response = np.maximum(response, blob_r)
        response = response / (response.max() + 1e-12)
        thresh = select_blob_threshold_adaptive(response)
        binary_mask = response >= thresh
    except Exception:
        response = img
        thresh = select_blob_threshold_adaptive(img)
        binary_mask = img >= thresh

    labeled_mask, n_blobs = label(binary_mask)
    if n_blobs > 0:
        sizes = [(labeled_mask == i).sum() for i in range(1, n_blobs + 1)]
        tumor_mask = labeled_mask == (np.argmax(sizes) + 1)
    else:
        tumor_mask = binary_mask
    return tumor_mask, response, thresh


def weighted_centroid(tumor_mask, enhanced_img, x_axis, y_axis, bias_alpha=0.3):
    """
    Intensity-weighted centroid blended with geometric center to correct
    front-surface bias. alpha=0 = pure intensity (biased), alpha=1 = pure geo.
    """
    ys, xs = np.where(tumor_mask)
    if len(xs) > 0:
        weights = enhanced_img[ys, xs]
        iw_x = np.average(x_axis[xs], weights=weights)
        iw_y = np.average(y_axis[ys], weights=weights)
        geo_x = np.mean(x_axis[xs])
        geo_y = np.mean(y_axis[ys])
        peak_x = (1.0 - bias_alpha) * iw_x + bias_alpha * geo_x
        peak_y = (1.0 - bias_alpha) * iw_y + bias_alpha * geo_y
    else:
        fb = np.unravel_index(np.argmax(enhanced_img), enhanced_img.shape)
        peak_y, peak_x = y_axis[fb[0]], x_axis[fb[1]]
    return peak_x, peak_y


def extract_blob_candidate(img, x_axis, y_axis, use_log=True, bias_alpha=0.3):
    """One-call blob extraction + localization with bias correction."""
    if use_log:
        tumor_mask, response, thresh = select_blob_mask_log(img)
    else:
        thresh = select_blob_threshold_adaptive(img)
        binary_mask = img >= thresh
        labeled_mask, n_blobs = label(binary_mask)
        if n_blobs > 0:
            sizes = [(labeled_mask == i).sum() for i in range(1, n_blobs + 1)]
            tumor_mask = labeled_mask == (np.argmax(sizes) + 1)
        else:
            tumor_mask = binary_mask
        response = img

    peak_x, peak_y = weighted_centroid(tumor_mask, img, x_axis, y_axis,
                                       bias_alpha=bias_alpha)

    area_px = int(tumor_mask.sum())
    if area_px > 0:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(tumor_mask)
        perimeter_px = max(int((tumor_mask & ~eroded).sum()), 1)
        compactness = (4 * np.pi * area_px) / (perimeter_px ** 2)
    else:
        compactness = 0.0

    return dict(
        tumor_mask=tumor_mask, blob_response=response, threshold_used=thresh,
        peak_x=peak_x, peak_y=peak_y, blob_area_px=area_px,
        blob_compactness=compactness,
    )