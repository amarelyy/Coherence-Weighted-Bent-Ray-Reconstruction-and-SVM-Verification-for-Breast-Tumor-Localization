"""
visualize_results.py

Reconstruct + visualize ALL 8 imaging variants from the latest GitHub ablation
matrix, using the proven src/pipeline (return_diagnostics=True) so images match
the computed LE/SCR metrics exactly.

Enhancement strategy (validated via diagnostic_pattern analysis):
  - MER power enhancement to suppress clutter / diffraction rings.
  - CF variants  -> low power (image already peaky after coherence weighting).
  - Raw/GIBR     -> high power (heavy clutter needs strong suppression).
  - NO gamma 0.5  (it re-introduces side-lobes & ring artifacts).
  - Adaptive low floor-threshold so the blob is never wiped to black.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan


# ---- The 8 variants, mirrored exactly from ablation_runner.VARIANTS ----
VARIANTS = [
    dict(name="Raw_DAS",      beamformer="das",  use_bent_ray=False, use_cf=False),
    dict(name="Raw_DMAS",     beamformer="dmas", use_bent_ray=False, use_cf=False),
    dict(name="DAS_CF",       beamformer="das",  use_bent_ray=False, use_cf=True),
    dict(name="DMAS_CF",      beamformer="dmas", use_bent_ray=False, use_cf=True),
    dict(name="DAS_GIBR",     beamformer="das",  use_bent_ray=False, use_cf=False,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DMAS_GIBR",    beamformer="dmas", use_bent_ray=False, use_cf=False,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DAS_GIBR_CF",  beamformer="das",  use_bent_ray=False, use_cf=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DMAS_GIBR_CF", beamformer="dmas", use_bent_ray=False, use_cf=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
]

N_SAMPLES = 10
OUTPUT_DIR = Path("results/sample_images")
RANDOM_SEED = 42


def enhance_for_display(img, is_cf):
    """
    MER-style power enhancement tuned per variant family.

    Returns (enhanced_image, floor_threshold_used).
    """
    img = np.abs(np.asarray(img)).astype(np.float64)
    m = img.max()
    if m < 1e-30:
        return np.zeros_like(img), 0.0

    norm = img / m

    # CF images are already peaky after coherence weighting -> gentle power.
    # Raw / GIBR images are clutter-heavy -> strong power to kill rings.
    power = 1.2 if is_cf else 2.0
    enhanced = norm ** power

    # Adaptive floor: remove only the lowest-energy noise floor, never the blob.
    nz = enhanced[enhanced > 0]
    if nz.size > 50:
        thresh = float(np.percentile(nz, 12))   # drop bottom 12% only
    else:
        thresh = 0.0
    thresh = min(thresh, 0.25)                  # safety cap
    enhanced = np.where(enhanced < thresh, 0.0, enhanced)

    em = enhanced.max()
    if em > 1e-12:
        enhanced = enhanced / em
    return enhanced, thresh


def save_image(enhanced, axis_mm, variant, row, result, var_dir):
    le = float(result["localization_error_mm"])
    scr = float(result["scr_db"])
    peak_x, peak_y = result["peak_x_mm"], result["peak_y_mm"]
    gt_x, gt_y, gt_r = result["gt_x_mm"], result["gt_y_mm"], result["gt_r_mm"]
    phant_id = result["phant_id"]
    birads = result.get("birads", np.nan)
    blob_compact = result.get("blob_compactness", np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [axis_mm[0], axis_mm[-1], axis_mm[0], axis_mm[-1]]
    im = ax.imshow(enhanced, extent=extent, origin="lower",
                   cmap="turbo", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Enhanced Intensity")

    # Ground-truth tumor circle
    ax.add_patch(plt.Circle((gt_x, gt_y), gt_r, fill=False, edgecolor="lime",
                            linewidth=2, linestyle="--",
                            label=f"GT ({gt_x:.0f},{gt_y:.0f}) r={gt_r:.0f}mm"))
    # Detected peak
    ax.plot(peak_x, peak_y, "w+", markersize=14, markeredgewidth=2,
            label=f"Det ({peak_x:.0f},{peak_y:.0f})")

    birads_txt = f" | BI-RADS {birads:.0f}" if not np.isnan(birads) else ""
    ax.set_title(f"{variant['name']} | {phant_id}{birads_txt}\n"
                 f"LE={le:.1f}mm | SCR={scr:.1f}dB | compact={blob_compact:.2f}",
                 fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel("X (mm)", color="white")
    ax.set_ylabel("Y (mm)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.legend(loc="upper right", fontsize=8, facecolor="black",
              edgecolor="white", labelcolor="white")

    out_path = var_dir / f"{phant_id}_LE{le:.0f}mm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out_path, le, scr


def main():
    print("Loading data...")
    d = load_all_data()
    s21 = d["s21"]
    tumor_model = d["tumor_model"]
    n_valid = d["n_valid_scans"]

    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = sorted(rng.choice(n_valid, size=min(N_SAMPLES, n_valid), replace=False))
    sample_indices = [int(i) for i in sample_indices]
    print(f"Selected {len(sample_indices)} random scans: {sample_indices}\n")

    for variant in VARIANTS:
        is_cf = variant["use_cf"]
        var_dir = OUTPUT_DIR / variant["name"]
        var_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'='*60}\nVariant: {variant['name']}  (cf={is_cf})\n{'='*60}")

        ok, fail = 0, 0
        for scan_idx in sample_indices:
            phant_id = tumor_model.iloc[scan_idx]["phant_id"]
            try:
                result = reconstruct_scan(
                    scan_idx=scan_idx,
                    s21=s21,
                    tumor_model=tumor_model,
                    beamformer=variant["beamformer"],
                    use_bent_ray=variant.get("use_bent_ray", False),
                    use_cf=is_cf,
                    bent_ray_params=variant.get("bent_ray_params"),
                    return_diagnostics=True,
                )
                img = result["diagnostics"]["image"]
                axis_mm = np.asarray(result["diagnostics"]["axis_mm"])

                enhanced, _ = enhance_for_display(img, is_cf)
                out_path, le, scr = save_image(enhanced, axis_mm, variant,
                                               tumor_model.iloc[scan_idx],
                                               result, var_dir)
                ok += 1
                print(f"  [{phant_id}] LE={le:.1f}mm SCR={scr:.1f}dB -> {out_path.name}")
            except Exception as e:
                fail += 1
                print(f"  [{phant_id}] FAILED: {e}")

        print(f"  -> {ok} ok, {fail} failed\n")

    print(f"{'='*60}\nDONE -> {OUTPUT_DIR}/\n{'='*60}")


if __name__ == "__main__":
    main()