"""
visualize_final_v5.py

v4 (proven good) + GIBR variants as visual proof of failure.
- Non-GIBR: straight-ray delay from v4 (clean blobs)
- GIBR: uses pipeline.py bent-ray logic (produces broken images)
- Same MER + gamma(0.5) enhancement for all variants
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.ndimage import label

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan


C_LIGHT = 3e8
EPS_FAT = 7.0
EPS_FIB = 45.0
TRAJECTORY_RADIUS_MM = 215.0
SEPARATION_DEG = 60.0


def align_channels(signal_scan, delay_grid, time):
    n_channel = signal_scan.shape[1]
    npix = delay_grid.shape[1]
    aligned = np.zeros((n_channel, npix), dtype=complex)
    for ch in range(n_channel):
        interp_real = interp1d(time, signal_scan[:, ch].real,
                                bounds_error=False, fill_value=0.0)
        interp_imag = interp1d(time, signal_scan[:, ch].imag,
                                bounds_error=False, fill_value=0.0)
        aligned[ch] = interp_real(delay_grid[ch]) + 1j * interp_imag(delay_grid[ch])
    return aligned


def beamform_das(aligned):
    return np.abs(aligned.sum(axis=0)) / aligned.shape[0]


def beamform_dmas(aligned):
    sum_s = aligned.sum(axis=0)
    sum_s2 = (aligned ** 2).sum(axis=0)
    return np.abs(0.5 * (sum_s ** 2 - sum_s2))


def coherence_factor(aligned):
    n = aligned.shape[0]
    num = np.abs(aligned.sum(axis=0)) ** 2
    den = n * (np.abs(aligned) ** 2).sum(axis=0)
    return num / (den + 1e-12)


def mer_enhance(img, power=2.0):
    return (img / (img.max() + 1e-12)) ** power


def gamma_correction(img, gamma=0.5):
    return (img / (img.max() + 1e-12)) ** gamma


def select_tvsvd_rank_adaptive(S, min_rank=1, max_energy=0.98):
    energy = S ** 2
    cum = np.cumsum(energy) / np.sum(energy)
    n = len(cum)
    if n < 3:
        return max(min_rank, int(np.argmax(cum >= 0.90)))
    x_norm = np.arange(n) / (n - 1)
    p1 = np.array([x_norm[0], cum[0]])
    p2 = np.array([x_norm[-1], cum[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return max(min_rank, int(np.argmax(cum >= 0.90)))
    line_unit = line_vec / line_len
    pts = np.stack([x_norm, cum], axis=1) - p1
    proj_pts = np.outer(pts @ line_unit, line_unit)
    dist = np.linalg.norm(pts - proj_pts, axis=1)
    knee = max(min_rank, int(np.argmax(dist)))
    hard_cap = int(np.argmax(cum >= max_energy))
    return min(knee, hard_cap) if hard_cap > 0 else knee


def select_blob_threshold_adaptive(img):
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(img)
    except Exception:
        return np.percentile(img, 90)


# ---- Variants: 4 working + 4 GIBR (broken) ----
VARIANTS = [
    # Working variants (v4 straight-ray logic)
    dict(name="Raw_DAS",       use_cf=False, beamformer="das",  use_gibr=False),
    dict(name="Raw_DMAS",      use_cf=False, beamformer="dmas", use_gibr=False),
    dict(name="DAS_CF",        use_cf=True,  beamformer="das",  use_gibr=False),
    dict(name="DMAS_CF",       use_cf=True,  beamformer="dmas", use_gibr=False),
    # GIBR variants (use pipeline.py bent-ray, expected to be broken)
    dict(name="DAS_GIBR",      use_cf=False, beamformer="das",  use_gibr=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DMAS_GIBR",     use_cf=False, beamformer="dmas", use_gibr=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DAS_GIBR_CF",   use_cf=True,  beamformer="das",  use_gibr=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
    dict(name="DMAS_GIBR_CF",  use_cf=True,  beamformer="dmas", use_gibr=True,
         bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80}),
]

N_SAMPLES = 10
OUTPUT_DIR = Path("results/v5_images")
RANDOM_SEED = 42


def reconstruct_v4_style(scan_idx, s21_all, tumor_model, variant, nx=120, ny=120):
    """v4 straight-ray logic for non-GIBR variants."""
    scan_params = tumor_model.iloc[scan_idx]
    breast_radius_mm = scan_params["breast_radius_mm"]
    fib_frac = scan_params.get("fib_fraction", 0.3)
    fat_frac = scan_params.get("fat_fraction", 0.7)

    eps_eff = fat_frac * EPS_FAT + fib_frac * EPS_FIB
    wave_velocity = C_LIGHT / np.sqrt(eps_eff)

    s21_scan = s21_all[scan_idx]
    N_FREQ = s21_scan.shape[0]
    window = tukey(N_FREQ, alpha=0.25)
    time_signal = np.fft.ifft(s21_scan * window[:, None], axis=0)
    rms = np.sqrt(np.mean(np.abs(time_signal) ** 2, axis=0, keepdims=True))
    time_signal = time_signal / (rms + 1e-12)

    F_START, F_STOP = 2e9, 9e9
    frequency = np.linspace(F_START, F_STOP, N_FREQ)
    delta_f = frequency[1] - frequency[0]
    delta_t = 1 / (N_FREQ * delta_f)
    time = np.arange(N_FREQ) * delta_t

    x_axis = np.linspace(-breast_radius_mm, breast_radius_mm, nx)
    y_axis = np.linspace(-breast_radius_mm, breast_radius_mm, ny)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    gx, gy = grid_x.ravel(), grid_y.ravel()

    N_POS = s21_scan.shape[1]
    pos_angles = np.linspace(0, 2 * np.pi, N_POS, endpoint=False)
    ant_x = TRAJECTORY_RADIUS_MM * np.cos(pos_angles)
    ant_y = TRAJECTORY_RADIUS_MM * np.sin(pos_angles)

    sep_steps = int(round(SEPARATION_DEG / 360.0 * N_POS))
    tx_idx = np.arange(N_POS)
    rx_idx = (np.arange(N_POS) + sep_steps) % N_POS

    dist_tx = np.sqrt((gx[None, :] - ant_x[tx_idx, None]) ** 2 +
                      (gy[None, :] - ant_y[tx_idx, None]) ** 2)
    dist_rx = np.sqrt((gx[None, :] - ant_x[rx_idx, None]) ** 2 +
                      (gy[None, :] - ant_y[rx_idx, None]) ** 2)

    delay_grid = ((dist_tx + dist_rx) / 1000) / wave_velocity

    # TVSVD (all variants)
    U, S, Vt = np.linalg.svd(time_signal, full_matrices=False)
    remove = select_tvsvd_rank_adaptive(S)
    S_filtered = S.copy()
    S_filtered[:remove] = 0
    time_signal_tsvd = U @ np.diag(S_filtered) @ Vt

    aligned = align_channels(time_signal_tsvd, delay_grid, time)

    if variant["beamformer"] == "das":
        img = beamform_das(aligned)
        if variant["use_cf"]:
            img = img * coherence_factor(aligned)
    else:
        img = beamform_dmas(aligned)
        if variant["use_cf"]:
            img = img * coherence_factor(aligned)

    img = img.reshape(ny, nx)
    enhanced = gamma_correction(mer_enhance(img, power=2.0), gamma=0.5)
    return enhanced, x_axis, y_axis


def reconstruct_gibr_style(scan_idx, s21_all, tumor_model, variant):
    """Use pipeline.py GIBR logic — expected to produce broken images."""
    result = reconstruct_scan(
        scan_idx=scan_idx,
        s21=s21_all,
        tumor_model=tumor_model,
        beamformer=variant["beamformer"],
        use_cf=variant.get("use_cf", False),
        use_bent_ray=False,
        bent_ray_params=variant.get("bent_ray_params"),
        return_diagnostics=True,
    )
    img_raw = np.asarray(result["diagnostics"]["image"])
    axis_mm = np.asarray(result["diagnostics"]["axis_mm"])

    img_abs = np.abs(img_raw)
    enhanced = gamma_correction(mer_enhance(img_abs, power=2.0), gamma=0.5)
    return enhanced, axis_mm


def find_peak(enhanced, x_axis, y_axis):
    thresh = select_blob_threshold_adaptive(enhanced)
    binary_mask = enhanced >= thresh
    labeled_mask, n_blobs = label(binary_mask)

    if n_blobs > 0:
        sizes = [(labeled_mask == i).sum() for i in range(1, n_blobs + 1)]
        tumor_mask = labeled_mask == (np.argmax(sizes) + 1)
    else:
        tumor_mask = binary_mask

    ys, xs = np.where(tumor_mask)
    if len(xs) > 0:
        weights = enhanced[ys, xs]
        return float(np.average(x_axis[xs], weights=weights)), \
               float(np.average(y_axis[ys], weights=weights))

    fb = np.unravel_index(np.argmax(enhanced), enhanced.shape)
    return float(x_axis[fb[1]]), float(y_axis[fb[0]])


def main():
    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]
    n_valid = d["n_valid_scans"]

    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = [int(i) for i in sorted(rng.choice(
        n_valid, size=min(N_SAMPLES, n_valid), replace=False))]
    print(f"Selected {len(sample_indices)} scans: {sample_indices}\n")

    for variant in VARIANTS:
        var_dir = OUTPUT_DIR / variant["name"]
        var_dir.mkdir(parents=True, exist_ok=True)
        is_gibr = variant.get("use_gibr", False)
        tag = "GIBR (expected broken)" if is_gibr else "v4 straight-ray"
        print(f"{'='*60}\nVariant: {variant['name']} [{tag}]\n{'='*60}")

        ok = fail = 0
        for scan_idx in sample_indices:
            row = tumor_model.iloc[scan_idx]
            phant_id = row["phant_id"]
            birads = row.get("birads", np.nan)
            gt_x = float(row["tumor_x_mm"])
            gt_y = float(row["tumor_y_mm"])
            gt_r = float(row["tumor_radius_mm"])

            try:
                if is_gibr:
                    enhanced, axis_mm = reconstruct_gibr_style(
                        scan_idx, s21, tumor_model, variant)
                    x_axis = y_axis = axis_mm
                else:
                    enhanced, x_axis, y_axis = reconstruct_v4_style(
                        scan_idx, s21, tumor_model, variant)

                peak_x, peak_y = find_peak(enhanced, x_axis, y_axis)
                le = float(np.sqrt((peak_x - gt_x) ** 2 + (peak_y - gt_y) ** 2))

            except Exception as e:
                print(f"  [{phant_id}] FAILED: {e}")
                fail += 1
                continue

            fig, ax = plt.subplots(figsize=(8, 7))
            extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
            im = ax.imshow(enhanced, extent=extent, origin="lower",
                           cmap="turbo", vmin=0, vmax=1, aspect="equal")
            plt.colorbar(im, ax=ax, label="Enhanced Intensity")

            ax.add_patch(plt.Circle((gt_x, gt_y), gt_r, fill=False, edgecolor="lime",
                                    linewidth=2, linestyle="--",
                                    label=f"GT ({gt_x:.0f},{gt_y:.0f}) r={gt_r:.0f}mm"))
            ax.plot(peak_x, peak_y, "w+", markersize=14, markeredgewidth=2,
                    label=f"Det ({peak_x:.0f},{peak_y:.0f})")

            birads_txt = f" | BI-RADS {birads:.0f}" if not np.isnan(birads) else ""
            gibr_note = "\n[GIBR: bent-ray delay failure]" if is_gibr else ""
            ax.set_title(f"{variant['name']} | {phant_id}{birads_txt}\n"
                         f"LE={le:.1f}mm{gibr_note}",
                         fontsize=12, fontweight="bold", color="white")
            ax.set_xlabel("X (mm)", color="white")
            ax.set_ylabel("Y (mm)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.legend(loc="upper right", fontsize=9, facecolor="black",
                      edgecolor="white", labelcolor="white")

            out_path = var_dir / f"{phant_id}_LE{le:.0f}mm.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="black")
            plt.close(fig)
            ok += 1
            print(f"  [{phant_id}] LE={le:.1f}mm -> {out_path.name}")

        print(f"  -> {ok} ok, {fail} failed\n")

    print(f"{'='*60}\nDONE -> {OUTPUT_DIR}/\n{'='*60}")


if __name__ == "__main__":
    main()