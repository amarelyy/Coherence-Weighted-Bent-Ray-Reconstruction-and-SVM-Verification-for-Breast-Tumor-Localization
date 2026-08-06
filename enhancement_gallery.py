"""
enhancement_gallery.py

Generate a gallery of 12 enhancement variants for DAS+CF on a few sample scans.
YOU look at the output and tell me which panel shows the cleanest blob.
Then we lock that exact setting into the final visualizer.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

OUTPUT_DIR = Path("results/gallery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCANS = [100, 216, 350]  # pick a few; edit as you like


def norm01(a):
    a = np.abs(a).astype(float)
    m = a.max()
    return a / (m + 1e-30)


def pct_clip(a, lo=1, hi=99.5):
    """Percentile windowing — robust to extreme outliers."""
    a = norm01(a)
    lo_v = np.percentile(a, lo)
    hi_v = np.percentile(a, hi)
    out = np.clip(a, lo_v, hi_v)
    return norm01(out)


# ---- 12 enhancement strategies ----
def make_variants(img):
    n = norm01(img)
    nz = n[n > 0]
    p15 = float(np.percentile(nz, 15)) if nz.size else 0.1

    def thresh(a, t):
        return np.where(a < t, 0.0, a)

    return [
        ("1. Linear (norm)", n),
        ("2. dB clip (-40dB)", norm01(np.clip(20*np.log10(n + 1e-6), -40, 0))),
        ("3. MER p=1.5", n ** 1.5),
        ("4. MER p=2.0", n ** 2.0),
        ("5. MER p=3.0", n ** 3.0),
        ("6. MER p=2 + thr15", thresh(n ** 2.0, min(p15, 0.2))),
        ("7. MER p=2 + thr30", thresh(n ** 2.0, 0.3)),
        ("8. MER p=2 + gamma0.7", norm01((n ** 2.0) ** 0.7)),
        ("9. Pct clip 1-99.5", pct_clip(img, 1, 99.5)),
        ("10. Pct clip 5-99", pct_clip(img, 5, 99)),
        ("11. Sqrt (gamma2)", np.sqrt(n)),
        ("12. MER p=2 + pct", pct_clip(n ** 2.0, 5, 99.5)),
    ]


def main():
    print("Loading data...")
    d = load_all_data()
    s21, tumor_model = d["s21"], d["tumor_model"]

    for scan_idx in SCANS:
        row = tumor_model.iloc[scan_idx]
        print(f"\nScan {scan_idx} ({row['phant_id']}) — DAS+CF")

        result = reconstruct_scan(
            scan_idx=scan_idx, s21=s21, tumor_model=tumor_model,
            beamformer="das", use_cf=True, return_diagnostics=True,
        )
        img = np.asarray(result["diagnostics"]["image"])
        axis = np.asarray(result["diagnostics"]["axis_mm"])
        gt_x, gt_y, gt_r = result["gt_x_mm"], result["gt_y_mm"], result["gt_r_mm"]
        le = result["localization_error_mm"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]

        variants = make_variants(img)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"DAS+CF | {row['phant_id']} | LE={le:.1f}mm | "
                     f"GT=({gt_x:.0f},{gt_y:.0f}) r={gt_r:.0f}mm — PICK THE BEST PANEL",
                     fontsize=16, color="white", fontweight="bold")

        for ax, (title, panel) in zip(axes.ravel(), variants):
            ax.imshow(panel, extent=extent, origin="lower",
                      cmap="turbo", vmin=0, vmax=1, aspect="equal")
            ax.add_patch(plt.Circle((gt_x, gt_y), gt_r, fill=False,
                                    edgecolor="white", linewidth=1.5, linestyle="--"))
            ax.set_title(title, color="white", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        out = OUTPUT_DIR / f"gallery_scan{scan_idx}_{row['phant_id']}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"  saved -> {out}")

    print(f"\nDONE. Open the images in {OUTPUT_DIR}/ and tell me which panel number looks best.")


if __name__ == "__main__":
    main()