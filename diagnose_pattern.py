"""
diagnose_pattern.py — lihat setiap step preprocessing untuk identify pattern source
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

# Load 1 scan
d = load_all_data()
s21 = d["s21"]
tumor_model = d["tumor_model"]

scan_idx = 100  # pick one scan
result = reconstruct_scan(
    scan_idx=scan_idx,
    s21=s21,
    tumor_model=tumor_model,
    beamformer="das",
    use_cf=True,
    return_diagnostics=True,
)

img = result["diagnostics"]["image"]
time_sig = result["diagnostics"]["time_signal"]
time_sig_filt = result["diagnostics"]["time_signal_filtered"]

# Plot diagnostic
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Raw time signal (first 5 channels)
axes[0, 0].plot(np.abs(time_sig[:, :5]))
axes[0, 0].set_title("Raw time signal (5 channels)")
axes[0, 0].set_xlabel("Time sample")
axes[0, 0].set_ylabel("Amplitude")

# 2. Filtered time signal
axes[0, 1].plot(np.abs(time_sig_filt[:, :5]))
axes[0, 1].set_title("After TVSVD filter")

# 3. Raw image (no enhancement)
axes[0, 2].imshow(np.abs(img), cmap="hot", aspect="equal")
axes[0, 2].set_title("Raw |img| (no enhancement)")

# 4. Image with MER enhancement
img_abs = np.abs(img)
mer = (img_abs / (img_abs.max() + 1e-12)) ** 2.0
axes[1, 0].imshow(mer, cmap="turbo", vmin=0, vmax=1, aspect="equal")
axes[1, 0].set_title("MER (power=2)")

# 5. Image with gamma
gamma = mer ** 0.5
axes[1, 1].imshow(gamma, cmap="turbo", vmin=0, vmax=1, aspect="equal")
axes[1, 1].set_title("MER + gamma(0.5)")

# 6. Image with aggressive threshold
threshold = 0.3
clean = gamma.copy()
clean[clean < threshold] = 0
axes[1, 2].imshow(clean, cmap="turbo", vmin=0, vmax=1, aspect="equal")
axes[1, 2].set_title(f"Threshold={threshold} (pattern removal)")

plt.tight_layout()
plt.savefig("diagnostic_pattern.png", dpi=150, bbox_inches="tight")
print("Saved: diagnostic_pattern.png")
plt.show()