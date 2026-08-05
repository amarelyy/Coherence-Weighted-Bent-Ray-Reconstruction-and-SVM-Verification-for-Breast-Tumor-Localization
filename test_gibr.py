from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan

d = load_all_data()
r = reconstruct_scan(0, d["s21"], d["tumor_model"],
    beamformer="das", use_cf=False,
    bent_ray_params={"model": "geometry_informed", "eps_fibro": 45.0, "z_frac": 0.80})
le = r["localization_error_mm"]
print(f"LE: {le:.1f}mm")