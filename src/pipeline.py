"""
src/pipeline.py

reconstruct_scan() — self-contained, per-scan reconstruction.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from . import physics
from . import signal_processing as sp
from . import beamforming as bf
from . import blob_detection as bd
from . import metrics as mx

DEFAULT_GRID_MARGIN_FACTOR = 1.5
DEFAULT_GRID_STEP_MM = 1.0

DEFAULT_BENT_RAY_PARAMS = {
    "model": "two_medium",
    "eps_adipose": 7.0,
    "eps_fibro": 45.0,
    "z_frac": 0.80,
}

_DELAY_CACHE = {}


def _delay_cache_key(phant_id, use_bent_ray, ant_rad_mm, breast_radius_mm,
                     v_tissue, bent_ray_params, margin_factor, grid_step_mm,
                     shell_center):
    extra = tuple(sorted((bent_ray_params or {}).items()))
    return (phant_id, use_bent_ray, round(ant_rad_mm, 3),
            round(breast_radius_mm, 3), round(v_tissue, 3),
            extra, margin_factor, grid_step_mm)


def build_grid(breast_radius_mm, margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
               grid_step_mm=DEFAULT_GRID_STEP_MM):
    grid_radius_mm = breast_radius_mm * margin_factor
    axis_mm = np.arange(-grid_radius_mm, grid_radius_mm + grid_step_mm,
                        grid_step_mm)
    grid_x_mm, grid_y_mm = np.meshgrid(axis_mm, axis_mm)
    return grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm


def reconstruct_scan(scan_idx, s21, tumor_model,
                     beamformer="das", use_bent_ray=False, use_cf=True,
                     bent_ray_params=None, shell_center=(0.0, 0.0),
                     margin_factor=DEFAULT_GRID_MARGIN_FACTOR,
                     grid_step_mm=DEFAULT_GRID_STEP_MM,
                     return_diagnostics=False):
    t_start = perf_counter()

    row = tumor_model.iloc[scan_idx]
    breast_radius_mm = float(row["breast_radius_mm"])
    fat_frac = float(row["fat_fraction"])
    fib_frac = float(row["fib_fraction"])
    v_tissue, eps_tissue = physics.compute_tissue_velocity(fat_frac, fib_frac)

    ant_rad_cm = float(row.get("ant_rad", 21.5))
    ant_rad_mm = ant_rad_cm * 10.0
    geom = physics.get_antenna_geometry(ant_rad_mm)

    fd_scan = s21[scan_idx]
    time_signal = sp.to_time_domain(fd_scan)
    time_axis = sp.get_time_axis(time_signal.shape[0])
    time_signal_filtered, n_removed = sp.apply_hybrid_tvsvd(time_signal)

    grid_x_mm, grid_y_mm, axis_mm, grid_radius_mm = build_grid(
        breast_radius_mm, margin_factor, grid_step_mm)
    grid_x_m = grid_x_mm.ravel() / 1000.0
    grid_y_m = grid_y_mm.ravel() / 1000.0
    shell_center_m = (shell_center[0] / 1000.0, shell_center[1] / 1000.0)

    params = {**DEFAULT_BENT_RAY_PARAMS, **(bent_ray_params or {})}
    delay_model = params.get("model", "two_medium")

    if use_bent_ray and delay_model == "two_medium":
        delay_model = "multilayer_noskin"

    cache_key = _delay_cache_key(
        row["phant_id"], use_bent_ray, ant_rad_mm, breast_radius_mm,
        v_tissue, bent_ray_params, margin_factor, grid_step_mm, shell_center,
    )

    if cache_key in _DELAY_CACHE:
        delay_grid = _DELAY_CACHE[cache_key]
    else:
        if delay_model == "geometry_informed":
            fib_model = str(row.get("fib_model", "F1"))
            stl_path = Path(__file__).resolve().parent.parent / "data" / f"{fib_model}.stl"
            z_frac = params.get("z_frac", 0.80)
            bx, by = physics.load_stl_boundary(stl_path, z_frac=z_frac)
            boundary_r = np.sqrt(bx**2 + by**2)
            mean_r = np.mean(boundary_r)
            if mean_r > 0:
                scale = (breast_radius_mm / 1000.0) / mean_r
                bx = bx * scale
                by = by * scale
            v_fibro = physics.C_LIGHT / np.sqrt(params.get("eps_fibro", 45.0))
            delay_grid = physics.geometry_informed_bent_ray_delay(
                geom["ant_x"], geom["ant_y"],
                geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m, bx, by,
                physics.V_AIR, v_fibro,
            )
        elif delay_model == "multilayer_noskin":
            fib_radius_mm = physics.estimate_fib_radius_mm(breast_radius_mm, fib_frac)
            fib_radius_m = fib_radius_mm / 1000.0
            v_adipose = physics.C_LIGHT / np.sqrt(params["eps_adipose"])
            v_fibro = physics.C_LIGHT / np.sqrt(params["eps_fibro"])
            delay_grid = physics.bent_ray_noskin_delay(
                geom["ant_x"], geom["ant_y"],
                geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, fib_radius_m,
                physics.V_AIR, v_adipose, v_fibro,
            )
        else:
            delay_grid = physics.two_medium_delay(
                geom["ant_x"], geom["ant_y"],
                geom["ant_x_b"], geom["ant_y_b"],
                grid_x_m, grid_y_m,
                breast_radius_mm / 1000.0, v_tissue,
                shell_center=shell_center_m,
            )

        delay_grid = delay_grid.reshape(-1, *grid_x_mm.shape)
        _DELAY_CACHE[cache_key] = delay_grid

    if beamformer == "das":
        if use_cf:
            _, cf_map, img = bf.das_coherent_cf(time_signal_filtered, time_axis, delay_grid)
        else:
            img = bf.das_coherent(time_signal_filtered, time_axis, delay_grid)
            cf_map = None
    elif beamformer == "dmas":
        td_mag = np.abs(time_signal_filtered)
        if use_cf:
            img, cf_map = bf.dmas_cf(time_signal_filtered, td_mag, time_axis, delay_grid)
        else:
            img = bf.dmas(td_mag, time_axis, delay_grid)
            cf_map = None
    else:
        raise ValueError(f"Unknown beamformer: {beamformer!r}")

    blob = bd.extract_blob_candidate(img, axis_mm, axis_mm)

    gt_x_mm = float(row["tumor_x_mm"])
    gt_y_mm = float(row["tumor_y_mm"])
    gt_r_mm = float(row["tumor_radius_mm"])

    computed = mx.compute_all_metrics(
        img, blob["tumor_mask"], axis_mm, axis_mm,
        blob["peak_x"], blob["peak_y"], gt_x_mm, gt_y_mm, gt_r_mm,
    )

    cf_at_peak = None
    if cf_map is not None:
        peak_iy = np.argmin(np.abs(axis_mm - blob["peak_y"]))
        peak_ix = np.argmin(np.abs(axis_mm - blob["peak_x"]))
        cf_at_peak = float(cf_map[peak_iy, peak_ix])

    runtime_sec = perf_counter() - t_start

    result = dict(
        scan_idx=scan_idx,
        phant_id=row["phant_id"],
        birads=row.get("birads", np.nan),
        beamformer=beamformer, use_bent_ray=use_bent_ray, use_cf=use_cf,
        delay_model=delay_model,
        breast_radius_mm=breast_radius_mm, grid_radius_mm=grid_radius_mm,
        tvsvd_removed=n_removed,
        peak_x_mm=blob["peak_x"], peak_y_mm=blob["peak_y"],
        gt_x_mm=gt_x_mm, gt_y_mm=gt_y_mm, gt_r_mm=gt_r_mm,
        blob_area_px=blob["blob_area_px"], blob_compactness=blob["blob_compactness"],
        cf_at_peak=cf_at_peak,
        runtime_sec=runtime_sec,
        **computed,
    )

    if return_diagnostics:
        result["diagnostics"] = dict(
            image=img, cf_map=cf_map, tumor_mask=blob["tumor_mask"],
            axis_mm=axis_mm, time_signal=time_signal,
            time_signal_filtered=time_signal_filtered, delay_grid=delay_grid,
        )

    return result