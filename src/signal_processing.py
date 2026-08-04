"""
src/signal_processing.py
Frequency-to-time transform (ICZT) and hybrid TVSVD clutter suppression.
ICZT and TVSVD run on CPU (numpy) because umbmid.sigproc is numpy-only.
GPU acceleration happens downstream in beamforming and physics modules.

Bandwidth updated to 4-6 GHz per Olaya Lopez (2024).
"""

import numpy as np
from scipy.signal.windows import tukey

try:
    from umbmid.sigproc import iczt
    ICZT_AVAILABLE = True
except ImportError:
    ICZT_AVAILABLE = False

# Updated bandwidth per Olaya Lopez: 4-6 GHz sub-band
FREQ_START_HZ = 4e9
FREQ_STOP_HZ = 6e9
TIME_START_S = 0.0
TIME_STOP_S = 6e-9
N_TIME_PTS = 1024


def to_time_domain(fd_signal, window_alpha=0.25, n_time_pts=N_TIME_PTS):
    """Frequency-to-time domain via ICZT. Always runs on CPU."""
    if not ICZT_AVAILABLE:
        raise ImportError(
            "umbmid.sigproc.iczt not importable — copy the umbmid/ "
            "package into the repo root before running the pipeline."
        )
    fd_np = np.asarray(fd_signal)
    window = tukey(fd_np.shape[0], alpha=window_alpha)
    fd_windowed = fd_np * window[:, None]
    return iczt(fd_windowed, ini_t=TIME_START_S, fin_t=TIME_STOP_S,
                n_time_pts=n_time_pts,
                ini_f=FREQ_START_HZ, fin_f=FREQ_STOP_HZ)


def get_time_axis(n_time_pts=N_TIME_PTS):
    return np.linspace(TIME_START_S, TIME_STOP_S, n_time_pts)


def select_tvsvd_rank_adaptive(S, min_rank=1, max_energy=0.98):
    """Kneedle elbow detection on cumulative-energy curve."""
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
    proj_len = pts @ line_unit
    proj_pts = np.outer(proj_len, line_unit)
    dist = np.linalg.norm(pts - proj_pts, axis=1)
    knee = max(min_rank, int(np.argmax(dist)))
    hard_cap = int(np.argmax(cum >= max_energy))
    return min(knee, hard_cap) if hard_cap > 0 else knee


def select_tvsvd_rank_hybrid(S, Vt, energy_lower=0.01,
                              flatness_thresh=0.70,
                              max_energy=0.98):
    """Hybrid TVSVD rank selection. flatness_thresh=0.70 protects tumor signals."""
    knee = select_tvsvd_rank_adaptive(S, max_energy=max_energy)
    energy_frac = (S ** 2) / np.sum(S ** 2)
    remove_mask = np.zeros(len(S), dtype=bool)
    for k in range(knee):
        row = Vt[k]
        flatness = np.abs(np.mean(row)) / (np.sqrt(np.mean(row ** 2)) + 1e-12)
        if energy_frac[k] > energy_lower and flatness > flatness_thresh:
            remove_mask[k] = True
    return remove_mask


def apply_hybrid_tvsvd(time_signal):
    """Hybrid TVSVD clutter suppression. Runs entirely on CPU (numpy)."""
    sig_np = np.asarray(time_signal)
    U, S, Vt = np.linalg.svd(sig_np, full_matrices=False)
    remove_mask = select_tvsvd_rank_hybrid(S, Vt)
    S_filtered = S.copy()
    S_filtered[remove_mask] = 0.0
    filtered = U @ np.diag(S_filtered) @ Vt
    return filtered, int(remove_mask.sum())