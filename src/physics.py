"""
src/physics.py
GPU-accelerated antenna geometry + delay models.
The bent-ray 3-layer Fermat solver runs as a CUDA kernel (CuPy RawKernel),
processing all antenna-pixel pairs in parallel. Falls back to vectorized
NumPy on CPU if no GPU is available.

Speedup: 20-100x over original Python golden-section search.
"""

import numpy as np
from src.backend import xp, HAS_GPU, to_gpu, to_cpu

C_LIGHT = 3e8
EPSILON_AIR = 1.0006
V_AIR = C_LIGHT / np.sqrt(EPSILON_AIR)

N_ANT = 72
SEPARATION_DEG = 60.0


# ---------------------------------------------------------------------------
# Antenna geometry (unchanged, runs on CPU — small arrays)
# ---------------------------------------------------------------------------
def get_corrected_ant_radius_m(raw_rad_mm):
    return (0.97 * (raw_rad_mm - 0.106) + 0.148) / 1000.0


def get_antenna_geometry(ant_rad_mm, n_ant=N_ANT,
                         separation_deg=SEPARATION_DEG,
                         apply_correction=True):
    ant_rad_m = (get_corrected_ant_radius_m(ant_rad_mm)
                 if apply_correction else ant_rad_mm / 1000.0)
    angles = np.linspace(0, -2 * np.pi, n_ant, endpoint=False)
    ant_x = ant_rad_m * np.cos(angles)
    ant_y = ant_rad_m * np.sin(angles)
    offset = np.deg2rad(separation_deg)
    ant_x_b = ant_rad_m * np.cos(angles + offset)
    ant_y_b = ant_rad_m * np.sin(angles + offset)
    sep_steps = int(round(separation_deg / 360.0 * n_ant))
    tx_idx = np.arange(n_ant)
    rx_idx = (np.arange(n_ant) + sep_steps) % n_ant
    return dict(ant_x=ant_x, ant_y=ant_y, ant_x_b=ant_x_b,
                ant_y_b=ant_y_b, tx_idx=tx_idx, rx_idx=rx_idx,
                ant_rad_m=ant_rad_m)


# ---------------------------------------------------------------------------
# Tissue velocity
# ---------------------------------------------------------------------------
def compute_tissue_velocity(fat_fraction, fib_fraction,
                            eps_fat=7.0, eps_fib=45.0):
    eps_tissue = fat_fraction * eps_fat + fib_fraction * eps_fib
    v_tissue = C_LIGHT / np.sqrt(eps_tissue)
    return v_tissue, eps_tissue


# ---------------------------------------------------------------------------
# CUDA kernel for bent-ray 3-layer delay
# ---------------------------------------------------------------------------
_BENT_RAY_CUDA_KERNEL = r"""
extern "C" __global__
void bent_ray_3layer_kernel(
    const double* __restrict__ ant_x,
    const double* __restrict__ ant_y,
    const double* __restrict__ ant_x_b,
    const double* __restrict__ ant_y_b,
    const double* __restrict__ grid_x,
    const double* __restrict__ grid_y,
    double*       __restrict__ delay_out,
    const int n_ant,
    const int n_pix,
    const double r_outer,
    const double r_inner,
    const double v_air,
    const double v_skin,
    const double v_interior,
    const int n_iter,
    const int fp_iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_ant * n_pix;
    if (idx >= total) return;

    int i_ant = idx / n_pix;
    int i_pix = idx % n_pix;

    double ax  = ant_x[i_ant];
    double ay  = ant_y[i_ant];
    double axb = ant_x_b[i_ant];
    double ayb = ant_y_b[i_ant];
    double gx  = grid_x[i_pix];
    double gy  = grid_y[i_pix];

    const double gr = 0.6180339887498949;  // (sqrt(5)-1)/2
    const double PI = 3.14159265358979323846;

    // Helper: golden-section refraction point (inline for speed)
    // Returns (bx, by) on circle of radius r
    // Minimizes travel time from (sx,sy) to (tx,ty) via circle boundary
    // We inline this as a macro-like pattern since CUDA C doesn't have
    // easy closures.

    // --- LEG TX: antenna -> pixel ---
    double total_tx = 0.0;

    // Initial guess: direct air->interior refraction on outer boundary
    double b1x, b1y;
    {
        double ang_s = atan2(ay, ax);
        double ang_t = atan2(gy, gx);
        double lo = fmin(ang_s, ang_t) - 0.75;
        double hi = fmax(ang_s, ang_t) + 0.75;
        double a = lo, b = hi;
        double c = b - gr * (b - a);
        double d = a + gr * (b - a);
        for (int it = 0; it < n_iter; it++) {
            double cx = r_outer * cos(c), cy = r_outer * sin(c);
            double d1 = sqrt((cx-ax)*(cx-ax) + (cy-ay)*(cy-ay));
            double d2 = sqrt((cx-gx)*(cx-gx) + (cy-gy)*(cy-gy));
            double fc = d1/v_air + d2/v_interior;
            double dx = r_outer * cos(d), dy = r_outer * sin(d);
            d1 = sqrt((dx-ax)*(dx-ax) + (dy-ay)*(dy-ay));
            d2 = sqrt((dx-gx)*(dx-gx) + (dy-gy)*(dy-gy));
            double fd = d1/v_air + d2/v_interior;
            if (fc < fd) { b = d; d = c; c = b - gr*(b-a); }
            else         { a = c; c = d; d = a + gr*(b-a); }
        }
        double phi = 0.5*(a+b);
        b1x = r_outer * cos(phi);
        b1y = r_outer * sin(phi);
    }

    // Fixed-point iterations for 3-layer
    double b2x = b1x, b2y = b1y;
    for (int fp = 0; fp < fp_iters; fp++) {
        // Solve inner boundary refraction: b1 -> pixel via r_inner
        {
            double ang_s = atan2(b1y, b1x);
            double ang_t = atan2(gy, gx);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_inner*cos(c), cy = r_inner*sin(c);
                double d1 = sqrt((cx-b1x)*(cx-b1x)+(cy-b1y)*(cy-b1y));
                double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
                double fc = d1/v_skin + d2/v_interior;
                double dx = r_inner*cos(d), dy = r_inner*sin(d);
                d1 = sqrt((dx-b1x)*(dx-b1x)+(dy-b1y)*(dy-b1y));
                d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
                double fd = d1/v_skin + d2/v_interior;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b2x = r_inner*cos(phi);
            b2y = r_inner*sin(phi);
        }
        // Solve outer boundary refraction: antenna -> b2 via r_outer
        {
            double ang_s = atan2(ay, ax);
            double ang_t = atan2(b2y, b2x);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_outer*cos(c), cy = r_outer*sin(c);
                double d1 = sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay));
                double d2 = sqrt((cx-b2x)*(cx-b2x)+(cy-b2y)*(cy-b2y));
                double fc = d1/v_air + d2/v_skin;
                double dx = r_outer*cos(d), dy = r_outer*sin(d);
                d1 = sqrt((dx-ax)*(dx-ax)+(dy-ay)*(dy-ay));
                d2 = sqrt((dx-b2x)*(dx-b2x)+(dy-b2y)*(dy-b2y));
                double fd = d1/v_air + d2/v_skin;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b1x = r_outer*cos(phi);
            b1y = r_outer*sin(phi);
        }
    }

    double d_air_tx  = sqrt((b1x-ax)*(b1x-ax) + (b1y-ay)*(b1y-ay));
    double d_skin_tx = sqrt((b2x-b1x)*(b2x-b1x) + (b2y-b1y)*(b2y-b1y));
    double d_int_tx  = sqrt((gx-b2x)*(gx-b2x) + (gy-b2y)*(gy-b2y));
    total_tx = d_air_tx/v_air + d_skin_tx/v_skin + d_int_tx/v_interior;

    // --- LEG RX: antenna_b -> pixel (same logic) ---
    double total_rx = 0.0;
    {
        double ang_s = atan2(ayb, axb);
        double ang_t = atan2(gy, gx);
        double lo = fmin(ang_s, ang_t) - 0.75;
        double hi = fmax(ang_s, ang_t) + 0.75;
        double a = lo, b = hi;
        double c = b - gr*(b-a), d = a + gr*(b-a);
        for (int it = 0; it < n_iter; it++) {
            double cx = r_outer*cos(c), cy = r_outer*sin(c);
            double d1 = sqrt((cx-axb)*(cx-axb)+(cy-ayb)*(cy-ayb));
            double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
            double fc = d1/v_air + d2/v_interior;
            double dx = r_outer*cos(d), dy = r_outer*sin(d);
            d1 = sqrt((dx-axb)*(dx-axb)+(dy-ayb)*(dy-ayb));
            d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
            double fd = d1/v_air + d2/v_interior;
            if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
            else         { a=c; c=d; d=a+gr*(b-a); }
        }
        double phi = 0.5*(a+b);
        b1x = r_outer*cos(phi);
        b1y = r_outer*sin(phi);
    }
    b2x = b1x; b2y = b1y;
    for (int fp = 0; fp < fp_iters; fp++) {
        {
            double ang_s = atan2(b1y, b1x);
            double ang_t = atan2(gy, gx);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_inner*cos(c), cy = r_inner*sin(c);
                double d1 = sqrt((cx-b1x)*(cx-b1x)+(cy-b1y)*(cy-b1y));
                double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
                double fc = d1/v_skin + d2/v_interior;
                double dx = r_inner*cos(d), dy = r_inner*sin(d);
                d1 = sqrt((dx-b1x)*(dx-b1x)+(dy-b1y)*(dy-b1y));
                d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
                double fd = d1/v_skin + d2/v_interior;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b2x = r_inner*cos(phi);
            b2y = r_inner*sin(phi);
        }
        {
            double ang_s = atan2(ayb, axb);
            double ang_t = atan2(b2y, b2x);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_outer*cos(c), cy = r_outer*sin(c);
                double d1 = sqrt((cx-axb)*(cx-axb)+(cy-ayb)*(cy-ayb));
                double d2 = sqrt((cx-b2x)*(cx-b2x)+(cy-b2y)*(cy-b2y));
                double fc = d1/v_air + d2/v_skin;
                double dx = r_outer*cos(d), dy = r_outer*sin(d);
                d1 = sqrt((dx-axb)*(dx-axb)+(dy-ayb)*(dy-ayb));
                d2 = sqrt((dx-b2x)*(dx-b2x)+(dy-b2y)*(dy-b2y));
                double fd = d1/v_air + d2/v_skin;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b1x = r_outer*cos(phi);
            b1y = r_outer*sin(phi);
        }
    }
    double d_air_rx  = sqrt((b1x-axb)*(b1x-axb) + (b1y-ayb)*(b1y-ayb));
    double d_skin_rx = sqrt((b2x-b1x)*(b2x-b1x) + (b2y-b1y)*(b2y-b1y));
    double d_int_rx  = sqrt((gx-b2x)*(gx-b2x) + (gy-b2y)*(gy-b2y));
    total_rx = d_air_rx/v_air + d_skin_rx/v_skin + d_int_rx/v_interior;

    delay_out[idx] = total_tx + total_rx;
}
"""

# Compile CUDA kernel once at module load (cached)
_cuda_kernel = None
if HAS_GPU:
    import cupy as cp
    _cuda_kernel = cp.RawKernel(_BENT_RAY_CUDA_KERNEL,
                                'bent_ray_3layer_kernel')


# ---------------------------------------------------------------------------
# Two-medium delay (vectorized, GPU or CPU)
# ---------------------------------------------------------------------------
def _leg_time_two_medium(p0x, p0y, grid_x, grid_y, shell_radius_m,
                         v_air, v_tissue, shell_center=(0.0, 0.0)):
    cx, cy = shell_center
    p0x_s, p0y_s = p0x - cx, p0y - cy
    gx_s, gy_s = grid_x - cx, grid_y - cy

    dx = gx_s - p0x_s
    dy = gy_s - p0y_s
    seg_len = xp.sqrt(dx ** 2 + dy ** 2)

    a = dx ** 2 + dy ** 2
    b = 2.0 * (p0x_s * dx + p0y_s * dy)
    c = p0x_s ** 2 + p0y_s ** 2 - shell_radius_m ** 2

    disc = b ** 2 - 4.0 * a * c
    valid = disc >= 0

    a_safe = xp.where(a == 0, 1e-30, a)
    sqrt_disc = xp.zeros_like(dx)
    sqrt_disc[valid] = xp.sqrt(disc[valid])

    t1 = (-b - sqrt_disc) / (2.0 * a_safe)
    t2 = (-b + sqrt_disc) / (2.0 * a_safe)
    t_lo = xp.clip(xp.minimum(t1, t2), 0.0, 1.0)
    t_hi = xp.clip(xp.maximum(t1, t2), 0.0, 1.0)

    tissue_frac = xp.where(valid, xp.maximum(t_hi - t_lo, 0.0), 0.0)
    dist_tissue = tissue_frac * seg_len
    dist_air = seg_len - dist_tissue

    return dist_air / v_air + dist_tissue / v_tissue


def two_medium_delay(ant_x, ant_y, ant_x_b, ant_y_b, grid_x, grid_y,
                     shell_radius_m, v_tissue, shell_center=(0.0, 0.0),
                     v_air=V_AIR):
    """Vectorized two-medium bistatic delay grid. GPU or CPU."""
    ant_x, ant_y = to_gpu(ant_x), to_gpu(ant_y)
    ant_x_b, ant_y_b = to_gpu(ant_x_b), to_gpu(ant_y_b)
    grid_x, grid_y = to_gpu(grid_x), to_gpu(grid_y)

    n_ant = len(ant_x)
    n_pix = grid_x.shape[-1] if grid_x.ndim > 1 else len(grid_x)
    delay = xp.zeros((n_ant, n_pix))

    for i in range(n_ant):
        t_a = _leg_time_two_medium(
            ant_x[i], ant_y[i], grid_x, grid_y,
            shell_radius_m, v_air, v_tissue, shell_center)
        t_b = _leg_time_two_medium(
            ant_x_b[i], ant_y_b[i], grid_x, grid_y,
            shell_radius_m, v_air, v_tissue, shell_center)
        delay[i] = t_a + t_b
    return delay


# ---------------------------------------------------------------------------
# Bent-ray 3-layer delay — GPU CUDA kernel or CPU fallback
# ---------------------------------------------------------------------------
def bent_ray_3layer_delay(ant_x, ant_y, ant_x_b, ant_y_b,
                          grid_x_flat, grid_y_flat,
                          breast_radius_m, skin_thickness_m,
                          v_air, v_skin, v_interior,
                          fixed_point_iters=3, n_gs_iters=40):
    """
    3-layer bistatic delay grid. Uses CUDA kernel on GPU for massive
    parallelism. Falls back to serial Python on CPU.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_flat)
    r_outer = breast_radius_m
    r_inner = max(breast_radius_m - skin_thickness_m, 1e-4)

    if HAS_GPU and _cuda_kernel is not None:
        import cupy as cp

        # Move inputs to GPU
        d_ant_x = cp.asarray(ant_x, dtype=cp.float64)
        d_ant_y = cp.asarray(ant_y, dtype=cp.float64)
        d_ant_x_b = cp.asarray(ant_x_b, dtype=cp.float64)
        d_ant_y_b = cp.asarray(ant_y_b, dtype=cp.float64)
        d_gx = cp.asarray(grid_x_flat, dtype=cp.float64)
        d_gy = cp.asarray(grid_y_flat, dtype=cp.float64)
        d_delay = cp.zeros(n_ant * n_pix, dtype=cp.float64)

        total = n_ant * n_pix
        threads_per_block = 256
        n_blocks = (total + threads_per_block - 1) // threads_per_block

        _cuda_kernel(
            (n_blocks,), (threads_per_block,),
            (d_ant_x, d_ant_y, d_ant_x_b, d_ant_y_b,
             d_gx, d_gy, d_delay,
             n_ant, n_pix,
             r_outer, r_inner,
             v_air, v_skin, v_interior,
             n_gs_iters, fixed_point_iters)
        )

        delay_flat = cp.asnumpy(d_delay)
        return delay_flat.reshape(n_ant, n_pix)

    else:
        # CPU fallback: serial golden-section (original logic)
        delay = np.zeros((n_ant, n_pix))
        for i in range(n_ant):
            for p in range(n_pix):
                delay[i, p] = _bent_ray_single_cpu(
                    ant_x[i], ant_y[i], ant_x_b[i], ant_y_b[i],
                    grid_x_flat[p], grid_y_flat[p],
                    r_outer, r_inner, v_air, v_skin, v_interior,
                    fixed_point_iters, n_gs_iters)
        return delay

# ===========================================================================
# Multi-layer bent-ray TANPA SKIN: Air → Adipose → Fibro
# Physically correct untuk UM-BMID phantom berlapis
# ===========================================================================

EPS_ADIPOSE = 7.0
EPS_FIBRO = 45.0
V_ADIPOSE = C_LIGHT / np.sqrt(EPS_ADIPOSE)   # ~1.13e8 m/s
V_FIBRO = C_LIGHT / np.sqrt(EPS_FIBRO)        # ~4.47e7 m/s


def estimate_fib_radius_mm(breast_radius_mm, fib_fraction):
    """Estimasi radius boundary fibro dari volume fraction (asumsi silinder konsentris).
    fib_radius = breast_radius * sqrt(fib_fraction)"""
    return breast_radius_mm * np.sqrt(np.clip(fib_fraction, 0.0, 1.0))


_BENT_RAY_NOSKIN_CUDA_KERNEL = r"""
extern "C" __global__
void bent_ray_noskin_kernel(
    const double* __restrict__ ant_x,
    const double* __restrict__ ant_y,
    const double* __restrict__ ant_x_b,
    const double* __restrict__ ant_y_b,
    const double* __restrict__ grid_x,
    const double* __restrict__ grid_y,
    double*       __restrict__ delay_out,
    const int n_ant,
    const int n_pix,
    const double r_outer,
    const double r_inner,
    const double v_air,
    const double v_adipose,
    const double v_fibro,
    const int n_iter,
    const int fp_iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_ant * n_pix;
    if (idx >= total) return;

    int i_ant = idx / n_pix;
    int i_pix = idx % n_pix;

    double ax  = ant_x[i_ant];
    double ay  = ant_y[i_ant];
    double axb = ant_x_b[i_ant];
    double ayb = ant_y_b[i_ant];
    double gx  = grid_x[i_pix];
    double gy  = grid_y[i_pix];

    const double gr = 0.6180339887498949;

    // --- LEG TX: antenna -> pixel ---
    double total_tx = 0.0;
    double b1x, b1y, b2x, b2y;

    // Initial: air -> fibro direct via outer boundary
    {
        double ang_s = atan2(ay, ax);
        double ang_t = atan2(gy, gx);
        double lo = fmin(ang_s, ang_t) - 0.75;
        double hi = fmax(ang_s, ang_t) + 0.75;
        double a = lo, b = hi;
        double c = b - gr*(b-a), d = a + gr*(b-a);
        for (int it = 0; it < n_iter; it++) {
            double cx = r_outer*cos(c), cy = r_outer*sin(c);
            double d1 = sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay));
            double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
            double fc = d1/v_air + d2/v_fibro;
            double dx = r_outer*cos(d), dy = r_outer*sin(d);
            d1 = sqrt((dx-ax)*(dx-ax)+(dy-ay)*(dy-ay));
            d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
            double fd = d1/v_air + d2/v_fibro;
            if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
            else         { a=c; c=d; d=a+gr*(b-a); }
        }
        double phi = 0.5*(a+b);
        b1x = r_outer*cos(phi); b1y = r_outer*sin(phi);
    }

    // Fixed-point: air->adipose at r_outer, adipose->fibro at r_inner
    b2x = b1x; b2y = b1y;
    for (int fp = 0; fp < fp_iters; fp++) {
        // Inner boundary: adipose -> fibro at r_inner
        {
            double ang_s = atan2(b1y, b1x);
            double ang_t = atan2(gy, gx);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_inner*cos(c), cy = r_inner*sin(c);
                double d1 = sqrt((cx-b1x)*(cx-b1x)+(cy-b1y)*(cy-b1y));
                double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
                double fc = d1/v_adipose + d2/v_fibro;
                double dx = r_inner*cos(d), dy = r_inner*sin(d);
                d1 = sqrt((dx-b1x)*(dx-b1x)+(dy-b1y)*(dy-b1y));
                d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
                double fd = d1/v_adipose + d2/v_fibro;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b2x = r_inner*cos(phi); b2y = r_inner*sin(phi);
        }
        // Outer boundary: air -> adipose at r_outer
        {
            double ang_s = atan2(ay, ax);
            double ang_t = atan2(b2y, b2x);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_outer*cos(c), cy = r_outer*sin(c);
                double d1 = sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay));
                double d2 = sqrt((cx-b2x)*(cx-b2x)+(cy-b2y)*(cy-b2y));
                double fc = d1/v_air + d2/v_adipose;
                double dx = r_outer*cos(d), dy = r_outer*sin(d);
                d1 = sqrt((dx-ax)*(dx-ax)+(dy-ay)*(dy-ay));
                d2 = sqrt((dx-b2x)*(dx-b2x)+(dy-b2y)*(dy-b2y));
                double fd = d1/v_air + d2/v_adipose;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b1x = r_outer*cos(phi); b1y = r_outer*sin(phi);
        }
    }

    double d_air   = sqrt((b1x-ax)*(b1x-ax)+(b1y-ay)*(b1y-ay));
    double d_adi   = sqrt((b2x-b1x)*(b2x-b1x)+(b2y-b1y)*(b2y-b1y));
    double d_fib   = sqrt((gx-b2x)*(gx-b2x)+(gy-b2y)*(gy-b2y));
    total_tx = d_air/v_air + d_adi/v_adipose + d_fib/v_fibro;

    // --- LEG RX: antenna_b -> pixel (same logic) ---
    double total_rx = 0.0;
    {
        double ang_s = atan2(ayb, axb);
        double ang_t = atan2(gy, gx);
        double lo = fmin(ang_s, ang_t) - 0.75;
        double hi = fmax(ang_s, ang_t) + 0.75;
        double a = lo, b = hi;
        double c = b - gr*(b-a), d = a + gr*(b-a);
        for (int it = 0; it < n_iter; it++) {
            double cx = r_outer*cos(c), cy = r_outer*sin(c);
            double d1 = sqrt((cx-axb)*(cx-axb)+(cy-ayb)*(cy-ayb));
            double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
            double fc = d1/v_air + d2/v_fibro;
            double dx = r_outer*cos(d), dy = r_outer*sin(d);
            d1 = sqrt((dx-axb)*(dx-axb)+(dy-ayb)*(dy-ayb));
            d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
            double fd = d1/v_air + d2/v_fibro;
            if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
            else         { a=c; c=d; d=a+gr*(b-a); }
        }
        double phi = 0.5*(a+b);
        b1x = r_outer*cos(phi); b1y = r_outer*sin(phi);
    }
    b2x = b1x; b2y = b1y;
    for (int fp = 0; fp < fp_iters; fp++) {
        {
            double ang_s = atan2(b1y, b1x);
            double ang_t = atan2(gy, gx);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_inner*cos(c), cy = r_inner*sin(c);
                double d1 = sqrt((cx-b1x)*(cx-b1x)+(cy-b1y)*(cy-b1y));
                double d2 = sqrt((cx-gx)*(cx-gx)+(cy-gy)*(cy-gy));
                double fc = d1/v_adipose + d2/v_fibro;
                double dx = r_inner*cos(d), dy = r_inner*sin(d);
                d1 = sqrt((dx-b1x)*(dx-b1x)+(dy-b1y)*(dy-b1y));
                d2 = sqrt((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy));
                double fd = d1/v_adipose + d2/v_fibro;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b2x = r_inner*cos(phi); b2y = r_inner*sin(phi);
        }
        {
            double ang_s = atan2(ayb, axb);
            double ang_t = atan2(b2y, b2x);
            double lo = fmin(ang_s, ang_t) - 0.75;
            double hi = fmax(ang_s, ang_t) + 0.75;
            double a = lo, b = hi;
            double c = b - gr*(b-a), d = a + gr*(b-a);
            for (int it = 0; it < n_iter; it++) {
                double cx = r_outer*cos(c), cy = r_outer*sin(c);
                double d1 = sqrt((cx-axb)*(cx-axb)+(cy-ayb)*(cy-ayb));
                double d2 = sqrt((cx-b2x)*(cx-b2x)+(cy-b2y)*(cy-b2y));
                double fc = d1/v_air + d2/v_adipose;
                double dx = r_outer*cos(d), dy = r_outer*sin(d);
                d1 = sqrt((dx-axb)*(dx-axb)+(dy-ayb)*(dy-ayb));
                d2 = sqrt((dx-b2x)*(dx-b2x)+(dy-b2y)*(dy-b2y));
                double fd = d1/v_air + d2/v_adipose;
                if (fc < fd) { b=d; d=c; c=b-gr*(b-a); }
                else         { a=c; c=d; d=a+gr*(b-a); }
            }
            double phi = 0.5*(a+b);
            b1x = r_outer*cos(phi); b1y = r_outer*sin(phi);
        }
    }
    d_air = sqrt((b1x-axb)*(b1x-axb)+(b1y-ayb)*(b1y-ayb));
    d_adi = sqrt((b2x-b1x)*(b2x-b1x)+(b2y-b1y)*(b2y-b1y));
    d_fib = sqrt((gx-b2x)*(gx-b2x)+(gy-b2y)*(gy-b2y));
    total_rx = d_air/v_air + d_adi/v_adipose + d_fib/v_fibro;

    delay_out[idx] = total_tx + total_rx;
}
"""

_noskin_cuda_kernel = None
if HAS_GPU:
    import cupy as cp
    _noskin_cuda_kernel = cp.RawKernel(_BENT_RAY_NOSKIN_CUDA_KERNEL,
                                       'bent_ray_noskin_kernel')


def bent_ray_noskin_delay(ant_x, ant_y, ant_x_b, ant_y_b,
                          grid_x_flat, grid_y_flat,
                          breast_radius_m, fib_radius_m,
                          v_air, v_adipose, v_fibro,
                          fixed_point_iters=3, n_gs_iters=40):
    """
    Multi-layer bent-ray TANPA SKIN: Air → Adipose → Fibro.
    Uses CUDA kernel on GPU, CPU fallback otherwise.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_flat)
    r_outer = breast_radius_m
    r_inner = max(fib_radius_m, 1e-6)

    if HAS_GPU and _noskin_cuda_kernel is not None:
        import cupy as cp
        d_ant_x = cp.asarray(ant_x, dtype=cp.float64)
        d_ant_y = cp.asarray(ant_y, dtype=cp.float64)
        d_ant_x_b = cp.asarray(ant_x_b, dtype=cp.float64)
        d_ant_y_b = cp.asarray(ant_y_b, dtype=cp.float64)
        d_gx = cp.asarray(grid_x_flat, dtype=cp.float64)
        d_gy = cp.asarray(grid_y_flat, dtype=cp.float64)
        d_delay = cp.zeros(n_ant * n_pix, dtype=cp.float64)

        total = n_ant * n_pix
        threads = 256
        blocks = (total + threads - 1) // threads

        _noskin_cuda_kernel(
            (blocks,), (threads,),
            (d_ant_x, d_ant_y, d_ant_x_b, d_ant_y_b,
             d_gx, d_gy, d_delay,
             n_ant, n_pix, r_outer, r_inner,
             v_air, v_adipose, v_fibro,
             n_gs_iters, fixed_point_iters)
        )
        return cp.asnumpy(d_delay).reshape(n_ant, n_pix)
    else:
        delay = np.zeros((n_ant, n_pix))
        for i in range(n_ant):
            for p in range(n_pix):
                delay[i, p] = _bent_ray_noskin_single_cpu(
                    ant_x[i], ant_y[i], ant_x_b[i], ant_y_b[i],
                    grid_x_flat[p], grid_y_flat[p],
                    r_outer, r_inner, v_air, v_adipose, v_fibro,
                    fixed_point_iters, n_gs_iters)
        return delay


def _bent_ray_noskin_single_cpu(ax, ay, axb, ayb, gx, gy,
                                r_outer, r_inner, v_air, v_adipose,
                                v_fibro, fp_iters, n_iter):
    """CPU fallback for single antenna-pixel no-skin bent-ray."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def _gs(sx, sy, tx, ty, r, v_out, v_in):
        ang_s = np.arctan2(sy, sx)
        ang_t = np.arctan2(ty, tx)
        lo = min(ang_s, ang_t) - 0.75
        hi = max(ang_s, ang_t) + 0.75
        a, b = lo, hi
        c = b - gr*(b-a); d = a + gr*(b-a)
        for _ in range(n_iter):
            cx, cy = r*np.cos(c), r*np.sin(c)
            fc = np.sqrt((cx-sx)**2+(cy-sy)**2)/v_out + np.sqrt((cx-tx)**2+(cy-ty)**2)/v_in
            dx, dy = r*np.cos(d), r*np.sin(d)
            fd = np.sqrt((dx-sx)**2+(dy-sy)**2)/v_out + np.sqrt((dx-tx)**2+(dy-ty)**2)/v_in
            if fc < fd: b=d; d=c; c=b-gr*(b-a)
            else: a=c; c=d; d=a+gr*(b-a)
        phi = 0.5*(a+b)
        return r*np.cos(phi), r*np.sin(phi)

    def _leg(sx, sy):
        b1x, b1y = _gs(sx, sy, gx, gy, r_outer, v_air, v_fibro)
        for _ in range(fp_iters):
            b2x, b2y = _gs(b1x, b1y, gx, gy, r_inner, v_adipose, v_fibro)
            b1x, b1y = _gs(sx, sy, b2x, b2y, r_outer, v_air, v_adipose)
        d_air = np.sqrt((b1x-sx)**2+(b1y-sy)**2)
        d_adi = np.sqrt((b2x-b1x)**2+(b2y-b1y)**2)
        d_fib = np.sqrt((gx-b2x)**2+(gy-b2y)**2)
        return d_air/v_air + d_adi/v_adipose + d_fib/v_fibro

    return _leg(ax, ay) + _leg(axb, ayb)

def _bent_ray_single_cpu(ax, ay, axb, ayb, gx, gy,
                         r_outer, r_inner, v_air, v_skin, v_interior,
                         fp_iters, n_iter):
    """Single antenna-pixel bent-ray on CPU. Used only as fallback."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def _gs_solve(sx, sy, tx, ty, r, v_out, v_in):
        ang_s = np.arctan2(sy, sx)
        ang_t = np.arctan2(ty, tx)
        lo = min(ang_s, ang_t) - 0.75
        hi = max(ang_s, ang_t) + 0.75
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        for _ in range(n_iter):
            cx, cy = r * np.cos(c), r * np.sin(c)
            fc = (np.sqrt((cx-sx)**2 + (cy-sy)**2) / v_out +
                  np.sqrt((cx-tx)**2 + (cy-ty)**2) / v_in)
            dx, dy = r * np.cos(d), r * np.sin(d)
            fd = (np.sqrt((dx-sx)**2 + (dy-sy)**2) / v_out +
                  np.sqrt((dx-tx)**2 + (dy-ty)**2) / v_in)
            if fc < fd:
                b = d; d = c; c = b - gr * (b - a)
            else:
                a = c; c = d; d = a + gr * (b - a)
        phi = 0.5 * (a + b)
        return r * np.cos(phi), r * np.sin(phi)

    def _leg(sx, sy):
        b1x, b1y = _gs_solve(sx, sy, gx, gy, r_outer, v_air, v_interior)
        for _ in range(fp_iters):
            b2x, b2y = _gs_solve(b1x, b1y, gx, gy, r_inner, v_skin, v_interior)
            b1x, b1y = _gs_solve(sx, sy, b2x, b2y, r_outer, v_air, v_skin)
        d_air = np.sqrt((b1x-sx)**2 + (b1y-sy)**2)
        d_skin = np.sqrt((b2x-b1x)**2 + (b2y-b1y)**2)
        d_int = np.sqrt((gx-b2x)**2 + (gy-b2y)**2)
        return d_air/v_air + d_skin/v_skin + d_int/v_interior

    return _leg(ax, ay) + _leg(axb, ayb)

# ===========================================================================
# Geometry-Informed Bent-Ray: Air → Fibro dengan boundary dari STL
# Single-layer, tapi boundary irregular (bukan lingkaran)
# ===========================================================================

def load_stl_boundary(stl_path, z_frac=0.80, n_points=360):
    """Load STL, extract 2D cross-section boundary at z_frac, return centered (bx, by) in meters."""
    import trimesh
    mesh = trimesh.load(str(stl_path), force='mesh')
    z = mesh.bounds[0][2] + z_frac * (mesh.bounds[1][2] - mesh.bounds[0][2])
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        raise ValueError(f"No cross-section at z={z:.1f}")
    
    pts = section.vertices
    x, y = pts[:, 0], pts[:, 1]
    cx, cy = np.mean(x), np.mean(y)
    x -= cx; y -= cy
    
    angles = np.arctan2(y, x)
    si = np.argsort(angles)
    x, y, angles = x[si], y[si], angles[si]
    radii = np.sqrt(x**2 + y**2)
    
    from scipy.interpolate import interp1d
    ae = np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi])
    re = np.concatenate([radii, radii, radii])
    es = np.argsort(ae); ae, re = ae[es], re[es]
    au = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    ri = interp1d(ae, re, kind='linear', bounds_error=False, fill_value='extrapolate')(au)
    
    bx = (ri * np.cos(au)) / 1000.0  # mm → meters
    by = (ri * np.sin(au)) / 1000.0
    return bx.astype(np.float64), by.astype(np.float64)


_GIBR_CUDA_KERNEL = r"""
extern "C" __global__
void gibr_kernel(
    const double* __restrict__ ant_x,
    const double* __restrict__ ant_y,
    const double* __restrict__ ant_x_b,
    const double* __restrict__ ant_y_b,
    const double* __restrict__ grid_x,
    const double* __restrict__ grid_y,
    const double* __restrict__ bnd_x,
    const double* __restrict__ bnd_y,
    double* __restrict__ delay_out,
    const int n_ant,
    const int n_pix,
    const int n_bnd,
    const double v_air,
    const double v_tissue,
    const int n_iter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_ant * n_pix;
    if (idx >= total) return;

    int i_ant = idx / n_pix;
    int i_pix = idx % n_pix;

    double ax = ant_x[i_ant], ay = ant_y[i_ant];
    double axb = ant_x_b[i_ant], ayb = ant_y_b[i_ant];
    double gx = grid_x[i_pix], gy = grid_y[i_pix];
    const double gr = 0.6180339887498949;

    // Golden-section search along boundary for minimum-time refraction point
    // Search over boundary parameter t in [0, n_bnd)
    // Boundary point: bnd_x[t], bnd_y[t]

    // --- LEG TX: antenna -> pixel via boundary ---
    double lo = 0.0, hi = (double)(n_bnd - 1);
    double c = hi - gr * (hi - lo);
    double d = lo + gr * (hi - lo);

    for (int it = 0; it < n_iter; it++) {
        int ci = (int)(c + 0.5) % n_bnd;
        int di = (int)(d + 0.5) % n_bnd;
        double bcx = bnd_x[ci], bcy = bnd_y[ci];
        double bdx = bnd_x[di], bdy = bnd_y[di];

        double fc = sqrt((bcx-ax)*(bcx-ax)+(bcy-ay)*(bcy-ay))/v_air
                  + sqrt((bcx-gx)*(bcx-gx)+(bcy-gy)*(bcy-gy))/v_tissue;
        double fd = sqrt((bdx-ax)*(bdx-ax)+(bdy-ay)*(bdy-ay))/v_air
                  + sqrt((bdx-gx)*(bdx-gx)+(bdy-gy)*(bdy-gy))/v_tissue;

        if (fc < fd) { hi = d; d = c; c = hi - gr*(hi-lo); }
        else { lo = c; c = d; d = lo + gr*(hi-lo); }
    }

    int best_i = (int)(0.5*(lo+hi) + 0.5) % n_bnd;
    double bx = bnd_x[best_i], by = bnd_y[best_i];
    double d_air_tx = sqrt((bx-ax)*(bx-ax)+(by-ay)*(by-ay));
    double d_tis_tx = sqrt((bx-gx)*(bx-gx)+(by-gy)*(by-gy));
    double total_tx = d_air_tx/v_air + d_tis_tx/v_tissue;

    // --- LEG RX: antenna_b -> pixel via boundary ---
    lo = 0.0; hi = (double)(n_bnd - 1);
    c = hi - gr*(hi-lo); d = lo + gr*(hi-lo);

    for (int it = 0; it < n_iter; it++) {
        int ci = (int)(c + 0.5) % n_bnd;
        int di = (int)(d + 0.5) % n_bnd;
        double bcx = bnd_x[ci], bcy = bnd_y[ci];
        double bdx = bnd_x[di], bdy = bnd_y[di];

        double fc = sqrt((bcx-axb)*(bcx-axb)+(bcy-ayb)*(bcy-ayb))/v_air
                  + sqrt((bcx-gx)*(bcx-gx)+(bcy-gy)*(bcy-gy))/v_tissue;
        double fd = sqrt((bdx-axb)*(bdx-axb)+(bdy-ayb)*(bdy-ayb))/v_air
                  + sqrt((bdx-gx)*(bdx-gx)+(bdy-gy)*(bdy-gy))/v_tissue;

        if (fc < fd) { hi = d; d = c; c = hi - gr*(hi-lo); }
        else { lo = c; c = d; d = lo + gr*(hi-lo); }
    }

    best_i = (int)(0.5*(lo+hi) + 0.5) % n_bnd;
    bx = bnd_x[best_i]; by = bnd_y[best_i];
    double d_air_rx = sqrt((bx-axb)*(bx-axb)+(by-ayb)*(by-ayb));
    double d_tis_rx = sqrt((bx-gx)*(bx-gx)+(by-gy)*(by-gy));
    double total_rx = d_air_rx/v_air + d_tis_rx/v_tissue;

    delay_out[idx] = total_tx + total_rx;
}
"""

_gibr_cuda_kernel = None
if HAS_GPU:
    import cupy as cp
    _gibr_cuda_kernel = cp.RawKernel(_GIBR_CUDA_KERNEL, 'gibr_kernel')


def geometry_informed_bent_ray_delay(ant_x, ant_y, ant_x_b, ant_y_b,
                                      grid_x_flat, grid_y_flat,
                                      boundary_x, boundary_y,
                                      v_air, v_tissue,
                                      n_gs_iters=60):
    """
    Single-layer bent-ray with arbitrary boundary from STL.
    Air outside boundary (v_air), tissue inside (v_tissue).
    Boundary is irregular contour, not a circle.
    """
    n_ant = len(ant_x)
    n_pix = len(grid_x_flat)
    n_bnd = len(boundary_x)

    if HAS_GPU and _gibr_cuda_kernel is not None:
        import cupy as cp
        d_ax = cp.asarray(ant_x, dtype=cp.float64)
        d_ay = cp.asarray(ant_y, dtype=cp.float64)
        d_axb = cp.asarray(ant_x_b, dtype=cp.float64)
        d_ayb = cp.asarray(ant_y_b, dtype=cp.float64)
        d_gx = cp.asarray(grid_x_flat, dtype=cp.float64)
        d_gy = cp.asarray(grid_y_flat, dtype=cp.float64)
        d_bx = cp.asarray(boundary_x, dtype=cp.float64)
        d_by = cp.asarray(boundary_y, dtype=cp.float64)
        d_delay = cp.zeros(n_ant * n_pix, dtype=cp.float64)

        total = n_ant * n_pix
        threads = 256
        blocks = (total + threads - 1) // threads

        _gibr_cuda_kernel(
            (blocks,), (threads,),
            (d_ax, d_ay, d_axb, d_ayb, d_gx, d_gy, d_bx, d_by, d_delay,
             n_ant, n_pix, n_bnd, v_air, v_tissue, n_gs_iters)
        )
        return cp.asnumpy(d_delay).reshape(n_ant, n_pix)
    else:
        # CPU fallback
        delay = np.zeros((n_ant, n_pix))
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        for i in range(n_ant):
            for p in range(n_pix):
                delay[i, p] = _gibr_single_cpu(
                    ant_x[i], ant_y[i], ant_x_b[i], ant_y_b[i],
                    grid_x_flat[p], grid_y_flat[p],
                    boundary_x, boundary_y, v_air, v_tissue, n_gs_iters, gr)
        return delay


def _gibr_single_cpu(ax, ay, axb, ayb, gx, gy, bx, by, v_air, v_tis, n_iter, gr):
    n_bnd = len(bx)
    def _leg(sx, sy):
        lo, hi = 0.0, float(n_bnd - 1)
        c = hi - gr*(hi-lo); d = lo + gr*(hi-lo)
        for _ in range(n_iter):
            ci = int(c + 0.5) % n_bnd
            di = int(d + 0.5) % n_bnd
            fc = (np.sqrt((bx[ci]-sx)**2+(by[ci]-sy)**2)/v_air +
                  np.sqrt((bx[ci]-gx)**2+(by[ci]-gy)**2)/v_tis)
            fd = (np.sqrt((bx[di]-sx)**2+(by[di]-sy)**2)/v_air +
                  np.sqrt((bx[di]-gx)**2+(by[di]-gy)**2)/v_tis)
            if fc < fd: hi=d; d=c; c=hi-gr*(hi-lo)
            else: lo=c; c=d; d=lo+gr*(hi-lo)
        bi = int(0.5*(lo+hi)+0.5) % n_bnd
        return (np.sqrt((bx[bi]-sx)**2+(by[bi]-sy)**2)/v_air +
                np.sqrt((bx[bi]-gx)**2+(by[bi]-gy)**2)/v_tis)
    return _leg(ax, ay) + _leg(axb, ayb)