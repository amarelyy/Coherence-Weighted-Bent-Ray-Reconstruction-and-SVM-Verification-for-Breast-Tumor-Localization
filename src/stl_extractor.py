"""
src/stl_extractor.py
Load STL mesh files from UM-BMID dataset and extract 2D cross-section
boundaries at the imaging plane for use in bent-ray delay computation.

Usage:
    python -m src.stl_extractor --stl-dir data/stl --output-dir data/boundaries
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt


def load_stl(stl_path):
    """Load an STL file and return a trimesh object."""
    mesh = trimesh.load(stl_path, force='mesh')
    print(f"  Loaded: {Path(stl_path).name}")
    print(f"    Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"    Bounds: {mesh.bounds[0]} -> {mesh.bounds[1]}")
    print(f"    Extent (mm): {mesh.extents}")
    return mesh


def find_imaging_plane_z(mesh, n_samples=50):
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    z_candidates = np.linspace(z_min + 0.1 * (z_max - z_min),
                                z_max - 0.1 * (z_max - z_min),
                                n_samples)
    best_z = z_candidates[len(z_candidates)//2]
    best_n_verts = 0

    for z in z_candidates:
        try:
            section = mesh.section(plane_origin=[0, 0, z],
                                   plane_normal=[0, 0, 1])
            if section is not None and hasattr(section, 'vertices'):
                n_verts = len(section.vertices)
                if n_verts > best_n_verts:
                    best_n_verts = n_verts
                    best_z = z
        except Exception:
            continue

    return best_z, best_n_verts


def extract_boundary_at_z(mesh, z_plane, n_points=360):
    section = mesh.section(plane_origin=[0, 0, z_plane],
                           plane_normal=[0, 0, 1])
    if section is None:
        raise ValueError(f"No cross-section found at z={z_plane:.2f}")

    # section.vertices is (N, 3) — use x, y directly
    points = np.array(section.vertices)
    x = points[:, 0]
    y = points[:, 1]

    cx, cy = np.mean(x), np.mean(y)
    x -= cx
    y -= cy

    angles = np.arctan2(y, x)
    sort_idx = np.argsort(angles)
    x = x[sort_idx]
    y = y[sort_idx]
    angles_sorted = angles[sort_idx]
    radii = np.sqrt(x**2 + y**2)

    from scipy.interpolate import interp1d
    angles_ext = np.concatenate([angles_sorted - 2*np.pi,
                                  angles_sorted,
                                  angles_sorted + 2*np.pi])
    radii_ext = np.concatenate([radii, radii, radii])
    ext_sort = np.argsort(angles_ext)
    angles_ext = angles_ext[ext_sort]
    radii_ext = radii_ext[ext_sort]

    angles_uniform = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    interp_func = interp1d(angles_ext, radii_ext, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    radii_uniform = interp_func(angles_uniform)

    x_uniform = radii_uniform * np.cos(angles_uniform)
    y_uniform = radii_uniform * np.sin(angles_uniform)

    return x_uniform, y_uniform, (cx, cy)

def process_stl_file(stl_path, output_dir, n_points=360):
    """Process a single STL file: find best z-plane, extract boundary, save."""
    mesh = load_stl(stl_path)
    name = Path(stl_path).stem

    # Find optimal imaging plane
    z_best, area_best = find_imaging_plane_z(mesh)
    print(f"    Best z-plane: {z_best:.2f} mm (area: {area_best:.1f} mm²)")

    # Extract boundary
    bx, by, center = extract_boundary_at_z(mesh, z_best, n_points)

    # Compute stats
    radii = np.sqrt(bx**2 + by**2)
    print(f"    Boundary radius: mean={np.mean(radii):.2f}, "
          f"min={np.min(radii):.2f}, max={np.max(radii):.2f} mm")
    print(f"    Centroid offset: ({center[0]:.2f}, {center[1]:.2f}) mm")

    # Save boundary data
    out_path = Path(output_dir) / f"{name}_boundary.json"
    boundary_data = {
        "name": name,
        "stl_file": str(stl_path),
        "z_plane_mm": float(z_best),
        "cross_section_area_mm2": float(area_best),
        "centroid_offset_mm": [float(center[0]), float(center[1])],
        "n_points": n_points,
        "boundary_x_mm": bx.tolist(),
        "boundary_y_mm": by.tolist(),
        "radius_stats_mm": {
            "mean": float(np.mean(radii)),
            "min": float(np.min(radii)),
            "max": float(np.max(radii)),
            "std": float(np.std(radii)),
        }
    }
    with open(out_path, 'w') as f:
        json.dump(boundary_data, f, indent=2)
    print(f"    Saved: {out_path}")

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3D mesh view
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
                       mesh.vertices[:, 2],
                       triangles=mesh.faces, alpha=0.3, color='lightblue')
    ax3d.set_title(f'{name} - 3D Mesh')
    ax3d.set_xlabel('X (mm)')
    ax3d.set_ylabel('Y (mm)')
    ax3d.set_zlabel('Z (mm)')

    # 2D cross-section
    ax2d = axes[1]
    ax2d.fill(bx, by, alpha=0.3, color='coral', label='Boundary')
    ax2d.plot(bx, by, 'r-', linewidth=1.5)
    ax2d.plot(0, 0, 'k+', markersize=10, label='Centroid')
    circle = plt.Circle((0, 0), np.mean(radii), fill=False,
                         linestyle='--', color='blue', label='Mean radius circle')
    ax2d.add_patch(circle)
    ax2d.set_aspect('equal')
    ax2d.set_title(f'{name} - Cross-section at z={z_best:.1f} mm')
    ax2d.set_xlabel('X (mm)')
    ax2d.set_ylabel('Y (mm)')
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = Path(output_dir) / f"{name}_cross_section.png"
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f"    Visualization: {viz_path}")

    return boundary_data


def main():
    parser = argparse.ArgumentParser(description="Extract 2D boundaries from UM-BMID STL files")
    parser.add_argument("--stl-dir", type=str, default="data/stl",
                        help="Directory containing STL files")
    parser.add_argument("--output-dir", type=str, default="data/boundaries",
                        help="Output directory for boundary JSON files")
    parser.add_argument("--n-points", type=int, default=360,
                        help="Number of points in resampled boundary")
    parser.add_argument("--pattern", type=str, default="*.stl",
                        help="File pattern to match (e.g., 'F*.stl' for fibro only)")
    args = parser.parse_args()

    stl_dir = Path(args.stl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = sorted(stl_dir.glob(args.pattern))
    if not stl_files:
        print(f"No STL files found in {stl_dir} matching '{args.pattern}'")
        print(f"\nExpected folder structure:")
        print(f"  {stl_dir}/")
        print(f"    F1.stl, F2.stl, ... F14.stl  (fibro shells)")
        print(f"    A1.stl, A2.stl, ... A16.stl  (adipose shells)")
        print(f"\nDownload STL files from: https://bit.ly/UM-bmid")
        print(f"  Navigate to: UM-BMID/phantoms/gen-two/")
        return

    print(f"Found {len(stl_files)} STL files in {stl_dir}\n")

    all_boundaries = {}
    for stl_path in stl_files:
        print(f"Processing: {stl_path.name}")
        try:
            data = process_stl_file(stl_path, output_dir, args.n_points)
            all_boundaries[data["name"]] = data
            print()
        except Exception as e:
            print(f"  ERROR: {e}\n")

    # Save summary
    summary_path = output_dir / "boundaries_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_boundaries, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"Total processed: {len(all_boundaries)}/{len(stl_files)}")


if __name__ == "__main__":
    main()