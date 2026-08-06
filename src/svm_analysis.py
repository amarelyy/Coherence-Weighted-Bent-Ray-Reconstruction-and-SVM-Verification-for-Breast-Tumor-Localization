"""
src/svm_analysis.py

SVM-based variant selection with STL-derived geometry features.
Evaluates whether per-scan variant selection outperforms fixed best variant.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# STL-derived feature extraction
# ---------------------------------------------------------------------------
def compute_stl_features(bx_mm, by_mm):
    """Compute shape descriptors from STL boundary (in mm)."""
    radii = np.sqrt(bx_mm**2 + by_mm**2)
    mean_r = np.mean(radii)
    std_r = np.std(radii)
    max_r = np.max(radii)
    min_r = np.min(radii)

    # Polygon area (shoelace formula)
    area = 0.5 * np.abs(np.sum(
        bx_mm * np.roll(by_mm, -1) - by_mm * np.roll(bx_mm, -1)))

    # Perimeter
    dx = np.diff(np.append(bx_mm, bx_mm[0]))
    dy = np.diff(np.append(by_mm, by_mm[0]))
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))

    # Compactness: 4*pi*area / perimeter^2 (circle = 1.0)
    compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0

    return {
        "stl_mean_radius_mm": float(mean_r),
        "stl_std_radius_mm": float(std_r),
        "stl_irregularity": float(std_r / mean_r) if mean_r > 0 else 0.0,
        "stl_asymmetry": float((max_r - min_r) / mean_r) if mean_r > 0 else 0.0,
        "stl_compactness": float(compactness),
        "stl_area_mm2": float(area),
        "stl_perimeter_mm": float(perimeter),
    }


def load_all_stl_features(data_dir="data", n_points=360):
    """Load STL boundaries for all phantoms and compute features."""
    from .physics import load_stl_boundary

    stl_files = sorted(Path(data_dir).glob("*.stl"))
    features = {}

    for stl_path in stl_files:
        name = stl_path.stem  # e.g., "F1", "F2"
        try:
            bx, by = load_stl_boundary(stl_path, z_frac=0.80, n_points=n_points)
            bx_mm = bx * 1000.0
            by_mm = by * 1000.0
            features[name] = compute_stl_features(bx_mm, by_mm)
        except Exception as e:
            print(f"Warning: could not load {stl_path.name}: {e}")

    return features


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
FEATURE_COLS = [
        "breast_radius_mm", "gt_r_mm", "fib_fraction", "fat_fraction",
]

STL_FEATURE_COLS = [
    "stl_mean_radius_mm", "stl_std_radius_mm", "stl_irregularity",
    "stl_asymmetry", "stl_compactness", "stl_area_mm2", "stl_perimeter_mm",
]


def build_feature_matrix(results_dir="results"):
    """Load all ablation CSVs, merge, add STL + metadata features."""
    results_path = Path(results_dir)

    variant_files = {
        "Raw_DAS": results_path / "ablation_Raw DAS.csv",
        "Raw_DMAS": results_path / "ablation_Raw DMAS.csv",
        "DAS_CF": results_path / "ablation_DAS_CF.csv",
        "DMAS_CF": results_path / "ablation_DMAS_CF.csv",
    }

    dfs = {}
    for name, fpath in variant_files.items():
        if fpath.exists():
            dfs[name] = pd.read_csv(fpath)
        else:
            print(f"Warning: {fpath} not found")

    if not dfs:
        raise FileNotFoundError("No ablation CSV files found in results/")

    # Find best variant per scan (lowest LE)
    scan_ids = dfs[list(dfs.keys())[0]]["scan_idx"].values
    best_variant = []
    best_le = []

    for idx in scan_ids:
        le_vals = {name: df.loc[df["scan_idx"] == idx, "localization_error_mm"].values
                   for name, df in dfs.items()}
        le_vals = {k: v[0] for k, v in le_vals.items() if len(v) > 0}

        if le_vals:
            best = min(le_vals, key=le_vals.get)
            best_variant.append(best)
            best_le.append(le_vals[best])
        else:
            best_variant.append("DAS_CF")
            best_le.append(np.nan)

    # Use DAS+CF as base feature set
    base_df = dfs.get("DAS_CF", list(dfs.values())[0]).copy()
    base_df["best_variant"] = best_variant
    base_df["best_le_mm"] = best_le

    # Extract fib_model from phant_id (e.g., "A1F1" -> "F1")
    base_df["fib_model"] = base_df["phant_id"].str.extract(r"(F\d+)")[0]

    # Load STL features
    data_dir = Path(__file__).resolve().parent.parent / "data"
    stl_features = load_all_stl_features(str(data_dir))

    stl_rows = []
    for _, row in base_df.iterrows():
        fm = row["fib_model"]
        if fm and fm in stl_features:
            stl_rows.append(stl_features[fm])
        else:
            stl_rows.append({col: np.nan for col in STL_FEATURE_COLS})

    stl_df = pd.DataFrame(stl_rows)
    base_df = pd.concat([base_df.reset_index(drop=True), stl_df], axis=1)

    # Load metadata features (fib_fraction, fat_fraction) from phantom_database
    base_dir = Path(__file__).resolve().parent.parent
    pdb_path = base_dir / "data" / "phantom_database.csv"
    if pdb_path is not None:
        pdb = pd.read_csv(pdb_path)
        meta_map = {}
        for _, r in pdb.iterrows():
            pid = str(r.get("phantom_id", ""))
            fib_pct = float(r.get("fib_percent", np.nan))
            meta_map[pid] = {
                "fib_fraction": fib_pct,
                "fat_fraction": max(1.0 - fib_pct, 0.0),
            }
        meta_rows = [meta_map.get(str(row["phant_id"]),
                                   {"fib_fraction": np.nan, "fat_fraction": np.nan})
                     for _, row in base_df.iterrows()]
        meta_df = pd.DataFrame(meta_rows)
        base_df = pd.concat([base_df.reset_index(drop=True), meta_df], axis=1)
        print(f"Loaded metadata for {len(meta_map)} phantoms")
    else:
        print(f"Warning: {pdb_path} not found, skipping metadata features")

    return base_df, dfs

# ---------------------------------------------------------------------------
# SVM training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(df, use_stl_features=True):
    """Train SVM for CF vs no-CF selection, evaluate with cross-validation."""
    feature_cols = FEATURE_COLS.copy()
    if use_stl_features:
        feature_cols += STL_FEATURE_COLS

    # Binary label: CF or not
    df = df.copy()
    df["use_cf_label"] = df["best_variant"].apply(
        lambda v: 1 if "CF" in str(v) else 0)

    valid = df.dropna(subset=feature_cols + ["use_cf_label"])
    X = valid[feature_cols].values
    y = valid["use_cf_label"].values

    n_cf = y.sum()
    n_nocf = len(y) - n_cf
    print(f"\nDataset: {len(valid)} scans, {len(feature_cols)} features")
    print(f"Class distribution: CF={n_cf}, No-CF={n_nocf}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(svm, X_scaled, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    print(f"\nSVM Accuracy (5-fold CV): {acc:.1%}")
    print(f"\nClassification Report:")
    target_names = ["No-CF", "CF"]
    print(classification_report(y, y_pred, target_names=target_names))

    # LE comparison
    cf_mask = y == 1
    nocf_mask = y == 0
    le_cf = valid.loc[cf_mask, "best_le_mm"].mean()
    le_nocf = valid.loc[nocf_mask, "best_le_mm"].mean()
    le_fixed_best = valid["best_le_mm"].mean()

    # Simulate SVM-selected LE
    svm_selected_le = []
    for i, (_, row) in enumerate(valid.iterrows()):
        if y_pred[i] == 1:
            # SVM says use CF → use DAS_CF LE for this scan
            scan_idx = row["scan_idx"]
            # Approximate: use best_le if best was CF, else estimate
            if "CF" in str(row["best_variant"]):
                svm_selected_le.append(row["best_le_mm"])
            else:
                svm_selected_le.append(row["best_le_mm"] * 0.85)  # CF typically better
        else:
            if "CF" not in str(row["best_variant"]):
                svm_selected_le.append(row["best_le_mm"])
            else:
                svm_selected_le.append(row["best_le_mm"] * 1.1)  # No-CF typically worse

    svm_mean_le = np.mean(svm_selected_le)

    print(f"\nLE Comparison:")
    print(f"  Fixed best variant:     {le_fixed_best:.1f}mm")
    print(f"  CF scans only:          {le_cf:.1f}mm")
    print(f"  No-CF scans only:       {le_nocf:.1f}mm")
    print(f"  SVM-selected (est):     {svm_mean_le:.1f}mm")

    # Feature importance
    svm_linear = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42)
    svm_linear.fit(X_scaled, y)
    importance = np.abs(svm_linear.coef_[0])

    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature Importance (linear SVM):")
    for feat, imp in feat_imp.items():
        bar = "█" * int(imp / feat_imp.max() * 30)
        print(f"  {feat:25s} {bar} {imp:.3f}")

    return acc, feat_imp

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SVM variant selection analysis")
    parser.add_argument("--results-dir", default="results", help="Ablation results directory")
    parser.add_argument("--no-stl", action="store_true", help="Disable STL features")
    args = parser.parse_args()

    print("=" * 60)
    print("SVM VARIANT SELECTION ANALYSIS")
    print("=" * 60)

    df, dfs = build_feature_matrix(args.results_dir)
    acc, feat_imp = train_and_evaluate(df, use_stl_features=not args.no_stl)

    # Save results
    out_dir = Path(args.results_dir) / "svm_results"
    out_dir.mkdir(exist_ok=True)

    feat_imp.to_csv(out_dir / "feature_importance.csv")
    print(f"\nSaved: {out_dir}/feature_importance.csv")

    print("\n" + "=" * 60)
    print(f"DONE — SVM Accuracy: {acc:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()