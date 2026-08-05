"""
src/svm_analysis.py

Two SVM modules for post-ablation analysis:
1. Feature Importance — which features most influence localization accuracy
2. Adaptive Variant Selection — predict best variant from phantom metadata
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
import joblib

RECON_FEATURES = ["blob_area_px", "blob_compactness", "scr_db", "smr_db", "cnr", "cf_at_peak"]
PHANTOM_FEATURES = ["breast_radius_mm", "fib_fraction"]
ALL_IMPORTANCE_FEATURES = RECON_FEATURES + PHANTOM_FEATURES

VARIANT_SELECTOR_FEATURES = [
    "breast_radius_mm", "fib_fraction", "fat_fraction",
    "shell_volume", "fib_volume", "birads",
]


class FeatureImportanceSVM:
    def __init__(self, le_threshold_mm=20.0, C=1.0):
        self.le_threshold_mm = le_threshold_mm
        self.scaler = StandardScaler()
        self.svm_linear = LinearSVC(C=C, class_weight="balanced", max_iter=10000)
        self.svm_rbf = SVC(kernel="rbf", C=C, class_weight="balanced", probability=True)
        self.feature_cols = []
        self.is_trained = False
        self.results_ = {}

    def train(self, results_df, verbose=True):
        available = [c for c in ALL_IMPORTANCE_FEATURES if c in results_df.columns]
        self.feature_cols = available
        X = results_df[self.feature_cols].copy()
        if "cf_at_peak" in X.columns:
            X["cf_at_peak"] = X["cf_at_peak"].fillna(0.0)
        X = X.fillna(X.median()).values.astype(np.float64)
        y = (results_df["localization_error_mm"] <= self.le_threshold_mm).astype(int).values
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        if verbose:
            print(f"[FeatureSVM] {len(y)} samples | Good: {n_pos} ({n_pos/len(y):.1%}) | Bad: {n_neg}")
        X_scaled = self.scaler.fit_transform(X)
        n_splits = min(5, min(n_pos, n_neg))
        cv_scores = None
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.svm_rbf, X_scaled, y, cv=cv, scoring="accuracy")
            if verbose:
                print(f"[FeatureSVM] CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        self.svm_linear.fit(X_scaled, y)
        self.svm_rbf.fit(X_scaled, y)
        self.is_trained = True
        lw = np.abs(self.svm_linear.coef_[0])
        li = dict(zip(self.feature_cols, lw / lw.sum()))
        pr = permutation_importance(self.svm_rbf, X_scaled, y, n_repeats=30, random_state=42, scoring="accuracy")
        pi = dict(zip(self.feature_cols, pr.importances_mean))
        tp = sum(pi.values()) if sum(pi.values()) > 0 else 1.0
        pi_norm = {k: v / tp for k, v in pi.items()}
        rl = sorted(li.items(), key=lambda x: -x[1])
        rp = sorted(pi_norm.items(), key=lambda x: -x[1])
        self.results_ = {
            "linear_importance_norm": li, "permutation_importance_norm": pi_norm,
            "ranked_linear": rl, "ranked_permutation": rp,
            "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            "n_samples": len(y), "features_used": self.feature_cols,
        }
        if verbose:
            print(f"\n{'Rank':<6}{'Feature':<25}{'Linear':<18}{'Permutation':<18}")
            print("-" * 67)
            for i, ((fl, vl), (fp, vp)) in enumerate(zip(rl, rp)):
                print(f"{i+1:<6}{fl:<25}{vl:<18.4f}{vp:<18.4f}")
        return self.results_

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm_linear": self.svm_linear, "svm_rbf": self.svm_rbf,
                      "scaler": self.scaler, "feature_cols": self.feature_cols,
                      "le_threshold_mm": self.le_threshold_mm, "is_trained": self.is_trained,
                      "results_": self.results_}, path)
        print(f"[FeatureSVM] Saved to {path}")

    @classmethod
    def load(cls, path):
        b = joblib.load(path)
        obj = cls(le_threshold_mm=b["le_threshold_mm"])
        obj.svm_linear, obj.svm_rbf = b["svm_linear"], b["svm_rbf"]
        obj.scaler, obj.feature_cols = b["scaler"], b["feature_cols"]
        obj.is_trained, obj.results_ = b["is_trained"], b["results_"]
        return obj
    
class VariantSelectorSVM:
    def __init__(self, C=1.0, gamma="scale"):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced")
        self.feature_cols = []
        self.variant_names = []
        self.is_trained = False
        self.results_ = {}

    def _prepare_training_data(self, all_results_df):
        idx_col = "scan_idx" if "scan_idx" in all_results_df.columns else "phant_id"
        best_rows = []
        for scan_id, group in all_results_df.groupby(idx_col):
            best_idx = group["localization_error_mm"].idxmin()
            best_row = group.loc[best_idx]
            row_data = {"scan_id": scan_id, "best_le_mm": best_row["localization_error_mm"]}
            if "variant" in best_row.index:
                row_data["best_variant"] = best_row["variant"]
            elif "beamformer" in best_row.index:
                row_data["best_variant"] = str(best_row.get("beamformer", "unknown"))
            for col in VARIANT_SELECTOR_FEATURES:
                if col in best_row.index:
                    row_data[col] = best_row[col]
            best_rows.append(row_data)
        return pd.DataFrame(best_rows)

    def train(self, all_results_df, verbose=True):
        train_df = self._prepare_training_data(all_results_df)
        available = [c for c in VARIANT_SELECTOR_FEATURES if c in train_df.columns]
        self.feature_cols = available
        X = train_df[self.feature_cols].copy().fillna(train_df[self.feature_cols].median()).values.astype(np.float64)
        y_raw = train_df["best_variant"].values
        y = self.label_encoder.fit_transform(y_raw)
        self.variant_names = list(self.label_encoder.classes_)
        if verbose:
            print(f"[VariantSVM] {len(X)} scans, {len(self.variant_names)} variants")
            for v in self.variant_names:
                n = (y == self.label_encoder.transform([v])[0]).sum()
                print(f"  {v}: {n} ({n/len(y):.1%})")
        X_scaled = self.scaler.fit_transform(X)
        class_counts = np.bincount(y)
        n_splits = min(5, class_counts.min())
        cv_scores = None
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.svm, X_scaled, y, cv=cv, scoring="accuracy")
            if verbose:
                print(f"[VariantSVM] CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        baseline_acc = class_counts.max() / len(y)
        if verbose:
            print(f"[VariantSVM] Baseline (majority): {baseline_acc:.3f}")
        self.svm.fit(X_scaled, y)
        self.is_trained = True
        pr = permutation_importance(self.svm, X_scaled, y, n_repeats=30, random_state=42, scoring="accuracy")
        pi = dict(zip(self.feature_cols, pr.importances_mean))
        ranked = sorted(pi.items(), key=lambda x: -x[1])
        if verbose:
            print(f"\n[VariantSVM] Feature Importance:")
            for i, (f, v) in enumerate(ranked):
                print(f"  {i+1}. {f}: {v:.4f}")
        self.results_ = {
            "permutation_importance": pi, "ranked_features": ranked,
            "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            "baseline_accuracy": float(baseline_acc), "variant_names": self.variant_names,
            "n_scans": len(X), "mean_oracle_le_mm": float(train_df["best_le_mm"].mean()),
        }
        return self.results_

    def predict(self, phantom_features_dict):
        if not self.is_trained:
            raise RuntimeError("Train first")
        row = pd.DataFrame([phantom_features_dict])
        cols = [c for c in self.feature_cols if c in row.columns]
        X = row[cols].fillna(row[cols].median()).values.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        pred_idx = self.svm.predict(X_scaled)[0]
        proba = self.svm.predict_proba(X_scaled)[0]
        best = self.label_encoder.inverse_transform([pred_idx])[0]
        return {"best_variant": best, "confidence": float(proba[pred_idx]),
                "all_probabilities": dict(zip(self.variant_names, proba.tolist()))}

    def evaluate(self, all_results_df, verbose=True):
        if not self.is_trained:
            raise RuntimeError("Train first")
        train_df = self._prepare_training_data(all_results_df)
        X = train_df[self.feature_cols].copy().fillna(train_df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X.values.astype(np.float64))
        pred_variants = self.label_encoder.inverse_transform(self.svm.predict(X_scaled))
        idx_col = "scan_idx" if "scan_idx" in all_results_df.columns else "phant_id"
        svm_les = []
        for i, (_, row) in enumerate(train_df.iterrows()):
            mask = (all_results_df[idx_col] == row["scan_id"]) & (all_results_df["variant"] == pred_variants[i])
            matched = all_results_df[mask]
            svm_les.append(matched["localization_error_mm"].iloc[0] if len(matched) > 0 else row["best_le_mm"])
        svm_les = np.array(svm_les)
        oracle_les = train_df["best_le_mm"].values
        fixed_les = {}
        for v in all_results_df["variant"].unique():
            v_df = all_results_df[all_results_df["variant"] == v]
            if len(v_df) > 0:
                fixed_les[v] = v_df["localization_error_mm"].mean()
        results = {"svm_mean_le_mm": float(svm_les.mean()), "oracle_mean_le_mm": float(oracle_les.mean()),
                   "fixed_variant_means": fixed_les}
        if fixed_les:
            bf = min(fixed_les, key=fixed_les.get)
            results["best_fixed_variant"] = bf
            results["improvement_over_best_fixed"] = fixed_les[bf] - svm_les.mean()
        if verbose:
            print(f"\n[VariantSVM] === Evaluation ===")
            print(f"  Oracle:           {oracle_les.mean():.2f}mm")
            print(f"  SVM adaptive:     {svm_les.mean():.2f}mm")
            for v, le in sorted(fixed_les.items(), key=lambda x: x[1]):
                m = " <-- best fixed" if v == results.get("best_fixed_variant") else ""
                print(f"  Fixed {v:<20} {le:.2f}mm{m}")
            imp = results.get("improvement_over_best_fixed", 0)
            if imp > 0:
                print(f"  SVM improves over best fixed by: {imp:.2f}mm")
        self.results_["evaluation"] = results
        return results

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": self.svm, "scaler": self.scaler, "label_encoder": self.label_encoder,
                      "feature_cols": self.feature_cols, "variant_names": self.variant_names,
                      "is_trained": self.is_trained, "results_": self.results_}, path)
        print(f"[VariantSVM] Saved to {path}")

    @classmethod
    def load(cls, path):
        b = joblib.load(path)
        obj = cls()
        obj.svm, obj.scaler = b["svm"], b["scaler"]
        obj.label_encoder = b["label_encoder"]
        obj.feature_cols, obj.variant_names = b["feature_cols"], b["variant_names"]
        obj.is_trained, obj.results_ = b["is_trained"], b["results_"]
        return obj
    
def run_full_analysis(results_dir, output_dir=None):
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir / "svm"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(results_dir.glob("ablation_*.csv"))
    if not csv_files:
        print(f"[SVM] No ablation_*.csv in {results_dir}"); return None

    print(f"[SVM] Found {len(csv_files)} result files")
    all_dfs = []
    for cp in csv_files:
        df = pd.read_csv(cp)
        vn = cp.stem.replace("ablation_", "").replace("_", "+")
        df["variant"] = vn
        all_dfs.append(df)
        print(f"  {vn}: {len(df)} scans, mean LE={df['localization_error_mm'].mean():.2f}mm")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[SVM] Combined: {len(combined)} rows")

    print("\n" + "=" * 60)
    print("SVM #1: Feature Importance Analysis")
    print("=" * 60)
    feat_svm = FeatureImportanceSVM()
    feat_results = feat_svm.train(combined, verbose=True)
    feat_svm.save(output_dir / "feature_importance_svm.joblib")

    print("\n" + "=" * 60)
    print("SVM #2: Adaptive Variant Selection")
    print("=" * 60)
    var_svm = VariantSelectorSVM()
    var_results = var_svm.train(combined, verbose=True)
    var_eval = var_svm.evaluate(combined, verbose=True)
    var_svm.save(output_dir / "variant_selector_svm.joblib")

    def _ser(obj):
        if isinstance(obj, (np.floating, float)): return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, tuple): return list(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_ser(v) for v in obj]
        return obj

    report = {
        "feature_importance": {"ranked_linear": feat_results["ranked_linear"],
                               "ranked_permutation": feat_results["ranked_permutation"],
                               "cv_scores": feat_results["cv_scores"]},
        "variant_selection": {"cv_scores": var_results["cv_scores"],
                              "baseline_accuracy": var_results["baseline_accuracy"],
                              "evaluation": var_eval,
                              "feature_importance": var_results["ranked_features"]},
    }
    with open(output_dir / "svm_analysis_report.json", "w") as f:
        json.dump(_ser(report), f, indent=2)
    print(f"\n[SVM] Report saved: {output_dir / 'svm_analysis_report.json'}")
    combined.to_csv(output_dir / "combined_ablation.csv", index=False)
    return {"feature_svm": feat_svm, "variant_svm": var_svm, "combined_df": combined}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run_full_analysis(args.results_dir, args.output_dir)