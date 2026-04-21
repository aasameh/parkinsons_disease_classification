import argparse
import json
import collections
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from modeling import build_estimator, get_balanced_weights
from inference import predict_with_probs


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hierarchical two-stage SVM classifier")
    p.add_argument("--file-list-csv", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--show-progress", action="store_true")
    return p.parse_args()


def load_sample(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = arr[:, 1:]
    arr = arr.T
    if arr.shape[1] >= 1024:
        arr = arr[:, :1024]
    else:
        pad = np.zeros((6, 1024 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    # For balanced accuracy, scikit-learn handles multiclass automatically
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # precision, recall, f1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "confusion_matrix": cm.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist()
    }


def aggregate_metrics(fold_results):
    keys = fold_results[0].keys()
    mean_res = {}
    std_res = {}
    
    for k in keys:
        if k == "confusion_matrix":
            matrices = [np.array(r[k]) for r in fold_results]
            mean_res[k] = np.mean(matrices, axis=0).tolist()
            std_res[k] = np.std(matrices, axis=0).tolist()
        else:
            vals = [np.array(r[k]) for r in fold_results]
            mean_res[k] = np.mean(vals, axis=0).tolist()
            std_res[k] = np.std(vals, axis=0).tolist()
            
            # If it's a scalar, extract it from the list
            if isinstance(mean_res[k], list) and not isinstance(fold_results[0][k], list):
                mean_res[k] = mean_res[k]
                std_res[k] = std_res[k]
                
    return mean_res, std_res


def main():
    args = parse_args()
    
    artifact_root = Path(args.artifact_root).resolve()
    boss_cache_dir = str(artifact_root / "out" / "boss_cache")
    
    df = pd.read_csv(args.file_list_csv)
    
    # 1. Subject-level splitting
    # We need to stratify subjects. The terminal classes are uniquely identified by label_stage2
    # label_stage2 maps: -1 -> Healthy, 0 -> NonPD, 1 -> PD
    subject_df = df.drop_duplicates(subset=["subject_id"]).copy()
    subject_df = subject_df.reset_index(drop=True)
    
    subject_ids = subject_df["subject_id"].values
    subject_labels = subject_df["label_stage2"].values  # perfectly captures all 3 subclasses for stratification
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    stage1_results = []
    stage2_results = []
    e2e_results = []
    
    for fold, (train_subj_idx, test_subj_idx) in enumerate(skf.split(subject_ids, subject_labels), 1):
        print(f"--- Fold {fold}/{args.n_folds} ---")
        
        train_subjects = set(subject_ids[train_subj_idx])
        test_subjects = set(subject_ids[test_subj_idx])
        
        # Filter samples for Train / Test
        train_df = df[df["subject_id"].isin(train_subjects)].copy()
        test_df = df[df["subject_id"].isin(test_subjects)].copy()
        
        # -----------------------------------------------------
        # STAGE 1: Train Healthy(0) vs Diseased(1)
        # -----------------------------------------------------
        s1_train_df = train_df[train_df["label_stage1"].isin([0, 1])]
        X_train_s1 = np.stack([load_sample(p) for p in s1_train_df["path"]])
        y_train_s1 = s1_train_df["label_stage1"].values.astype(int)
        c_train_s1 = s1_train_df["gender"].values
        w_train_s1 = get_balanced_weights(y_train_s1, c_train_s1)
        
        if args.show_progress:
            print(f"Training Stage 1 (Fold {fold}): {len(X_train_s1)} samples")
            
        estimator_s1 = build_estimator((6, 1024), {}, boss_cache_dir, args.show_progress)
        estimator_s1.fit(X_train_s1, y_train_s1, sample_weight=w_train_s1)
        
        # Test Stage 1
        s1_test_df = test_df[test_df["label_stage1"].isin([0, 1])]
        y_true_s1 = s1_test_df["label_stage1"].values.astype(int)
        y_pred_s1 = []
        for p in s1_test_df["path"]:
            x = load_sample(p)
            # predict_with_probs expects (1, 6, 1024)
            pred, _ = predict_with_probs(estimator_s1, np.expand_dims(x, axis=0))
            y_pred_s1.append(pred)
            
        s1_metrics = compute_metrics(y_true_s1, y_pred_s1, labels=[0, 1])
        stage1_results.append(s1_metrics)
        print(f"Stage 1 -> Acc: {s1_metrics['accuracy']:.4f} | BAcc: {s1_metrics['balanced_accuracy']:.4f}")
        
        # -----------------------------------------------------
        # STAGE 2: Train NonPD(0) vs PD(1)
        # -----------------------------------------------------
        s2_train_df = train_df[train_df["label_stage2"].isin([0, 1])]
        X_train_s2 = np.stack([load_sample(p) for p in s2_train_df["path"]])
        y_train_s2 = s2_train_df["label_stage2"].values.astype(int)
        c_train_s2 = s2_train_df["gender"].values
        w_train_s2 = get_balanced_weights(y_train_s2, c_train_s2)
        
        if args.show_progress:
            print(f"Training Stage 2 (Fold {fold}): {len(X_train_s2)} samples")
            
        estimator_s2 = build_estimator((6, 1024), {}, boss_cache_dir, args.show_progress)
        estimator_s2.fit(X_train_s2, y_train_s2, sample_weight=w_train_s2)
        
        # Test Stage 2
        s2_test_df = test_df[test_df["label_stage2"].isin([0, 1])]
        y_true_s2 = s2_test_df["label_stage2"].values.astype(int)
        y_pred_s2 = []
        for p in s2_test_df["path"]:
            x = load_sample(p)
            pred, _ = predict_with_probs(estimator_s2, np.expand_dims(x, axis=0))
            y_pred_s2.append(pred)
            
        s2_metrics = compute_metrics(y_true_s2, y_pred_s2, labels=[0, 1])
        stage2_results.append(s2_metrics)
        print(f"Stage 2 -> Acc: {s2_metrics['accuracy']:.4f} | BAcc: {s2_metrics['balanced_accuracy']:.4f}")
        
        # -----------------------------------------------------
        # END-TO-END EVALUATION
        # true labels for e2e mapping: -1->Healthy, 0->NonPD, 1->PD
        # We will map predictions similarly to strings so we can compare directly: 'Healthy', 'NonPD', 'PD'
        # -----------------------------------------------------
        y_true_e2e_str = []
        for l in test_df["label_stage2"]:
            if l == -1: y_true_e2e_str.append("Healthy")
            elif l == 0: y_true_e2e_str.append("NonPD")
            elif l == 1: y_true_e2e_str.append("PD")
            else: raise ValueError(f"Unknown label_stage2: {l}")
            
        y_pred_e2e_str = []
        for p in test_df["path"]:
            x = np.expand_dims(load_sample(p), axis=0)
            pred1, _ = predict_with_probs(estimator_s1, x)
            if pred1 == 0:
                y_pred_e2e_str.append("Healthy")
            else:
                pred2, _ = predict_with_probs(estimator_s2, x)
                if pred2 == 0:
                    y_pred_e2e_str.append("NonPD")
                else:
                    y_pred_e2e_str.append("PD")
                    
        e2e_labels = ["Healthy", "NonPD", "PD"]
        e2e_metrics = compute_metrics(y_true_e2e_str, y_pred_e2e_str, labels=e2e_labels)
        e2e_results.append(e2e_metrics)
        print(f"E2E     -> Acc: {e2e_metrics['accuracy']:.4f} | BAcc: {e2e_metrics['balanced_accuracy']:.4f}\n")
        
    # Aggegrate and print
    s1_mean, s1_std = aggregate_metrics(stage1_results)
    s2_mean, s2_std = aggregate_metrics(stage2_results)
    e2e_mean, e2e_std = aggregate_metrics(e2e_results)
    
    print("========== FINAL RESULTS (MEAN ± STD) ==========")
    print(f"Stage 1 Acc: {s1_mean['accuracy']:.4f} ± {s1_std['accuracy']:.4f}")
    print(f"Stage 1 BAcc: {s1_mean['balanced_accuracy']:.4f} ± {s1_std['balanced_accuracy']:.4f}")
    print(f"Stage 2 Acc: {s2_mean['accuracy']:.4f} ± {s2_std['accuracy']:.4f}")
    print(f"Stage 2 BAcc: {s2_mean['balanced_accuracy']:.4f} ± {s2_std['balanced_accuracy']:.4f}")
    print(f"E2E Acc: {e2e_mean['accuracy']:.4f} ± {e2e_std['accuracy']:.4f}")
    print(f"E2E BAcc: {e2e_mean['balanced_accuracy']:.4f} ± {e2e_std['balanced_accuracy']:.4f}")
    
    # Dump JSON
    out_dict = {
        "stage1": {
            "per_fold": stage1_results,
            "mean": s1_mean,
            "std": s1_std
        },
        "stage2": {
            "per_fold": stage2_results,
            "mean": s2_mean,
            "std": s2_std
        },
        "end_to_end": {
            "per_fold": e2e_results,
            "mean": e2e_mean,
            "std": e2e_std
        }
    }
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
        
    print(f"Saved complete results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
