import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from modeling import build_estimator, get_balanced_weights
from inference import predict_with_probs


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Late-Fusion Task Ensemble SVM classifier")
    p.add_argument("--file-list-csv", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--tasks", nargs="+", default=["RelaxedTask", "StretchHold", "DrinkGlas"])
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


def get_task_window_sizes(task: str):
    if task == "RelaxedTask":
        return (20, 40)
    elif task == "DrinkGlas":
        return (80, 160)
    return (40, 80)


def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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
            if isinstance(mean_res[k], list) and not isinstance(fold_results[0][k], list):
                mean_res[k] = mean_res[k]
                std_res[k] = std_res[k]
    return mean_res, std_res


def main():
    args = parse_args()
    
    artifact_root = Path(args.artifact_root).resolve()
    df = pd.read_csv(args.file_list_csv)
    
    subject_df = df.drop_duplicates(subset=["subject_id"]).copy().reset_index(drop=True)
    subject_ids = subject_df["subject_id"].values
    subject_labels = subject_df["label_stage2"].values  # -1 (Healthy), 0 (NonPD), 1 (PD)
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    stage1_results, stage2_results, e2e_results = [], [], []
    
    for fold, (train_subj_idx, test_subj_idx) in enumerate(skf.split(subject_ids, subject_labels), 1):
        print(f"--- Fold {fold}/{args.n_folds} ---")
        
        train_subjects = set(subject_ids[train_subj_idx])
        test_subjects = set(subject_ids[test_subj_idx])
        
        train_df = df[df["subject_id"].isin(train_subjects)]
        test_df = df[df["subject_id"].isin(test_subjects)]
        
        # 1. Train Task Models for this fold
        models_stage1 = {}
        models_stage2 = {}
        for task in args.tasks:
            task_train_df = train_df[train_df["task"] == task]
            
            # Stage 1
            s1_df = task_train_df[task_train_df["label_stage1"].isin([0, 1])]
            if len(s1_df) > 0:
                X_s1 = np.stack([load_sample(p) for p in s1_df["path"]])
                y_s1 = s1_df["label_stage1"].values.astype(int)
                w_s1 = get_balanced_weights(y_s1, s1_df["gender"].values)
                
                est_s1 = build_estimator((6, 1024), {}, str(artifact_root / "out" / "boss_cache" / task), args.show_progress)
                est_s1.set_params(boss__window_sizes=get_task_window_sizes(task), clf__probability=True)
                est_s1.fit(X_s1, y_s1, sample_weight=w_s1)
                models_stage1[task] = est_s1
                
            # Stage 2
            s2_df = task_train_df[task_train_df["label_stage2"].isin([0, 1])]
            if len(s2_df) > 0:
                X_s2 = np.stack([load_sample(p) for p in s2_df["path"]])
                y_s2 = s2_df["label_stage2"].values.astype(int)
                w_s2 = get_balanced_weights(y_s2, s2_df["gender"].values)
                
                est_s2 = build_estimator((6, 1024), {}, str(artifact_root / "out" / "boss_cache" / task), args.show_progress)
                est_s2.set_params(boss__window_sizes=get_task_window_sizes(task), clf__probability=True)
                est_s2.fit(X_s2, y_s2, sample_weight=w_s2)
                models_stage2[task] = est_s2

        # 2. Evaluate stage 1 using late fusion across available tasks
        test_s1_subjs = test_df[test_df["label_stage1"].isin([0, 1])]["subject_id"].unique()
        y_true_s1, y_pred_s1 = [], []
        
        for subj in test_s1_subjs:
            subj_df = test_df[test_df["subject_id"] == subj]
            true_label = int(subj_df.iloc[0]["label_stage1"])
            
            task_prob_vectors = []
            for task in models_stage1:
                task_samples = subj_df[subj_df["task"] == task]["path"]
                task_window_probs = []
                for p in task_samples:
                    x = np.expand_dims(load_sample(p), axis=0)
                    _, probs = predict_with_probs(models_stage1[task], x)
                    if probs is not None:
                        task_window_probs.append(probs)
                
                if task_window_probs:
                    p_k = np.mean(task_window_probs, axis=0)
                    task_prob_vectors.append(p_k)
            
            if task_prob_vectors:
                avg_prob = np.mean(task_prob_vectors, axis=0)
                pred = int(np.argmax(avg_prob))
                y_true_s1.append(true_label)
                y_pred_s1.append(pred)
                
        metrics_s1 = compute_metrics(y_true_s1, y_pred_s1, [0, 1])
        stage1_results.append(metrics_s1)
        print(f"Stage 1 Acc: {metrics_s1['accuracy']:.4f} | BAcc: {metrics_s1['balanced_accuracy']:.4f}")
        
        # 3. Evaluate stage 2 using late fusion
        test_s2_subjs = test_df[test_df["label_stage2"].isin([0, 1])]["subject_id"].unique()
        y_true_s2, y_pred_s2 = [], []
        
        for subj in test_s2_subjs:
            subj_df = test_df[test_df["subject_id"] == subj]
            true_label = int(subj_df.iloc[0]["label_stage2"])
            
            task_prob_vectors = []
            for task in models_stage2:
                task_samples = subj_df[subj_df["task"] == task]["path"]
                task_window_probs = []
                for p in task_samples:
                    x = np.expand_dims(load_sample(p), axis=0)
                    _, probs = predict_with_probs(models_stage2[task], x)
                    if probs is not None:
                        task_window_probs.append(probs)
                
                if task_window_probs:
                    p_k = np.mean(task_window_probs, axis=0)
                    task_prob_vectors.append(p_k)
            
            if task_prob_vectors:
                avg_prob = np.mean(task_prob_vectors, axis=0)
                pred = int(np.argmax(avg_prob))
                y_true_s2.append(true_label)
                y_pred_s2.append(pred)
                
        metrics_s2 = compute_metrics(y_true_s2, y_pred_s2, [0, 1])
        stage2_results.append(metrics_s2)
        print(f"Stage 2 Acc: {metrics_s2['accuracy']:.4f} | BAcc: {metrics_s2['balanced_accuracy']:.4f}")

        # 4. Evaluate E2E using late fusion hierarchical
        y_true_e2e, y_pred_e2e = [], []
        for subj in test_df["subject_id"].unique():
            subj_df = test_df[test_df["subject_id"] == subj]
            l2 = subj_df.iloc[0]["label_stage2"]
            true_label = "Healthy" if l2 == -1 else ("NonPD" if l2 == 0 else "PD")
            
            # Predict Stage 1
            s1_task_probs = []
            valid_x_list = []
            for task in models_stage1:
                task_samples = subj_df[subj_df["task"] == task]["path"]
                task_window_probs = []
                for p in task_samples:
                    x = np.expand_dims(load_sample(p), axis=0)
                    valid_x_list.append((task, x))
                    _, probs = predict_with_probs(models_stage1[task], x)
                    if probs is not None:
                        task_window_probs.append(probs)
                
                if task_window_probs:
                    p_k = np.mean(task_window_probs, axis=0)
                    s1_task_probs.append(p_k)
            
            if not s1_task_probs:
                continue
                
            pred1 = int(np.argmax(np.mean(s1_task_probs, axis=0)))
            if pred1 == 0:
                y_pred_e2e.append("Healthy")
            else:
                # Stage 2
                s2_task_probs = []
                for task in models_stage2:
                    task_window_probs = []
                    for t, x in valid_x_list:
                        if t == task:
                            _, probs = predict_with_probs(models_stage2[task], x)
                            if probs is not None:
                                task_window_probs.append(probs)
                            
                    if task_window_probs:
                        p_k = np.mean(task_window_probs, axis=0)
                        s2_task_probs.append(p_k)
                
                if s2_task_probs:
                    pred2 = int(np.argmax(np.mean(s2_task_probs, axis=0)))
                    y_pred_e2e.append("NonPD" if pred2 == 0 else "PD")
                else:
                    y_pred_e2e.append("PD") # Fallback to more severe class if missing stage 2 data
            
            y_true_e2e.append(true_label)
            
        metrics_e2e = compute_metrics(y_true_e2e, y_pred_e2e, ["Healthy", "NonPD", "PD"])
        e2e_results.append(metrics_e2e)
        print(f"E2E Acc: {metrics_e2e['accuracy']:.4f} | BAcc: {metrics_e2e['balanced_accuracy']:.4f}")

    # Aggregate
    s1_mean, s1_std = aggregate_metrics(stage1_results)
    s2_mean, s2_std = aggregate_metrics(stage2_results)
    e2e_mean, e2e_std = aggregate_metrics(e2e_results)
    
    print("\n========== FINAL ENSEMBLE RESULTS ==========")
    print(f"Stage 1 BAcc: {s1_mean['balanced_accuracy']:.4f} ± {s1_std['balanced_accuracy']:.4f}")
    print(f"Stage 2 BAcc: {s2_mean['balanced_accuracy']:.4f} ± {s2_std['balanced_accuracy']:.4f}")
    print(f"E2E BAcc: {e2e_mean['balanced_accuracy']:.4f} ± {e2e_std['balanced_accuracy']:.4f}")
    
    out_dict = {
        "stage1": {"per_fold": stage1_results, "mean": s1_mean, "std": s1_std},
        "stage2": {"per_fold": stage2_results, "mean": s2_mean, "std": s2_std},
        "end_to_end": {"per_fold": e2e_results, "mean": e2e_mean, "std": e2e_std}
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    main()
