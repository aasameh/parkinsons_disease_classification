import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from modeling import build_estimator, get_balanced_weights


def parse_args():
    p = argparse.ArgumentParser(description="Train task-specific MultiBOSS SVM models")
    p.add_argument("--file-list-csv", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument(
        "--tasks", 
        nargs="+", 
        default=["RelaxedTask", "StretchHold", "DrinkGlas"],
        help="List of task strings to train individual models for."
    )
    p.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    p.add_argument("--show-progress", action="store_true")
    return p.parse_args()


def load_data(df, stage):
    X, y, c = [], [], []
    label_col = f"label_{stage}"
    
    for _, row in df.iterrows():
        label = row[label_col]
        if label not in [0, 1]:
            continue
            
        arr = np.load(row["path"]).astype(np.float32)
        arr = arr[:, 1:]
        arr = arr.T
        
        if arr.shape[1] >= 1024:
            arr = arr[:, :1024]
        else:
            pad = np.zeros((6, 1024 - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
            
        X.append(arr)
        y.append(label)
        c.append(row["gender"])
        
    return np.array(X), np.array(y), np.array(c)


def get_task_window_sizes(task: str):
    # Tremor is fast, bradykinesia is slow
    if task == "RelaxedTask":
        return (20, 40)
    elif task == "DrinkGlas":
        return (80, 160)
    else:
        return (40, 80)


def train_and_save_model(df, task, stage, artifact_root, show_progress):
    print(f"\n--- Training {task} for {stage} ---")
    task_df = df[df["task"] == task].copy()
    if len(task_df) == 0:
        print(f"Skipping {task}: no data found.")
        return
        
    X, y, c = load_data(task_df, stage)
    if len(X) == 0:
        print(f"No valid labeled data found for {task} ({stage})")
        return
        
    weights = get_balanced_weights(y, c)
    boss_cache_dir = str(Path(artifact_root) / "out" / "boss_cache" / task)
    
    window_sizes = get_task_window_sizes(task)
    print(f"[{task}] Using MultiBOSS window_sizes={window_sizes}, N={len(X)}")
    
    estimator = build_estimator(
        data_shape=(6, 1024),
        svm_params={},
        boss_cache_dir=boss_cache_dir,
        show_progress=show_progress
    )
    # Override the default (20, 40, 80) with task-specific sizes
    estimator.set_params(boss__window_sizes=window_sizes, clf__probability=True)
    
    estimator.fit(X, y, sample_weight=weights)
    
    cache_dir = Path(artifact_root) / "deploy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / f"{task}_{stage}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(estimator, f)
        
    print(f"Saved {task} model to {model_path}")


def main():
    args = parse_args()
    df = pd.read_csv(args.file_list_csv)
    
    for task in args.tasks:
        if args.stage in ["stage1", "both"]:
            train_and_save_model(df, task, "stage1", args.artifact_root, args.show_progress)
        if args.stage in ["stage2", "both"]:
            train_and_save_model(df, task, "stage2", args.artifact_root, args.show_progress)


if __name__ == "__main__":
    main()
