import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from modeling import build_estimator, get_balanced_weights

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file-list-csv", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--stage", choices=["stage1", "stage2", "both"], required=True)
    p.add_argument("--show-progress", action="store_true")
    return p.parse_args()

def load_data(df, stage):
    X = []
    y = []
    c = []
    
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
        
    X = np.array(X)
    y = np.array(y)
    c = np.array(c)
    X = X[:1000]
    y = y[:1000]
    c = c[:1000]
    return X, y, c

def train_and_save_model(df, stage, artifact_root, show_progress):
    X, y, c = load_data(df, stage)
    
    if len(X) == 0:
        print(f"No data found for {stage}")
        return
        
    weights = get_balanced_weights(y, c)
    
    boss_cache_dir = str(Path(artifact_root) / "out" / "boss_cache")
    estimator = build_estimator(
        data_shape=(6, 1024),
        svm_params={},
        boss_cache_dir=boss_cache_dir,
        show_progress=show_progress
    )
    
    estimator.fit(X, y, sample_weight=weights)
    
    cache_dir = Path(artifact_root) / "out" / "deploy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / f"{stage}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(estimator, f)
        
    print(f"Saved {stage} model to {model_path}")

def main():
    args = parse_args()
    
    df = pd.read_csv(args.file_list_csv)
    artifact_root = args.artifact_root
    
    if args.stage in ["stage1", "both"]:
        print("\nTraining stage 1...")
        train_and_save_model(df, "stage1", artifact_root, args.show_progress)
        
    if args.stage in ["stage2", "both"]:
        print("\nTraining stage 2...")
        train_and_save_model(df, "stage2", artifact_root, args.show_progress)

if __name__ == "__main__":
    main()
