import argparse
import threading
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from inference import predict_with_probs
from train_task_specific import load_data


def parse_args():
    p = argparse.ArgumentParser(description="State Machine Simulated Live Inference for Ensembled Tasks")
    p.add_argument("--watch-dir", default="./watch_incoming", help="Directory to monitor for files")
    p.add_argument("--artifact-root", required=True, help="Path containing deploy_cache models")
    p.add_argument("--file-list-csv", default=None, help="Dataset csv to simulate dropping files from")
    p.add_argument("--subject-id", default=None, help="Subject ID to simulate testing on")
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--tasks", nargs="+", default=["RelaxedTask", "StretchHold", "DrinkGlas"])
    return p.parse_args()


TASKS = ["RelaxedTask", "StretchHold", "DrinkGlas"]


def simulate_dropping(watch_dir: Path, task_list: list, df: pd.DataFrame, subject_id: str):
    """Wait and drop files one by one to simulate clinical phases."""
    subj_df = df[df["subject_id"] == subject_id]
    if len(subj_df) == 0:
        print(f"[Simulator] Could not find subject {subject_id} in dataset.")
        return
    
    for task in task_list:
        print(f"\n[Simulator] Waiting 5 seconds before dropping {task} file...")
        time.sleep(5)
        task_df = subj_df[subj_df["task"] == task]
        if len(task_df) > 0:
            target_path = Path(task_df.iloc[0]["path"])
            dest_path = watch_dir / f"esp32_{task}.npy"
            dest_path.write_bytes(target_path.read_bytes())
            print(f"[Simulator] Dropped {dest_path.name}")
        else:
            print(f"[Simulator] Subject {subject_id} has no {task} file in dataset! Dropping empty fallback.")
            dest_path = watch_dir / f"esp32_{task}.npy"
            dest_path.write_bytes(np.zeros((1024, 6), dtype=np.float32).tobytes())


def load_sample(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = arr[:, 1:]
    arr = arr.T
    if arr.shape[1] >= 1024:
        arr = arr[:, :1024]
    else:
        pad = np.zeros((6, 1024 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr


def main():
    args = parse_args()
    watch_dir = Path(args.watch_dir).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean watch dir before starting
    for f in watch_dir.glob("*.npy"):
        f.unlink()

    # Load 6 Models
    models_stage1 = {}
    models_stage2 = {}
    cache_dir = Path(args.artifact_root).resolve() / "deploy_cache"
    
    for task in args.tasks:
        m1_path = cache_dir / f"{task}_stage1_model.pkl"
        m2_path = cache_dir / f"{task}_stage2_model.pkl"
        
        if m1_path.exists():
            with open(m1_path, "rb") as f:
                models_stage1[task] = pickle.load(f)
        if m2_path.exists():
            with open(m2_path, "rb") as f:
                models_stage2[task] = pickle.load(f)
                
    if not models_stage1:
        print("[Error] No trained models found in deploy_cache. Train them first!")
        return

    # Start simulation thread if required
    if args.file_list_csv and args.subject_id:
        df = pd.read_csv(args.file_list_csv)
        t = threading.Thread(target=simulate_dropping, args=(watch_dir, args.tasks, df, args.subject_id))
        t.daemon = True
        t.start()

    print("\n--- CLINICAL EVALUATION PROTOCOL STARTED ---")
    
    collected_probs_s1 = []
    collected_probs_s2 = []
    
    for step_num, task in enumerate(args.tasks, 1):
        print(f"\n[PHASE {step_num}] Awaiting Task: {task}")
        print("Please perform the required movement for 10 seconds.")
        
        expected_file = f"esp32_{task}.npy"
        file_path = watch_dir / expected_file
        
        # Wait until file drops
        while not file_path.exists():
            time.sleep(args.poll_interval)
            
        print(f"-> Detected sensor data for {task}.")
        
        # Infer on the dropped file for this task
        if task in models_stage1:
            x = np.expand_dims(load_sample(file_path), axis=0)
            
            # Predict Stage 1
            pred1, prob1 = predict_with_probs(models_stage1[task], x)
            if prob1 is not None:
                collected_probs_s1.append(prob1)
                
            print(f"   [Stage 1] Logits/Probs for {task}: {prob1}")
            
            # Predict Stage 2 (always predict just to store probs)
            if task in models_stage2:
                pred2, prob2 = predict_with_probs(models_stage2[task], x)
                if prob2 is not None:
                    collected_probs_s2.append(prob2)
                print(f"   [Stage 2] Logits/Probs for {task}: {prob2}")
        else:
            print(f"   [Error] Model for {task} missing. Skipping task inference.")

    print("\n========== FINAL ENSEMBLE CLINICAL VERDICT ==========")
    if collected_probs_s1:
        avg_s1 = np.mean(collected_probs_s1, axis=0)
        class_1_pred = int(np.argmax(avg_s1))
        print(f"Averaged Stage 1 Confidence Vector: {avg_s1}")
        
        if class_1_pred == 0:
            print("=> Final Diagnosis: HEALTHY (Stage 1 Class 0)")
        else:
            print("=> Stage 1 predicts: DISEASED.")
            if collected_probs_s2:
                avg_s2 = np.mean(collected_probs_s2, axis=0)
                class_2_pred = int(np.argmax(avg_s2))
                print(f"Averaged Stage 2 Confidence Vector: {avg_s2}")
                if class_2_pred == 0:
                    print("=> Final Diagnosis: NON-PARKINSONS DISEASE (Stage 2 Class 0)")
                else:
                    print("=> Final Diagnosis: PARKINSONS DISEASE (Stage 2 Class 1)")
            else:
                print("=> Final Diagnosis: PARKINSONS (Fallback due to missing Sub-Models)")
    else:
        print("=> Failed: No data collected.")

if __name__ == "__main__":
    main()
