import argparse
import threading
import time
import pickle
from pathlib import Path

import numpy as np


from inference import predict_with_probs


def parse_args():
    p = argparse.ArgumentParser(description="Live directory-watching inference engine")
    p.add_argument("--watch-dir",      help="Directory to monitor for new .npy files", default="./watch_incoming")
    p.add_argument("--mode",           choices=["pd_vs_hc", "pd_vs_dd"], default="pd_vs_hc")
    p.add_argument("--artifact-root", help="Path to artifact root containing .pkl multiboss", default="C:\\college\\6th\\PBLx\\artifacts")
    p.add_argument("--data-root",      default=None,  help="Only required on first run when no cache exists. Ignored if cache hit.")
    p.add_argument("--poll-interval",  type=float, default=2.0)
    p.add_argument("--file-age",       type=float, default=3.0)
    p.add_argument("--sim-data-dir",   default=None,  help="If set, replay .npy files from this dir into watch-dir (testing only)")
    p.add_argument("--sim-interval",   type=float, default=30.0, help="Interval between simulated file drops")
    return p.parse_args()

def simulate_incoming(src_dir: Path, watch_dir: Path, interval: float = 5.0):
    """Copies .npy files one-by-one into watch_dir, simulating ESP32 output, one per patient."""
    files = sorted(Path(src_dir).glob("*.npy"))
    if not files:
        print(f"[Sim] No .npy files found in {src_dir}")
        return
        
    seen_patients = set()
    selected_files = []
    
    # Filter files to grab only the first one encountered for each unique patient ID
    for f in files:
        parts = f.stem.split("_")
        if len(parts) > 0:
            subject_id = parts[0]
            if subject_id not in seen_patients:
                seen_patients.add(subject_id)
                selected_files.append(f)
                
    print(f"[Sim] Found {len(selected_files)} unique patients. Starting simulation...")
    
    for f in selected_files:
        dest = Path(watch_dir) / f.name
        dest.write_bytes(f.read_bytes())
        print(f"[Sim] Dropped {f.name} into watch dir")
        time.sleep(interval)
        
    print("[Sim] All files replayed.")


def is_ready(file_path: Path, delay: float) -> bool:
    return (time.time() - file_path.stat().st_mtime) > delay


def main():
    args = parse_args()

    watch_dir = Path(args.watch_dir).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = Path(args.artifact_root).resolve()

    cache_dir = artifact_root / "deploy_cache"
    
    stage1_path = cache_dir / "stage1_model.pkl"
    print(f"[Inference] Loading stage1 model from {stage1_path} ...")
    with open(stage1_path, "rb") as f:
        stage1 = pickle.load(f)
        
    stage2_path = cache_dir / "stage2_model.pkl"
    print(f"[Inference] Loading stage2 model from {stage2_path} ...")
    with open(stage2_path, "rb") as f:
        stage2 = pickle.load(f)

    # Start simulation thread if testing without ESP32
    if args.sim_data_dir:
        t_sim = threading.Thread(
            target=simulate_incoming,
            args=(args.sim_data_dir, watch_dir, args.sim_interval),
            daemon=True,
        )
        t_sim.start()
        print(f"[Sim] Replaying files from {args.sim_data_dir}")

    print(f"\n[Inference] Model ready. Watching: {watch_dir}")
    print(f"[Inference] Polling every {args.poll_interval}s, min file age {args.file_age}s")

    processed = set()

    while True:
        try:
            for f in sorted(watch_dir.glob("*.npy")):
                if f not in processed and is_ready(f, args.file_age):
                    print(f"\n[Inference] Detected: {f.name}")
                    try:
                        arr = np.load(f).astype(np.float32)
                        arr = arr[:, 1:]
                        arr = arr.T
                        if arr.shape[1] >= 1024:
                            arr = arr[:, :1024]
                        else:
                            pad = np.zeros((6, 1024 - arr.shape[1]), dtype=np.float32)
                            arr = np.concatenate([arr, pad], axis=1)
                            
                        x = np.expand_dims(arr, axis=0)           # shape (1, 6, 1024)

                        pred1, probs1 = predict_with_probs(stage1, x)

                        if pred1 == 0:
                            print(f"Result: Healthy | Probs: {probs1}")
                        else:
                            pred2, probs2 = predict_with_probs(stage2, x)
                            label = "Parkinson's" if pred2 == 1 else "Non-PD Disease"
                            print(f"Result: {label} | Stage1 Probs: {probs1} | Stage2 Probs: {probs2}")
                            
                    except Exception as e:
                        print(f"❌ Error processing {f.name}: {e}")
                    processed.add(f)

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            print("\n[Inference] Stopped by user.")
            break


if __name__ == "__main__":
    main()