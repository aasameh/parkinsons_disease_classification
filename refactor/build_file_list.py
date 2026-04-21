import argparse
import json
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npy-dir", required=True)
    p.add_argument("--patients-dir", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    
    patients_dir = Path(args.patients_dir)
    npy_dir = Path(args.npy_dir)
    
    patients = {}
    for json_file in patients_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            subject_id = data.get("id")
            condition = data.get("condition")
            gender = data.get("gender")
            
            label_stage1 = 0 if condition == "Healthy" else 1
            if condition == "Healthy":
                label_stage2 = -1
            elif condition == "Parkinson's":
                label_stage2 = 1
            else:
                label_stage2 = 0
                
            patients[subject_id] = {
                "gender": gender,
                "label_stage1": label_stage1,
                "label_stage2": label_stage2
            }
            
    rows = []
    for npy_file in npy_dir.glob("*.npy"):
        parts = npy_file.stem.split("_")
        if len(parts) >= 3:
            subject_id = parts[0]
            task = parts[1]
            wrist = parts[2]
            
            if subject_id in patients:
                p_data = patients[subject_id]
                rows.append({
                    "path": str(npy_file.resolve()),
                    "subject_id": subject_id,
                    "task": task,
                    "wrist": wrist,
                    "gender": p_data["gender"],
                    "label_stage1": p_data["label_stage1"],
                    "label_stage2": p_data["label_stage2"]
                })
                
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    
    print(f"Total files: {len(df)}")
    print("\nStage 1 label counts (0=Healthy, 1=Diseased):")
    print(df["label_stage1"].value_counts())
    print("\nStage 2 label counts (-1=Excluded, 0=NonPD, 1=PD):")
    print(df["label_stage2"].value_counts())

if __name__ == "__main__":
    main()
