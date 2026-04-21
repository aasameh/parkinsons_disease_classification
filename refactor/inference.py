import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import build_estimator, get_balanced_weights, get_channels
from preprocessing import load_sample, split_segments, to_windows

MODES = ("pd_vs_hc", "pd_vs_dd")
MODE_LABELS = {
    "pd_vs_hc": {0: "HC", 1: "PD"},
    "pd_vs_dd": {0: "PD", 1: "DD"},
}
DEFAULT_EXP_NAME_BY_MODE = {
    "pd_vs_hc": "pd_vs_hc_mov_b_both_svm",
    "pd_vs_dd": "pd_vs_dd_mov_b_both_svm",
}


def read_fold_rows(artifact_root: Path):
    table = pd.read_csv(artifact_root / "out" / "mov_res_folds.csv", sep="\t")
    with open(artifact_root / "out" / "best_params_mov.json", "rb") as f:
        params_list = json.load(f)

    rows = []
    i = 0
    for mode in ["pd_vs_dd", "pd_vs_hc"]:
        for fold in range(1, 6):
            r = table[(table["mode"] == mode) & (table["test_fold"] == fold)].iloc[0]
            rows.append(
                {
                    "mode": mode,
                    "test_fold": int(fold),
                    "exp_name": str(r["exp_name"]),
                    "classifier": str(r["classifier"]),
                    "params": params_list[i],
                    "test_accuracy": float(r["test_accuracy"]),
                    "test_balanced_accuracy": float(r["test_balanced_accuracy"]),
                }
            )
            i += 1
    return rows


def select_rows(mode: str, artifact_root: Path, exp_name: str | None, fold: int | None, aggregate_folds: bool):
    exp_name = exp_name or DEFAULT_EXP_NAME_BY_MODE[mode]
    rows = [
        r
        for r in read_fold_rows(artifact_root)
        if r["mode"] == mode and r["exp_name"] == exp_name and r["classifier"].lower() == "svm"
    ]

    if not rows:
        raise ValueError(f"No SVM rows found for mode={mode}, exp_name={exp_name}")

    if fold is not None:
        rows = [r for r in rows if r["test_fold"] == fold]
        if not rows:
            raise ValueError(f"Fold {fold} not found for mode={mode}, exp_name={exp_name}")
        return rows

    if aggregate_folds:
        return sorted(rows, key=lambda x: x["test_fold"])

    return [max(rows, key=lambda x: (x["test_balanced_accuracy"], x["test_accuracy"]))]


def resolve_data_root(artifact_root: Path, data_root: str | None):
    if data_root is not None:
        return Path(data_root).resolve()
    if (artifact_root / "file_list.csv").is_file() and (artifact_root / "movement").is_dir():
        return artifact_root
    if (artifact_root / "preprocessed" / "file_list.csv").is_file() and (artifact_root / "preprocessed" / "movement").is_dir():
        return artifact_root / "preprocessed"
    raise FileNotFoundError("Could not resolve data root. Provide --data-root.")


def get_cache_path(cache_dir: Path, mode: str, exp_name: str, test_fold: int, params):
    payload = json.dumps(
        {"mode": mode, "exp_name": exp_name, "test_fold": test_fold, "params": params},
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return cache_dir / f"mov_{mode}_{exp_name}_f{test_fold}_{digest}.pkl"


def get_file_list(data_root: Path, mode: str):
    fl = pd.read_csv(data_root / "file_list.csv")
    if mode == "pd_vs_hc":
        fl = fl[(fl["label"] == 0) | (fl["label"] == 1)].copy()
    else:
        fl = fl[(fl["label"] == 1) | (fl["label"] == 2)].copy()
        fl["label"] = fl["label"].replace({1: 0, 2: 1})
    return fl


def load_train_data(data_root: Path, mode: str, show_progress: bool = False):
    fl = get_file_list(data_root, mode)
    y = fl["label"].to_numpy(dtype=np.int64)
    c = fl["gender"].to_numpy()

    x = []
    n = len(fl)
    for sid in fl["id"]:
        arr = np.fromfile(data_root / "movement" / f"{int(sid):03d}_ml.bin", dtype=np.float32).reshape((-1, 976))
        x.append(arr)
        if show_progress and (len(x) % 10 == 0 or len(x) == n):
            print(f"[TrainData] Loaded {len(x)}/{n} samples", flush=True)
    return np.stack(x), y, c


def fit_model(params, mode: str, artifact_root: Path, data_root: Path, show_progress: bool = False):
    if show_progress:
        print("[Fit] Loading train data...", flush=True)
    x, y, c = load_train_data(data_root=data_root, mode=mode, show_progress=show_progress)
    if show_progress:
        print(f"[Fit] Train tensor shape: {x.shape}", flush=True)
        print("[Fit] Building estimator...", flush=True)
    estimator = build_estimator(
        data_shape=x[0].shape,
        svm_params=params,
        boss_cache_dir=str(artifact_root / "out" / "boss_cache"),
        show_progress=show_progress,
    )
    weights = get_balanced_weights(y, c)
    if show_progress:
        print("[Fit] Estimator.fit starting...", flush=True)
    estimator.fit(x, y, sample_weight=weights)
    if show_progress:
        print("[Fit] Estimator.fit done.", flush=True)
    return estimator


def predict_with_probs(estimator, x_single: np.ndarray):
    pred = int(estimator.predict(x_single)[0])

    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(x_single)[0]
        return pred, [float(v) for v in probs]

    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(x_single))
        if score.ndim == 1:
            margin = float(score[0])
            p1 = 1.0 / (1.0 + np.exp(-margin))
            return pred, [float(1.0 - p1), float(p1)]

        logits = score[0].astype(float)
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / np.sum(exps)
        return pred, [float(v) for v in probs]

    return pred, None


def predict_window(estimator, window: np.ndarray, segment_samples: int, segment_stride: int):
    expected_channels = len(get_channels("movement"))

    # Defensive fix: some inputs arrive as (samples, channels).
    if window.ndim == 2 and window.shape[1] == expected_channels and window.shape[0] != expected_channels:
        window = window.T

    if window.shape[0] != expected_channels:
        raise ValueError(
            "Expected window layout (channels, samples). "
            f"Expected {expected_channels} channels on axis 0, got shape {window.shape}."
        )

    segs = split_segments(window, segment_samples, segment_stride)
    seg_preds = []
    seg_probs = []

    for seg in segs:
        pred, probs = predict_with_probs(estimator, np.expand_dims(seg, axis=0))
        seg_preds.append(pred)
        if probs is not None:
            seg_probs.append(probs)

    if seg_probs:
        probs = np.mean(np.asarray(seg_probs, dtype=float), axis=0)
        pred = int(np.argmax(probs))
        return pred, [float(v) for v in probs], len(segs)

    vals, cnts = np.unique(np.asarray(seg_preds), return_counts=True)
    pred = int(vals[np.argmax(cnts)])
    return pred, None, len(segs)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_artifact_root = str((repo_root / "data").resolve())

    p = argparse.ArgumentParser(description="Minimal SVM+MultiBOSS movement inference")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--sample", required=True, help="Input .bin or .npy")
    p.add_argument("--artifact-root", default=default_artifact_root)
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--fold", type=int, default=None, help="Explicit fold 1..5")
    p.add_argument("--aggregate-folds", action="store_true")
    p.add_argument(
        "--window-minutes",
        type=float,
        default=None,
        help="Fixed window size in minutes. If omitted, auto-detect mode uses full sample as one window.",
    )
    p.add_argument("--sampling-rate", type=int, default=100)
    p.add_argument("--segment-samples", type=int, default=976)
    p.add_argument("--segment-stride", type=int, default=976)
    p.add_argument("--bin-channels", type=int, default=len(get_channels("movement")))
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--show-training-progress", action="store_true")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    data_root = resolve_data_root(artifact_root, args.data_root)
    selected_rows = select_rows(
        mode=args.mode,
        artifact_root=artifact_root,
        exp_name=args.exp_name,
        fold=args.fold,
        aggregate_folds=bool(args.aggregate_folds),
    )

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else artifact_root / "out" / "deploy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for row in selected_rows:
        cache_path = get_cache_path(cache_dir, args.mode, row["exp_name"], row["test_fold"], row["params"])
        if (not args.no_cache) and cache_path.is_file():
            if args.show_training_progress:
                print(f"[Model] Deploy cache hit for fold {row['test_fold']}: {cache_path}", flush=True)
            with open(cache_path, "rb") as f:
                estimator = pickle.load(f)
        else:
            if args.show_training_progress:
                print(f"[Model] Deploy cache miss for fold {row['test_fold']}. Training model...", flush=True)
            estimator = fit_model(
                row["params"],
                mode=args.mode,
                artifact_root=artifact_root,
                data_root=data_root,
                show_progress=args.show_training_progress,
            )
            if not args.no_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(estimator, f)
                if args.show_training_progress:
                    print(f"[Model] Saved deploy cache: {cache_path}", flush=True)
        models.append((row, estimator))

    sample = load_sample(Path(args.sample).resolve(), bin_channels=args.bin_channels)
    if args.show_training_progress:
        if args.window_minutes is None:
            print("[Windowing] Auto mode: using full sample as one window unless sample is already windowed.", flush=True)
        else:
            print(f"[Windowing] Fixed mode: window_minutes={args.window_minutes}", flush=True)
    windows = to_windows(sample, args.window_minutes, args.sampling_rate)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if windows.shape[0] == 0:
        raise ValueError("No windows to infer")

    window_results = []
    for wi in range(windows.shape[0]):
        per_fold = []
        for row, estimator in models:
            pred, probs, nseg = predict_window(estimator, windows[wi], args.segment_samples, args.segment_stride)
            per_fold.append(
                {
                    "fold": int(row["test_fold"]),
                    "prediction": pred,
                    "probabilities": probs,
                    "n_segments": nseg,
                }
            )

        if len(per_fold) == 1:
            pred = int(per_fold[0]["prediction"])
            probs = per_fold[0]["probabilities"]
            source = "single_fold"
        else:
            probs_list = [r["probabilities"] for r in per_fold if r["probabilities"] is not None]
            if probs_list:
                avg = np.mean(np.asarray(probs_list, dtype=float), axis=0)
                pred = int(np.argmax(avg))
                probs = [float(v) for v in avg]
                source = "fold_probability_average"
            else:
                preds = [int(r["prediction"]) for r in per_fold]
                vals, cnts = np.unique(np.asarray(preds), return_counts=True)
                pred = int(vals[np.argmax(cnts)])
                probs = None
                source = "fold_vote"

        window_results.append(
            {
                "window_index": wi,
                "prediction": pred,
                "prediction_label": MODE_LABELS[args.mode].get(pred, str(pred)),
                "probabilities": probs,
                "source": source,
                "per_fold": per_fold,
            }
        )

    preds = [r["prediction"] for r in window_results]
    vals, cnts = np.unique(np.asarray(preds), return_counts=True)
    overall = int(vals[np.argmax(cnts)])

    output = {
        "mode": args.mode,
        "model": {
            "classifier": "svm",
            "transform": "MultiBOSS",
            "channels": "Both",
            "exp_name": selected_rows[0]["exp_name"],
        },
        "selected_folds": [int(r["test_fold"]) for r, _ in models],
        "aggregate_folds": bool(args.aggregate_folds),
        "sample": str(Path(args.sample).resolve()),
        "n_windows": len(window_results),
        "overall_prediction": overall,
        "overall_prediction_label": MODE_LABELS[args.mode].get(overall, str(overall)),
        "window_results": window_results,
    }

    print(json.dumps(output, indent=2))

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
