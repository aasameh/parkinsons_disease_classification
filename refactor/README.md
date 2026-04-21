notes
inference.py is a script that runs for single inference on one file
MultiBOSS acts as a transformer and feeds data into SVM
Q1: What is the shape of the transformed data? what does the SVM actually classify? how long is the transformation and does it bottleneck inference time? optimize the transform pipeline so it doesn't cause inference time issues.

Q2: How does the frequency diagnostic look like? Before and after filtration. Preprocessing and filteration must be designed AROUND the frequency spectrum not the other way around.

Q3: PRIORITY Write live inference data with file loader to load data in specified as NPY arrays and run inference

## Files
- `inference.py` - movement inference using SVM + MultiBOSS + Both-channel setup
- `modeling.py` - minimal channel schema, sample weighting, estimator builder
- `multi_boss.py` - local MultiBOSS transformer
- `preprocessing.py` - shared loading/windowing/raw-observation utilities used by all scripts
- `spectral.py` - basic PSD and 50/60 Hz diagnostics
- `analyze_raw_frequency.py` - raw pre-filter signal and spectrum plots
- `convert_txt_to_npy.py` - recursively convert numeric `.txt` files to `.npy`
- `preprocess_single_npy.py` - one-file preprocessing for converted `.npy` arrays

## Inference (single preprocessed bin sample)
Run from repo root (`pads-project`):

If `--window-minutes` is omitted, inference auto-detects windowing by using the full incoming 2D sample as one window.

```bash
python refactor/inference.py \
  --mode pd_vs_hc \
  --sample data/preprocessed/movement/001_ml.bin \
  --artifact-root data \
  --data-root data/preprocessed
```

## Inference with fold aggregation

```bash
python refactor/inference.py \
  --mode pd_vs_hc \
  --sample data/preprocessed/movement/001_ml.bin \
  --artifact-root data \
  --data-root data/preprocessed \
  --window-minutes 0.1626667 \
  --aggregate-folds
```

## Raw pre-filter frequency analysis

```bash
python refactor/analyze_raw_frequency.py \
  --observation-json pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement/observation_001.json \
  --sampling-rate 100 \
  --out-dir data/out/frequency_analysis
```

Outputs:
- `*_raw_signal.png`
- `*_psd.png`
- `*_summary.json`

Analyze a converted `.npy` directly:

```bash
python refactor/analyze_raw_frequency.py \
  --npy pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement/timeseries/001_Relaxed_LeftWrist.npy \
  --channel-index 1 \
  --sampling-rate 100 \
  --out-dir data/out/frequency_analysis
```

## Single preprocessing on one converted npy

```bash
python refactor/preprocess_single_npy.py \
  --input-npy pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement/timeseries/001_Relaxed_LeftWrist.npy \
  --output-npy data/out/single_preprocessed/001_Relaxed_LeftWrist_preprocessed.npy
```

## Convert all txt files to npy

Convert in place under a root directory:

```bash
python refactor/convert_txt_to_npy.py --input-root pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement
```

Convert into a separate output tree (same relative paths):

```bash
python refactor/convert_txt_to_npy.py \
  --input-root pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement \
  --out-root data/raw_npy
```
