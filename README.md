# Coin Detection CV

Computer-vision pipeline for coin detection, counting, and rule-based USD denomination labeling.

## Pipeline

1. Preprocessing (`src/preprocessing.py`)

- Load image with OpenCV
- Convert to grayscale
- Apply median + Gaussian blur
- Save processed grayscale image

2. Edge Detection (`src/edge_detection.py`)

- Manual Sobel gradients
- Gradient magnitude/direction
- Non-maximum suppression
- Double threshold + hysteresis
- Save binary edge map

3. Circle Detection (`src/circle_detection.py`)

- Circle proposals and filtering
- Optional overlap refinement via watershed (`src/watershed.py`)
- Save annotated detections

4. Feature Extraction (`src/feature_extraction.py`)

- Per-coin ROI and circular mask
- HSV mean features
- Radius normalization
- HSV histograms

5. Final Stage (`src/classification.py` + `main.py`)

- Coin counting
- Rule-based denomination prediction (USD labels): `QUARTER`, `DIME`, `PENNY`, `UNKNOWN`
- Count evaluation vs ground truth (precision/recall)
- Batch final aggregate score across all images

## Project Structure

- `main.py`: orchestrates demo and batch execution
- `src/`: core pipeline modules
- `scripts/build_renamed_count_gt.py`: build single renamed ground-truth CSV
- `data/raw/`: input images
- `data/processed/`, `data/edges/`, `data/circles/`: stage outputs
- `outputs/`: debug visualizations

## Setup

Using `uv`:

```bash
uv sync
```

Or pip:

```bash
pip install -r requirement.txt
```

## Ground Truth (Single CSV)

If your raw GT uses original hashed names, build renamed GT once:

```bash
python3 scripts/build_renamed_count_gt.py
```

Default output:

- `data/coins_count_values_renamed.csv`

## Run

Demo (single image):

```bash
python3 main.py data/raw/coin_001.jpg
```

Demo with count evaluation:

```bash
python3 main.py data/raw/coin_001.jpg --ground-truth data/coins_count_values_renamed.csv
```

Batch:

```bash
python3 main.py --batch
```

Batch with evaluation + final score:

```bash
python3 main.py --batch --ground-truth data/coins_count_values_renamed.csv
```

## CLI Options

```bash
python3 main.py --help
```

Key options:

- `--batch`
- `--ground-truth <csv>`
- `--min-r <int>`
- `--max-r <int>`
- `--vote-thresh <float>`
- `--low-ratio <float>`
- `--high-ratio <float>`
- `--no-show`

## Notes

- Current denomination labeling is rule-based only (no template matching).
- Thresholds for denomination rules live in `DenominationClassifier.RULES` inside `src/classification.py`.
- Count scoring is based on coin-count overlap (TP/FP/FN), with global batch metrics printed at the end.
