# deep-learning-project

Global Sentinel-1 land use / land cover (LULC) semantic segmentation - radar-only input, 7 classes, 10 m pixels,
12 monthly maps per location.

## Repository layout
- `notebooks/` - data collection (01, 04), training and model comparison (02, 03, 06, 06b), analysis and
  calibration (07, 08), one-time test evaluation (09).
- `results/` - metrics, bootstrap comparisons, calibration and test results; `results/m5/m5_results/app_config.json`
  is the frozen configuration used by the app.
- `data/` - sample index and normalisation statistics.
- `app/` - **Sentinel-1 Land Cover Monitor** (Gradio prototype): place search or coordinates + start month ->
  12 monthly maps, % area per class, change in percentage points, reliability notes, model comparison.
  See [app/README.md](app/README.md).

## Final model (held-out test set)
SegFormer-B2, temporal input (months m-1, m, m+1), 3-seed ensemble, temperature scaling (T = 1.733):
mIoU 0.570 (95% CI 0.537-0.605); expected calibration error 0.094 -> 0.020.
