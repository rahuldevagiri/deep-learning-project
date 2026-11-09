# Sentinel-1 Land Cover Monitor (Gradio app)

Monthly land use / land cover from **Sentinel-1 radar only**. Enter coordinates (or search a place name) and a
start month: the app fetches 14 monthly Sentinel-1 composites, predicts 12 monthly maps (7 classes, 10 m pixels),
and reports the % area per class per month, month-to-month and first-to-last change in percentage points, and how
reliable each result is.

Part of the course project *Global Sentinel-1 LULC Semantic Segmentation* - see the notebooks in the main repository.

## Features
- **Input**: preset cities (all in test regions never used for training), place-name search (OpenStreetMap
  Nominatim, Photon fallback) that fills the coordinates, coordinates that fill the place name (reverse lookup),
  start year + month, area 2.56 / 5.12 / 7.68 km, model choice (final model or one of 5 baselines).
- **Theme**: light by default; the "Light / Dark theme" button switches instantly (results stay on screen) and
  the browser remembers the choice. `?__theme=dark` or `?__theme=light` in the URL also works.
- **Summary**: reliability notes (test/train region, missing months, low-confidence months, snow/ice area, other
  years) and cards for area, % pixels changed first-to-last, largest net gain and loss, agreement with the
  independent reference.
- **Tab Monthly maps**: the 12 monthly LULC maps (+ WorldCover and mean confidence), a month inspector
  (model / Dynamic World / Sentinel-1 radar side by side) and a 12-month animation.
- **Tab Class shares**: P(c, m) = 100 x A(c, m) / A(valid, m) per month (pixels without radar data excluded),
  as chart and table.
- **Tab Change**: month-to-month and first-to-last change in percentage points as charts and tables, with the
  95% noise level from the test set shaded; first vs last month maps, changed-pixel map, transition matrix.
- **Tab Sanity check**: comparison with **Google Dynamic World** (Sentinel-2 optical, not used in training):
  class shares, agreement per month, confusion matrix. WorldCover 2021 is shown too but is the training-label
  source, so it is not independent.
- **Tab Downloads**: CSV tables, the 12-month PNG figure and the GIF, also as one zip.
- **Tab Model evaluation** (static, from `eval_data/`): test OA / mIoU / mean F1, IoU and F1 per class,
  confusion matrix, model-variant comparison, ensemble vs single seeds, bootstrap tests, training curves of all
  13 runs, calibration, test mIoU by terrain / latitude zone / biome.

## Results of the final model (held-out test set)
Overall accuracy 0.726, mIoU 0.570 (95% CI 0.537-0.605), mean F1 0.717; ECE 0.094 -> 0.020 after temperature
scaling. Details in the Model evaluation tab and `config/app_config.json`.

## Folder layout
```
app.py                 Gradio UI only
lulc_core/             science code (no UI): s1_fetch, inference, stats, reference (Dynamic World),
                       evaluation, geocode, viz, config
config/app_config.json frozen preprocessing, temperature, thresholds (from notebook 09)
config/cell_split.json 5-degree cell -> train/val/test split (for the region badge)
eval_data/             saved result files for the Model evaluation tab (copied from results/ and run histories)
weights/<run>/best.pth model weights (not in git)
cache/                 fetched Sentinel-1 / Dynamic World stacks (.npz)
build_cache.py         pre-fetch demo locations
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
earthengine authenticate          # once, with your own Google account (Earth Engine enabled)
python app.py                     # open http://127.0.0.1:7860
```
Weights: put each run folder in `weights/` (e.g. `weights/segformer_b2_t3_s42/best.pth`), or set
`LULC_WEIGHTS_DIR`. Only models whose weight files exist appear in the model list. The final model needs
`segformer_b2_t3_s42/43/44`.

Environment variables (all optional): `LULC_WEIGHTS_DIR`, `LULC_CACHE_DIR`, `EE_PROJECT` (default
`deep-learning-27`), `EE_SERVICE_ACCOUNT_JSON` (hosted use, see below).

## Hosting with an Earth Engine service account
1. Google Cloud project with the Earth Engine API enabled -> create a service account -> register it for
   Earth Engine -> create a JSON key.
2. Store the **content** of the key as a secret named `EE_SERVICE_ACCOUNT_JSON` on the hosting platform
   (Hugging Face Spaces: Settings -> Variables and secrets; Cloud Run: Secret Manager).
3. **Never commit the key** (it is in `.gitignore`), never paste it in notebooks, screenshots or chat. Give the
   service account only the Earth Engine role, and delete the key after the course if it is no longer needed.

A live run takes about 1-3 min on a CPU (Earth Engine fetch + 12 model passes; the comparison doubles the model
part). The queue runs one job at a time to protect the free CPU and the Earth Engine quota.

## Limits (honest)
- Trained on 2021 only (Sentinel-1A + 1B). Other years are not validated.
- Steep mountains (> 15 deg) test mIoU 0.39; flooded grasslands 0.37.
- Dark smooth sand can look like water; snow/ice is not one of the 7 classes; rivers narrower than ~30 m are
  missed (Sentinel-1 GRD resolution ~20 x 22 m, resampled to 10 m pixels).
- Open ocean has no Sentinel-1 IW acquisitions: no maps.
- `agree_WC_%` is agreement with ESA WorldCover 2021, not accuracy (WorldCover itself is ~77% accurate).
- The low-confidence cut-off (0.65) and the change thresholds are heuristics measured for the final model;
  for comparison models they are indicative only.

## Data and licences
- Copernicus Sentinel-1 data (2021), processed by ESA; accessed via Google Earth Engine.
- ESA WorldCover 10 m 2021 v200, CC BY 4.0.
- Dynamic World V1 (Google and World Resources Institute), CC BY 4.0 - used only as independent reference.
- Basemaps: (c) OpenStreetMap contributors; Esri World Imagery (Esri, Maxar, Earthstar Geographics).
- Place search: OpenStreetMap Nominatim and Photon (komoot) - the search text is sent to these services.
