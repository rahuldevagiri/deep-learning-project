"""Paths, frozen configuration (app_config.json) and the registry of selectable models."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("LULC_CONFIG", ROOT / "config" / "app_config.json"))
WEIGHTS_DIR = Path(os.environ.get("LULC_WEIGHTS_DIR", ROOT / "weights"))      # <run>/best.pth
CACHE_DIR = Path(os.environ.get("LULC_CACHE_DIR", ROOT / "cache"))            # fetched Sentinel-1 stacks
EE_PROJECT = os.environ.get("EE_PROJECT", "deep-learning-27")

CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PRE = CFG["preprocessing"]
CLASSES = CFG["classes"]
# Display palette (softer than the WorldCover colours in app_config.json, still conventional: green = tree,
# red = built-up, blue = water) chosen so the 7 classes are easy to tell apart on maps and on white charts.
COLORS = ["#2E7D32", "#A4B93A", "#E377C2", "#EE7B5F", "#B8A88A", "#2F80ED", "#2BB3A6"]
COLORS_WORLDCOVER = CFG["colors"]
NC = len(CLASSES)
NODATA = PRE["nodata"]
RES = 10            # metres per pixel (UTM grid, same as training)
TILE = 256          # model window = training patch size
THR_M2M = CFG["change_thresholds_pp"]["month_to_month_p95"]
THR_FL = CFG["change_thresholds_pp"]["first_to_last_p95"]
AREA_ERR = CFG["area_error_pp"]["mean"]
CONF_MIN = 0.65     # low-reliability month: mean calibrated confidence below this (demo heuristic)

# Selectable models. val_mIoU = M4 validation (seed 42 unless stated); only FINAL has a test score,
# a fitted temperature and validated change thresholds.
MODELS = {
    "final": dict(
        label="SegFormer-B2 temporal | 3-seed ensemble + calibration (FINAL)",
        arch="segformer", temporal=True, temperature=CFG["model"]["temperature"], calibrated=True,
        members=CFG["model"]["members"], val_mIoU=0.5569, test_mIoU=CFG["test_performance"]["mIoU_all_test"][0]),
    "segformer_t3_s42": dict(
        label="SegFormer-B2 temporal | single seed 42", arch="segformer", temporal=True,
        temperature=1.0, calibrated=False, members=["segformer_b2_t3_s42"], val_mIoU=0.5491, test_mIoU=None),
    "deeplab_t3_s42": dict(
        label="DeepLabV3+ R34 temporal | single seed 42", arch="deeplab", temporal=True,
        temperature=1.0, calibrated=False, members=["deeplabv3p_r34_t3_s42"], val_mIoU=0.5220, test_mIoU=None),
    "segformer_s42": dict(
        label="SegFormer-B2 single month | seed 42", arch="segformer", temporal=False,
        temperature=1.0, calibrated=False, members=["segformer_b2_s42"], val_mIoU=0.5058, test_mIoU=None),
    "deeplab_s42": dict(
        label="DeepLabV3+ R34 single month | seed 42", arch="deeplab", temporal=False,
        temperature=1.0, calibrated=False, members=["deeplabv3p_r34_s42"], val_mIoU=0.4914, test_mIoU=None),
    "deeplab_scratch_s42": dict(
        label="DeepLabV3+ R34 single month | no pre-training | seed 42", arch="deeplab", temporal=False,
        temperature=1.0, calibrated=False, members=["deeplabv3p_r34_scratch_s42"], val_mIoU=0.4712, test_mIoU=None),
}


def available_models():
    """Keys of models whose weight files are all present."""
    return [k for k, m in MODELS.items()
            if all((WEIGHTS_DIR / r / "best.pth").exists() for r in m["members"])]
